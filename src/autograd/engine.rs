use super::{
    Definition, GradientStatus, Kind, Program, ReadValue, Resources, Seed, Value, backend::Backend,
};
use crate::{
    Error, GradientMode, Layout, ReadGuard, Result, SavedValue, Shared, StorageVersion, Trackable,
    error::invalid,
};
use std::sync::Arc;

pub(crate) struct Gradient<T> {
    pub data: T,
    pub leaf: bool,
    pub owner_root: usize,
    pub initialized: bool,
}
pub(super) struct Edge<B: Backend> {
    pub plan: B::Backward,
    pub target: usize,
    pub layout: Arc<Layout>,
    pub dependencies: [bool; 3],
    pub contributes: bool,
    pub scratch: Option<B::Tensor>,
    pub projection: Option<B::Projection>,
}
pub(super) struct Step<B: Backend> {
    pub output: usize,
    pub inputs: [Option<usize>; 3],
    pub forward: B::Forward,
    pub gradient: Option<usize>,
    pub edges: Vec<Edge<B>>,
    pub dependencies: [bool; 3],
    pub saved: [Option<SavedValue<B::Tensor>>; 3],
}
pub(super) struct Root<B: Backend> {
    pub layout: Arc<Layout>,
    pub gradient: Option<usize>,
    pub projection: Option<B::Projection>,
}
pub(crate) struct Core<B: Backend> {
    pub(super) backend: B,
    pub(super) program: Program,
    pub(super) owners: Vec<Option<Shared<B::Tensor>>>,
    pub(super) gradients: Vec<Gradient<B::Tensor>>,
    pub(super) value_gradients: Vec<Option<usize>>,
    pub(super) steps: Vec<Step<B>>,
    pub(super) roots: Vec<Root<B>>,
    pub(super) forward_versions: Vec<Option<StorageVersion>>,
    pub(super) scalar_seed: Arc<B::Tensor>,
    pub(super) selected: Vec<bool>,
    pub(super) forward_ready: bool,
    pub(super) update_ready: bool,
    pub(super) gradient_state: GradientStatus,
    pub(crate) resources: Resources,
}

pub(super) fn read_value<'a, B: Backend>(
    backend: &B,
    owners: &'a [Option<Shared<B::Tensor>>],
    values: &[Definition],
    index: usize,
) -> Result<ReadGuard<'a, B::Tensor>> {
    let root = values[index].root;
    let read = owners[root]
        .as_ref()
        .ok_or_else(|| invalid("training value has no prepared storage"))?
        .read()?;
    backend.check(&read)?;
    if read.shape() != values[root].layout.shape() {
        return Err(invalid("bound training storage shape changed"));
    }
    Ok(read)
}
pub(super) fn check_version(current: StorageVersion, saved: StorageVersion) -> Result<()> {
    if current.storage_id() != saved.storage_id() {
        Err(Error::StorageIdentityMismatch {
            saved: saved.storage_id(),
            current: current.storage_id(),
        })
    } else if current.generation() != saved.generation() {
        Err(Error::StorageModified {
            storage: saved.storage_id(),
            saved: saved.generation(),
            current: current.generation(),
        })
    } else {
        Ok(())
    }
}
fn read_write<T>(values: &mut [T], read: usize, write: usize) -> Result<(&T, &mut T)> {
    if read == write {
        return Err(invalid(
            "internal gradient dependency aliases its own output",
        ));
    }
    if read < write {
        let (left, right) = values.split_at_mut(write);
        Ok((&left[read], &mut right[0]))
    } else {
        let (left, right) = values.split_at_mut(read);
        Ok((&right[0], &mut left[write]))
    }
}
fn mode(initialized: bool) -> GradientMode {
    if initialized {
        GradientMode::Accumulate
    } else {
        GradientMode::Overwrite
    }
}

impl<B: Backend> Core<B> {
    pub(crate) fn forward(&mut self) -> Result<()> {
        self.forward_ready = false;
        self.update_ready = false;
        for step in &mut self.steps {
            step.saved = [None, None, None];
        }
        let graph = &self.program.0;
        for leaf in &graph.leaves {
            self.forward_versions[leaf.index] =
                Some(read_value(&self.backend, &self.owners, &graph.values, leaf.index)?.version());
        }
        for step in &mut self.steps {
            let a = step.inputs[0]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            let b = step.inputs[1]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            let c = step.inputs[2]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            // Capture while the actual forward readers exclude mutation. Each node
            // retains its own versions, even when a later branch reads the same owner.
            for operand in 0..3 {
                if step.dependencies[operand] {
                    let i = step.inputs[operand].unwrap();
                    step.saved[operand] = Some(
                        self.owners[graph.values[i].root]
                            .as_ref()
                            .unwrap()
                            .save(&graph.values[i].layout)?,
                    );
                }
            }
            let owner = self.owners[step.output].as_ref().unwrap();
            {
                let mut output = owner.write()?;
                let input = [
                    a.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[0].unwrap()].layout)),
                    b.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[1].unwrap()].layout)),
                    c.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[2].unwrap()].layout)),
                ];
                self.backend.forward(
                    &mut step.forward,
                    input,
                    (&mut output, &graph.values[step.output].layout),
                )?;
            }
            self.forward_versions[step.output] = Some(owner.version()?);
        }
        for leaf in &graph.leaves {
            let read = read_value(&self.backend, &self.owners, &graph.values, leaf.index)?;
            check_version(read.version(), self.forward_versions[leaf.index].unwrap())?;
        }
        self.forward_ready = true;
        Ok(())
    }
    pub(crate) fn output(&self, value: Value) -> Result<ReadValue<'_, B::Tensor>> {
        let index = self.program.index(value)?;
        if !self.forward_ready {
            return Err(invalid("training outputs require a successful forward"));
        }
        if !self.program.roots().contains(&value) {
            return Err(invalid("value is not a declared output root"));
        }
        let definition = &self.program.0.values[index];
        let storage = read_value(&self.backend, &self.owners, &self.program.0.values, index)?;
        check_version(
            storage.version(),
            self.forward_versions[definition.root].unwrap(),
        )?;
        Ok(ReadValue {
            storage,
            layout: &definition.layout,
        })
    }
    pub(super) fn leaf_gradient(&self, leaf: Value) -> Result<Option<usize>> {
        let i = self.program.index(leaf)?;
        if !matches!(self.program.0.values[i].kind, Kind::Leaf) {
            return Err(Error::Unsupported(
                "only declared training leaves expose gradients".into(),
            ));
        }
        Ok(self.value_gradients[i])
    }
    pub(crate) fn gradient_status(&self, leaf: Value) -> Result<GradientStatus> {
        Ok(if self.leaf_gradient(leaf)?.is_some() {
            self.gradient_state
        } else {
            GradientStatus::Unused
        })
    }
    pub(crate) fn gradient(&self, leaf: Value) -> Result<Option<&B::Tensor>> {
        let Some(i) = self.leaf_gradient(leaf)? else {
            return Ok(None);
        };
        if self.gradient_state != GradientStatus::Valid {
            return Err(invalid("leaf gradient is not valid"));
        }
        Ok(Some(&self.gradients[i].data))
    }
    pub(crate) fn backward_scalar(&mut self, mode: GradientMode) -> Result<()> {
        if self.roots.len() != 1
            || !self.roots[0].layout.shape().dims().is_empty()
            || self.roots[0].layout.offset() != 0
        {
            return Err(invalid(
                "implicit backward seed requires one root with the default scalar seed layout",
            ));
        }
        let layout = Arc::clone(&self.roots[0].layout);
        let value = self.program.roots()[0];
        let one = Arc::clone(&self.scalar_seed);
        self.backward(&[Seed::new(value, &*one, &layout)], mode)
    }
    fn validate_backward(&self, seeds: &[Seed<'_, B::Tensor>], mode: GradientMode) -> Result<()> {
        if !self.forward_ready {
            return Err(invalid("backward requires a successful forward"));
        }
        if mode == GradientMode::Accumulate && self.gradient_state == GradientStatus::Invalid {
            return Err(invalid(
                "failed gradients require zeroing or a successful overwrite",
            ));
        }
        if seeds.len() != self.roots.len() {
            return Err(invalid(
                "one explicit seed is required for every declared root",
            ));
        }
        for ((seed, root), expected) in seeds.iter().zip(&self.roots).zip(self.program.roots()) {
            if seed.value != *expected || seed.layout != &*root.layout {
                return Err(invalid("root seed identity or prepared layout differs"));
            }
            self.backend.check(seed.tensor)?;
            root.layout.validate_storage(seed.tensor.shape().len())?;
        }
        for step in &self.steps {
            for value in step.saved.iter().flatten() {
                let read = value.read()?;
                self.backend.check(read.storage())?;
            }
        }
        Ok(())
    }
    pub(crate) fn backward(
        &mut self,
        seeds: &[Seed<'_, B::Tensor>],
        requested: GradientMode,
    ) -> Result<()> {
        self.validate_backward(seeds, requested)?;
        let preserve =
            requested == GradientMode::Accumulate && self.gradient_state == GradientStatus::Valid;
        for gradient in &mut self.gradients {
            gradient.initialized = gradient.leaf && preserve;
        }
        self.gradient_state = GradientStatus::Invalid;
        self.update_ready = false;
        // A failure below leaves state invalid, including unwinding after a partial
        // write. No caller can inspect or apply partially accumulated gradients.
        for (seed, root) in seeds.iter().zip(&self.roots) {
            if let Some(target) = root.gradient {
                let gradient = &mut self.gradients[target];
                self.backend.project(
                    root.projection.as_ref().unwrap(),
                    (seed.tensor, seed.layout),
                    &mut gradient.data,
                    mode(gradient.initialized),
                )?;
                gradient.initialized = true;
            }
        }
        let definitions = &self.program.0.values;
        for step in self.steps.iter_mut().rev() {
            let Some(upstream_index) = step.gradient else {
                continue;
            };
            if !self.gradients[upstream_index].initialized {
                return Err(invalid(
                    "internal gradient was not initialized by a root or consumer",
                ));
            }
            for edge in &mut step.edges {
                if !edge.contributes {
                    let target = &mut self.gradients[edge.target];
                    if !target.initialized {
                        self.backend.clear(&mut target.data)?;
                        target.initialized = true;
                    }
                    continue;
                }
                let a = if edge.dependencies[0] {
                    Some(step.saved[0].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let b = if edge.dependencies[1] {
                    Some(step.saved[1].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let c = if edge.dependencies[2] {
                    Some(step.saved[2].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let inputs = [
                    a.as_ref().map(|v| (v.storage(), v.layout())),
                    b.as_ref().map(|v| (v.storage(), v.layout())),
                    c.as_ref().map(|v| (v.storage(), v.layout())),
                ];
                let upstream_layout = &definitions[step.output].layout;
                if let Some(scratch) = edge.scratch.as_mut() {
                    self.backend.backward(
                        &mut edge.plan,
                        inputs,
                        (&self.gradients[upstream_index].data, upstream_layout),
                        (scratch, &edge.layout),
                        GradientMode::Overwrite,
                    )?;
                    let target = &mut self.gradients[edge.target];
                    self.backend.project(
                        edge.projection.as_ref().unwrap(),
                        (scratch, &edge.layout),
                        &mut target.data,
                        mode(target.initialized),
                    )?;
                    target.initialized = true;
                } else {
                    let (upstream, target) =
                        read_write(&mut self.gradients, upstream_index, edge.target)?;
                    self.backend.backward(
                        &mut edge.plan,
                        inputs,
                        (&upstream.data, upstream_layout),
                        (&mut target.data, &edge.layout),
                        mode(target.initialized),
                    )?;
                    target.initialized = true;
                }
            }
        }
        if self.gradients.iter().any(|g| !g.initialized) {
            return Err(invalid("a reachable gradient has no computed contribution"));
        }
        self.gradient_state = GradientStatus::Valid;
        self.update_ready = true;
        Ok(())
    }
    pub(crate) fn zero_grad(&mut self) -> Result<()> {
        self.gradient_state = GradientStatus::Invalid;
        self.update_ready = false;
        for gradient in &mut self.gradients {
            if gradient.leaf {
                self.backend.clear(&mut gradient.data)?;
            }
        }
        self.gradient_state = GradientStatus::Valid;
        Ok(())
    }
    pub(crate) fn synchronize(&mut self) -> Result<()> {
        let result = self.backend.synchronize();
        if result.is_err() {
            self.forward_ready = false;
            self.update_ready = false;
            self.gradient_state = GradientStatus::Invalid;
        }
        result
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn backend(&self) -> &B {
        &self.backend
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn program(&self) -> &Program {
        &self.program
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn leaf_owner(&self, root: usize) -> &Shared<B::Tensor> {
        self.owners[root].as_ref().unwrap()
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn value_owner(&self, root: usize) -> &Shared<B::Tensor> {
        self.owners[root].as_ref().unwrap()
    }
    pub(crate) fn scalar_seed_info(&self) -> Result<(Value, Arc<B::Tensor>, Arc<Layout>)> {
        if self.roots.len() != 1
            || !self.roots[0].layout.shape().dims().is_empty()
            || self.roots[0].layout.offset() != 0
        {
            return Err(invalid(
                "implicit backward seed requires one root with the default scalar seed layout",
            ));
        }
        Ok((
            self.program.roots()[0],
            Arc::clone(&self.scalar_seed),
            Arc::clone(&self.roots[0].layout),
        ))
    }
    pub(crate) fn validate_seeds_for_capture(&self, seeds: &[Seed<'_, B::Tensor>]) -> Result<()> {
        if seeds.len() != self.roots.len() {
            return Err(invalid(
                "one explicit seed is required for every declared root",
            ));
        }
        for ((seed, root), expected) in seeds.iter().zip(&self.roots).zip(self.program.roots()) {
            if seed.value != *expected || seed.layout != &*root.layout {
                return Err(invalid("root seed identity or prepared layout differs"));
            }
            self.backend.check(seed.tensor)?;
            root.layout.validate_storage(seed.tensor.shape().len())?;
        }
        Ok(())
    }
    pub(crate) fn select_updated_slots(&mut self, leaves: &[Value]) -> Result<&[bool]> {
        self.selected.fill(false);
        for &leaf in leaves {
            let slot = self
                .leaf_gradient(leaf)?
                .ok_or_else(|| invalid("SGD selected an unused or non-gradient leaf"))?;
            self.selected[slot] = true;
        }
        Ok(&self.selected)
    }
    pub(crate) fn analyze_fusion(
        &self,
        updated_slots: &[bool],
    ) -> Option<super::fusion::FusedStepPlan> {
        super::fusion::analyze_step_fusion(&self.program.0, &self.value_gradients, updated_slots)
    }
    pub(crate) fn preflight_region(&self, region: &super::fusion::RegionPlan) -> Result<()> {
        let values = &self.program.0.values;
        for ext in &region.exts {
            let owner = self.owners[ext.owner_root]
                .as_ref()
                .ok_or_else(|| invalid("training value has no prepared storage"))?;
            {
                let read = owner.read()?;
                self.backend.check(&read)?;
                if read.shape() != values[ext.owner_root].layout.shape() {
                    return Err(invalid("bound training storage shape changed"));
                }
            }
            if ext.update_sgd {
                owner.check_writable()?;
            }
        }
        Ok(())
    }
    pub(crate) fn forward_up_to(&mut self, last_step: usize) -> Result<()> {
        self.forward_ready = false;
        self.update_ready = false;
        for step in &mut self.steps {
            step.saved = [None, None, None];
        }
        let graph = &self.program.0;
        for leaf in &graph.leaves {
            self.forward_versions[leaf.index] =
                Some(read_value(&self.backend, &self.owners, &graph.values, leaf.index)?.version());
        }
        for step in &mut self.steps[..=last_step] {
            let a = step.inputs[0]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            let b = step.inputs[1]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            let c = step.inputs[2]
                .map(|i| read_value(&self.backend, &self.owners, &graph.values, i))
                .transpose()?;
            for operand in 0..3 {
                if step.dependencies[operand] {
                    let i = step.inputs[operand].unwrap();
                    step.saved[operand] = Some(
                        self.owners[graph.values[i].root]
                            .as_ref()
                            .unwrap()
                            .save(&graph.values[i].layout)?,
                    );
                }
            }
            let owner = self.owners[step.output].as_ref().unwrap();
            {
                let mut output = owner.write()?;
                let input = [
                    a.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[0].unwrap()].layout)),
                    b.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[1].unwrap()].layout)),
                    c.as_ref()
                        .map(|t| (&**t, &*graph.values[step.inputs[2].unwrap()].layout)),
                ];
                self.backend.forward(
                    &mut step.forward,
                    input,
                    (&mut output, &graph.values[step.output].layout),
                )?;
            }
            self.forward_versions[step.output] = Some(owner.version()?);
        }
        Ok(())
    }
    pub(crate) fn backward_from(
        &mut self,
        last_step: usize,
        preinitialized_slots: &[usize],
    ) -> Result<()> {
        for (idx, gradient) in self.gradients.iter_mut().enumerate() {
            gradient.initialized = preinitialized_slots.contains(&idx);
        }
        self.gradient_state = GradientStatus::Invalid;
        self.update_ready = false;
        for step in &self.steps[..=last_step] {
            for value in step.saved.iter().flatten() {
                let read = value.read()?;
                self.backend.check(read.storage())?;
            }
        }
        let definitions = &self.program.0.values;
        for step in self.steps[..=last_step].iter_mut().rev() {
            let Some(upstream_index) = step.gradient else {
                continue;
            };
            if !self.gradients[upstream_index].initialized {
                return Err(invalid(
                    "internal gradient was not initialized by a root or consumer",
                ));
            }
            for edge in &mut step.edges {
                if !edge.contributes {
                    let target = &mut self.gradients[edge.target];
                    if !target.initialized {
                        self.backend.clear(&mut target.data)?;
                        target.initialized = true;
                    }
                    continue;
                }
                let a = if edge.dependencies[0] {
                    Some(step.saved[0].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let b = if edge.dependencies[1] {
                    Some(step.saved[1].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let c = if edge.dependencies[2] {
                    Some(step.saved[2].as_ref().unwrap().read()?)
                } else {
                    None
                };
                let inputs = [
                    a.as_ref().map(|v| (v.storage(), v.layout())),
                    b.as_ref().map(|v| (v.storage(), v.layout())),
                    c.as_ref().map(|v| (v.storage(), v.layout())),
                ];
                let upstream_layout = &definitions[step.output].layout;
                if let Some(scratch) = edge.scratch.as_mut() {
                    self.backend.backward(
                        &mut edge.plan,
                        inputs,
                        (&self.gradients[upstream_index].data, upstream_layout),
                        (scratch, &edge.layout),
                        GradientMode::Overwrite,
                    )?;
                    let target = &mut self.gradients[edge.target];
                    self.backend.project(
                        edge.projection.as_ref().unwrap(),
                        (scratch, &edge.layout),
                        &mut target.data,
                        mode(target.initialized),
                    )?;
                    target.initialized = true;
                } else {
                    let (upstream, target) =
                        read_write(&mut self.gradients, upstream_index, edge.target)?;
                    self.backend.backward(
                        &mut edge.plan,
                        inputs,
                        (&upstream.data, upstream_layout),
                        (&mut target.data, &edge.layout),
                        mode(target.initialized),
                    )?;
                    target.initialized = true;
                }
            }
        }
        self.gradient_state = GradientStatus::Valid;
        self.update_ready = true;
        Ok(())
    }
    pub(crate) fn sgd_selected_except(
        &mut self,
        rate: f32,
        updated_in_region: &[bool],
    ) -> Result<()> {
        for (slot, (&selected, gradient)) in self.selected.iter().zip(&self.gradients).enumerate() {
            if !selected || updated_in_region.get(slot).copied().unwrap_or(false) {
                continue;
            }
            let root = gradient.owner_root;
            {
                let parameter =
                    read_value(&self.backend, &self.owners, &self.program.0.values, root)?;
                check_version(parameter.version(), self.forward_versions[root].unwrap())?;
            }
            self.owners[root].as_ref().unwrap().check_writable()?;
        }
        self.forward_ready = false;
        self.update_ready = false;
        self.gradient_state = GradientStatus::Invalid;
        for (slot, (&selected, gradient)) in self.selected.iter().zip(&self.gradients).enumerate() {
            if !selected || updated_in_region.get(slot).copied().unwrap_or(false) {
                continue;
            }
            let root = gradient.owner_root;
            let mut parameter = self.owners[root]
                .as_ref()
                .unwrap()
                .write_checked(self.forward_versions[root].unwrap())?;
            self.backend.update(&mut parameter, &gradient.data, rate)?;
        }
        self.gradient_state = GradientStatus::Valid;
        Ok(())
    }
    #[allow(clippy::type_complexity)]
    pub(crate) fn owners_and_gradients_mut(
        &mut self,
    ) -> (&[Option<Shared<B::Tensor>>], &mut [Gradient<B::Tensor>]) {
        (&self.owners, &mut self.gradients)
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn gradient_mut(&mut self, slot: usize) -> &mut B::Tensor {
        &mut self.gradients[slot].data
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn capture_leaf_roots(&mut self, leaves: &[Value]) -> Result<Vec<(usize, bool)>> {
        self.selected.fill(false);
        for &leaf in leaves {
            let slot = self
                .leaf_gradient(leaf)?
                .ok_or_else(|| invalid("SGD selected an unused or non-gradient leaf"))?;
            self.selected[slot] = true;
        }
        let mut updated_roots = Vec::new();
        for (selected, gradient) in self.selected.iter().zip(&self.gradients) {
            if *selected {
                updated_roots.push(gradient.owner_root);
            }
        }
        let mut out = Vec::with_capacity(self.program.0.leaves.len());
        for leaf in &self.program.0.leaves {
            let root = self.program.0.values[leaf.index].root;
            if out.iter().any(|&(r, _)| r == root) {
                continue;
            }
            out.push((root, updated_roots.contains(&root)));
        }
        Ok(out)
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn matches_updated_leaves(
        &mut self,
        leaves: &[Value],
        expected_roots: &[usize],
    ) -> Result<bool> {
        self.selected.fill(false);
        for &leaf in leaves {
            let slot = self
                .leaf_gradient(leaf)?
                .ok_or_else(|| invalid("SGD selected an unused or non-gradient leaf"))?;
            self.selected[slot] = true;
        }
        let mut matched = 0usize;
        for (selected, gradient) in self.selected.iter().zip(&self.gradients) {
            if !*selected {
                continue;
            }
            if !expected_roots.contains(&gradient.owner_root) {
                return Ok(false);
            }
            matched += 1;
        }
        Ok(matched == expected_roots.len())
    }
    pub(crate) fn mark_captured_step_done(&mut self, ok: bool) {
        self.forward_ready = false;
        self.update_ready = false;
        self.gradient_state = if ok {
            GradientStatus::Valid
        } else {
            GradientStatus::Invalid
        };
    }
    pub(crate) fn record_leaf_versions(&mut self) -> Result<()> {
        let graph = &self.program.0;
        for leaf in &graph.leaves {
            self.forward_versions[leaf.index] =
                Some(read_value(&self.backend, &self.owners, &graph.values, leaf.index)?.version());
        }
        Ok(())
    }
    pub(crate) fn mark_gradients_ready(&mut self) {
        self.gradient_state = GradientStatus::Valid;
        self.update_ready = true;
    }
    pub(crate) fn validate_optimizer_leaves(
        &self,
        leaves: &[Value],
    ) -> Result<Vec<(usize, usize)>> {
        if !self.update_ready || self.gradient_state != GradientStatus::Valid {
            return Err(invalid(
                "optimizer step requires valid gradients from backward",
            ));
        }
        let mut pairs = Vec::with_capacity(leaves.len());
        for &leaf in leaves {
            let slot = self
                .leaf_gradient(leaf)?
                .ok_or_else(|| invalid("optimizer selected an unused or non-gradient leaf"))?;
            let root = self.gradients[slot].owner_root;
            {
                let parameter =
                    read_value(&self.backend, &self.owners, &self.program.0.values, root)?;
                check_version(parameter.version(), self.forward_versions[root].unwrap())?;
            }
            self.owners[root].as_ref().unwrap().check_writable()?;
            pairs.push((root, slot));
        }
        Ok(pairs)
    }
    pub(crate) fn apply_optimizer_update<F>(&mut self, leaves: &[Value], update: F) -> Result<()>
    where
        F: FnOnce(&mut [&mut B::Tensor], &[&B::Tensor]) -> Result<()>,
    {
        let pairs = self.validate_optimizer_leaves(leaves)?;
        self.forward_ready = false;
        self.update_ready = false;
        self.gradient_state = GradientStatus::Invalid;
        let mut guards = Vec::with_capacity(pairs.len());
        for &(root, _) in &pairs {
            let expected = self.forward_versions[root].unwrap();
            guards.push(
                self.owners[root]
                    .as_ref()
                    .unwrap()
                    .write_checked(expected)?,
            );
        }
        let mut param_refs: Vec<&mut B::Tensor> = guards.iter_mut().map(|g| &mut **g).collect();
        let grad_refs: Vec<&B::Tensor> = pairs
            .iter()
            .map(|&(_, slot)| &self.gradients[slot].data)
            .collect();
        let result = update(&mut param_refs, &grad_refs);
        if result.is_ok() {
            self.gradient_state = GradientStatus::Valid;
        }
        result
    }
}
