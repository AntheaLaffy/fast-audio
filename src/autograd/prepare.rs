use super::{
    Binding, GradientStatus, Kind, PrepareOptions, Program, Resources,
    backend::{Backend, PlanMemory},
    engine::{Core, Edge, Gradient, Root, Step},
};
use crate::{Layout, Result, Shared, Trackable, error::invalid};
use std::{collections::HashMap, sync::Arc};

fn add(target: &mut usize, bytes: usize) -> Result<()> {
    *target = target
        .checked_add(bytes)
        .ok_or_else(|| invalid("prepared training resource total overflows"))?;
    Ok(())
}
fn account(resources: &mut Resources, plan: PlanMemory) -> Result<()> {
    add(&mut resources.operator_workspace_bytes, plan.workspace)?;
    add(&mut resources.operator_host_metadata_bytes, plan.metadata)?;
    add(&mut resources.view_index_bytes, plan.indices)?;
    resources.max_static_shared_bytes_per_block =
        resources.max_static_shared_bytes_per_block.max(plan.shared);
    Ok(())
}
fn projection<B: Backend>(
    backend: &B,
    base: &crate::Shape,
    view: &Layout,
    upstream: &Layout,
    remaining: &mut usize,
    resources: &mut Resources,
) -> Result<B::Projection> {
    let plan = backend.project_plan(base, view, upstream, *remaining)?;
    let memory = B::project_memory(&plan);
    *remaining = remaining
        .checked_sub(memory.indices)
        .ok_or_else(|| invalid("combined view index budget exceeded"))?;
    account(resources, memory)?;
    Ok(plan)
}
impl<B: Backend> Core<B> {
    pub(crate) fn prepare(
        backend: B,
        program: &Program,
        bindings: &[Binding<B::Tensor>],
        options: PrepareOptions<'_>,
    ) -> Result<Self> {
        let graph = &program.0;
        let mut owners = vec![None; graph.values.len()];
        if bindings.len() != graph.leaves.len() {
            return Err(invalid("bind every declared leaf exactly once"));
        }
        for binding in bindings {
            let i = program.index(binding.value)?;
            if !matches!(graph.values[i].kind, Kind::Leaf) || owners[i].is_some() {
                return Err(invalid("training binding is duplicated or not a leaf"));
            }
            {
                let read = binding.owner.read()?;
                backend.check(&read)?;
                if read.shape() != graph.values[i].layout.shape() {
                    return Err(invalid("training leaf binding shape differs"));
                }
            }
            owners[i] = Some(binding.owner.clone());
        }
        if graph.leaves.iter().any(|v| owners[v.index].is_none()) {
            return Err(invalid("training leaf binding is missing"));
        }
        let mut resources = Resources::default();
        let mut gradients = Vec::new();
        let mut value_gradients = vec![None; graph.values.len()];
        let mut canonical = HashMap::new();
        for (i, value) in graph.values.iter().enumerate() {
            if !graph.gradient_roots[i] {
                continue;
            }
            let leaf = matches!(value.kind, Kind::Leaf);
            let identity = if leaf {
                Some(owners[i].as_ref().unwrap().version()?.storage_id())
            } else {
                None
            };
            if let Some(existing) = identity.and_then(|id| canonical.get(&id).copied()) {
                value_gradients[i] = Some(existing);
                continue;
            }
            let slot = gradients.len();
            add(
                &mut resources.gradient_bytes,
                value.layout.shape().len() * size_of::<f32>(),
            )?;
            gradients.push(Gradient {
                data: backend.zeros(value.layout.shape())?,
                leaf,
                owner_root: i,
                initialized: false,
            });
            value_gradients[i] = Some(slot);
            if let Some(id) = identity {
                canonical.insert(id, slot);
            }
        }
        for leaf in &graph.leaves {
            if graph.values[leaf.index].requires_grad {
                let id = owners[leaf.index].as_ref().unwrap().version()?.storage_id();
                value_gradients[leaf.index] = canonical.get(&id).copied();
            }
        }
        let mut remaining = options.view_index_limit_bytes;
        let mut steps = Vec::with_capacity(graph.forward.len());
        for &i in &graph.forward {
            let Kind::Compute { operation, inputs } = &graph.values[i].kind else {
                unreachable!()
            };
            let layouts = inputs.map(|input| input.map(|n| &*graph.values[n].layout));
            let forward = backend.forward_plan(operation, layouts, &graph.values[i].layout)?;
            account(&mut resources, B::forward_memory(&forward))?;
            let mut edges = Vec::new();
            let mut dependencies = [false; 3];
            if graph.gradient_roots[i] {
                for (operand, input) in inputs.iter().enumerate() {
                    let Some(input) = *input else {
                        continue;
                    };
                    let value = &graph.values[input];
                    if !value.requires_grad {
                        continue;
                    }
                    let target = value_gradients[value.root].unwrap();
                    let layout = Arc::new(Layout::contiguous(value.layout.shape().dims())?);
                    let plan = backend.backward_plan(
                        operation,
                        operand,
                        layouts,
                        &graph.values[i].layout,
                        &layout,
                    )?;
                    account(&mut resources, B::backward_memory(&plan))?;
                    let edge_dependencies = B::dependencies(&plan);
                    for j in 0..3 {
                        dependencies[j] |= edge_dependencies[j];
                    }
                    let base = graph.values[value.root].layout.shape();
                    let direct = value.layout.offset() == 0
                        && value.layout.is_contiguous()
                        && value.layout.shape().len() == base.len();
                    let contributes = !value.layout.shape().is_empty()
                        && !graph.values[i].layout.shape().is_empty();
                    let (scratch, projector) = if direct || !contributes {
                        (None, None)
                    } else {
                        let projector = projection(
                            &backend,
                            base,
                            &value.layout,
                            &layout,
                            &mut remaining,
                            &mut resources,
                        )?;
                        add(
                            &mut resources.edge_scratch_bytes,
                            layout.shape().len() * size_of::<f32>(),
                        )?;
                        (Some(backend.zeros(layout.shape())?), Some(projector))
                    };
                    edges.push(Edge {
                        plan,
                        target,
                        layout,
                        dependencies: edge_dependencies,
                        contributes,
                        scratch,
                        projection: projector,
                    });
                }
            }
            steps.push(Step {
                output: i,
                inputs: *inputs,
                forward,
                gradient: value_gradients[i],
                edges,
                dependencies,
                saved: [None, None, None],
            });
        }
        // Exact activation liveness analysis & storage reuse: any intermediate buffer that is
        // neither an output root nor saved for backward may be safely reused by strictly later
        // same-shape steps (free_after < pos prevents read-write aliasing and version conflicts).
        let mut death = vec![0usize; graph.values.len()];
        for (pos, step) in steps.iter().enumerate() {
            death[step.output] = pos;
        }
        for (pos, step) in steps.iter().enumerate() {
            for (operand, input) in step.inputs.iter().enumerate() {
                if let Some(v) = *input {
                    let root = graph.values[v].root;
                    if step.dependencies[operand] {
                        death[root] = usize::MAX;
                    } else if death[root] < pos {
                        death[root] = pos;
                    }
                }
            }
        }
        for root_val in &graph.roots {
            let root = graph.values[root_val.index].root;
            death[root] = usize::MAX;
        }
        struct ActivationSlot<T: Trackable> {
            shape: crate::Shape,
            shared: Shared<T>,
            free_after: usize,
        }
        let mut pool: Vec<ActivationSlot<B::Tensor>> = Vec::new();
        for (pos, step) in steps.iter().enumerate() {
            let i = step.output;
            let shape = graph.values[i].layout.shape();
            let slot_death = death[i];
            if let Some(slot) = pool
                .iter_mut()
                .find(|s| s.shape == *shape && s.free_after < pos)
            {
                slot.free_after = slot_death;
                owners[i] = Some(slot.shared.clone());
            } else {
                add(
                    &mut resources.activation_bytes,
                    shape.len() * size_of::<f32>(),
                )?;
                let shared = Shared::new(backend.zeros(shape)?)?;
                owners[i] = Some(shared.clone());
                pool.push(ActivationSlot {
                    shape: shape.clone(),
                    shared,
                    free_after: slot_death,
                });
            }
        }
        if let Some(layouts) = options.seed_layouts {
            if layouts.len() != graph.roots.len() {
                return Err(invalid("prepare one seed layout per output root"));
            }
        }
        let mut roots = Vec::with_capacity(graph.roots.len());
        for (position, value) in graph.roots.iter().enumerate() {
            let definition = &graph.values[value.index];
            let layout = Arc::new(if let Some(layouts) = options.seed_layouts {
                layouts[position].clone()
            } else {
                Layout::contiguous(definition.layout.shape().dims())?
            });
            if layout.shape() != definition.layout.shape() {
                return Err(invalid("root seed logical shape differs"));
            }
            let gradient = if definition.requires_grad {
                value_gradients[definition.root]
            } else {
                None
            };
            let projector = if gradient.is_some() {
                Some(projection(
                    &backend,
                    graph.values[definition.root].layout.shape(),
                    &definition.layout,
                    &layout,
                    &mut remaining,
                    &mut resources,
                )?)
            } else {
                None
            };
            roots.push(Root {
                layout,
                gradient,
                projection: projector,
            });
        }
        let scalar_seed = Arc::new(backend.scalar(1.)?);
        resources.scalar_seed_bytes = size_of::<f32>();
        let selected = vec![false; gradients.len()];
        let forward_versions = vec![None; graph.values.len()];
        Ok(Self {
            backend,
            program: program.clone(),
            owners,
            gradients,
            value_gradients,
            steps,
            roots,
            forward_versions,
            scalar_seed,
            selected,
            forward_ready: false,
            update_ready: false,
            gradient_state: GradientStatus::Uncomputed,
            resources,
        })
    }
}
