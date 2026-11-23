use super::{
    Binding, Operation, PrepareOptions, Program, Seed, Value,
    backend::{Backend, PlanMemory, Read, Write},
    engine::Core,
    fusion::{self, CpuFusedScratch, FusedStepPlan},
};
use crate::{
    BinaryOperand, GradientMode, Layout, Result, SdpaOperand, Shape, Tensor,
    cpu::{self, Cpu},
    error::invalid,
};

pub(crate) struct Host(pub Cpu);
pub(crate) enum Forward {
    Unary(Box<cpu::UnaryPlan>),
    Binary(Box<cpu::BinaryPlan>),
    Reduce(Box<cpu::ReducePlan>),
    Matmul(Box<cpu::MatmulPlan>),
    Bmm(Box<cpu::BatchedMatmulPlan>),
    Sdpa(Box<cpu::SdpaPlan>),
    Conv1d(Box<cpu::Conv1dPlan>),
    Conv2d(Box<cpu::Conv2dPlan>),
    ConvTranspose1d(Box<cpu::ConvTranspose1dPlan>),
    ConvTranspose2d(Box<cpu::ConvTranspose2dPlan>),
    Stft(Box<cpu::StftPlan>),
    CenteredStft(Box<cpu::CenteredStftPlan>),
    Istft(Box<cpu::IstftPlan>),
    Norm(Box<cpu::NormPlan>),
    Softmax(Box<cpu::SoftmaxPlan>),
    Rope(Box<cpu::RopePlan>),
    Pad(Box<cpu::PadPlan>),
    Cat(Box<cpu::CatPlan>),
}
pub(crate) enum Backward {
    Unary(Box<cpu::UnaryBackward>),
    Binary(Box<cpu::BinaryBackward>),
    Reduce(Box<cpu::ReduceBackward>),
    Matmul(Box<cpu::MatmulBackward>),
    Bmm(Box<cpu::BatchedMatmulBackward>),
    Sdpa(Box<cpu::SdpaBackward>),
    Conv1d(Box<cpu::Conv1dBackward>),
    Conv2d(Box<cpu::Conv2dBackward>),
    ConvTranspose1d(Box<cpu::ConvTranspose1dBackward>),
    ConvTranspose2d(Box<cpu::ConvTranspose2dBackward>),
    Stft(Box<cpu::StftBackward>),
    CenteredStft(Box<cpu::CenteredStftBackward>),
    Istft(Box<cpu::IstftBackward>),
    Norm(Box<cpu::NormBackward>),
    Softmax(Box<cpu::SoftmaxBackward>),
    Rope(Box<cpu::RopeBackward>),
    Pad(Box<cpu::PadBackward>),
    Cat(Box<cpu::CatBackward>),
}
fn memory(workspace: usize, metadata: usize) -> PlanMemory {
    PlanMemory {
        workspace,
        metadata,
        ..PlanMemory::default()
    }
}
impl Backend for Host {
    type Tensor = Tensor;
    type Forward = Forward;
    type Backward = Backward;
    type Projection = cpu::ViewBackward;
    fn check(&self, _: &Tensor) -> Result<()> {
        Ok(())
    }
    fn zeros(&self, shape: &Shape) -> Result<Tensor> {
        Tensor::zeros(shape.dims())
    }
    fn scalar(&self, value: f32) -> Result<Tensor> {
        Tensor::from_vec([], vec![value])
    }
    fn clear(&self, tensor: &mut Tensor) -> Result<()> {
        tensor.data_mut().fill(0.);
        Ok(())
    }
    fn update(&self, parameter: &mut Tensor, gradient: &Tensor, rate: f32) -> Result<()> {
        if parameter.shape() != gradient.shape() {
            return Err(invalid("parameter and gradient shapes differ"));
        }
        for (p, &g) in parameter.data_mut().iter_mut().zip(gradient.data()) {
            *p -= rate * g;
        }
        Ok(())
    }
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn forward_plan(
        &self,
        op: &Operation,
        inputs: [Option<&Layout>; 3],
        output: &Layout,
    ) -> Result<Forward> {
        let a = inputs[0].unwrap();
        Ok(match op {
            Operation::Unary(op) => Forward::Unary(Box::new(cpu::UnaryPlan::new(*op, a, output)?)),
            Operation::Binary(op) => Forward::Binary(Box::new(cpu::BinaryPlan::new(
                *op,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Reduce(spec) => {
                Forward::Reduce(Box::new(cpu::ReducePlan::new(spec, a, output)?))
            }
            Operation::Matmul => Forward::Matmul(Box::new(cpu::MatmulPlan::new(
                self.0,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Bmm(spec) => Forward::Bmm(Box::new(cpu::BatchedMatmulPlan::new(
                self.0,
                spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Sdpa(spec) => Forward::Sdpa(Box::new(cpu::SdpaPlan::new(
                *spec,
                a,
                inputs[1].unwrap(),
                inputs[2].unwrap(),
                output,
            )?)),
            Operation::Conv1d(spec) => Forward::Conv1d(Box::new(cpu::Conv1dPlan::new(
                self.0,
                *spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Conv2d(spec) => Forward::Conv2d(Box::new(cpu::Conv2dPlan::new(
                self.0,
                *spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::ConvTranspose1d(spec) => Forward::ConvTranspose1d(Box::new(
                cpu::ConvTranspose1dPlan::new(self.0, *spec, a, inputs[1].unwrap(), output)?,
            )),
            Operation::ConvTranspose2d(spec) => Forward::ConvTranspose2d(Box::new(
                cpu::ConvTranspose2dPlan::new(self.0, *spec, a, inputs[1].unwrap(), output)?,
            )),
            Operation::Stft { spec, window } => Forward::Stft(Box::new(
                cpu::StftPlan::with_window_slice(self.0, *spec, window.as_deref(), a, output)?,
            )),
            Operation::CenteredStft { spec, window } => {
                Forward::CenteredStft(Box::new(cpu::CenteredStftPlan::with_window_slice(
                    self.0,
                    *spec,
                    window.as_deref(),
                    a,
                    output,
                )?))
            }
            Operation::Istft { spec, window } => Forward::Istft(Box::new(
                cpu::IstftPlan::with_window_slice(self.0, *spec, window.as_deref(), a, output)?,
            )),
            Operation::Norm(spec) => Forward::Norm(Box::new(cpu::NormPlan::new(spec, a, output)?)),
            Operation::Softmax(spec) => {
                Forward::Softmax(Box::new(cpu::SoftmaxPlan::new(spec, a, output)?))
            }
            Operation::Rope(spec) => Forward::Rope(Box::new(cpu::RopePlan::new(spec, a, output)?)),
            Operation::Pad(spec) => Forward::Pad(Box::new(cpu::PadPlan::new(spec, a, output)?)),
            Operation::Cat(spec) => Forward::Cat(Box::new(cpu::CatPlan::new(
                spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
        })
    }
    fn backward_plan(
        &self,
        op: &Operation,
        operand: usize,
        inputs: [Option<&Layout>; 3],
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Backward> {
        let a = inputs[0].unwrap();
        let side = if operand == 0 {
            BinaryOperand::Left
        } else {
            BinaryOperand::Right
        };
        Ok(match op {
            Operation::Unary(op) => {
                Backward::Unary(Box::new(cpu::UnaryBackward::new(*op, a, upstream, output)?))
            }
            Operation::Binary(op) => Backward::Binary(Box::new(cpu::BinaryBackward::new(
                *op,
                side,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Reduce(spec) => Backward::Reduce(Box::new(cpu::ReduceBackward::with_input(
                spec, a, upstream, output,
            )?)),
            Operation::Matmul => Backward::Matmul(Box::new(cpu::MatmulBackward::new(
                self.0,
                side,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Bmm(spec) => Backward::Bmm(Box::new(cpu::BatchedMatmulBackward::new(
                self.0,
                side,
                spec,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Sdpa(spec) => {
                let sdpa_op = match operand {
                    0 => SdpaOperand::Query,
                    1 => SdpaOperand::Key,
                    _ => SdpaOperand::Value,
                };
                Backward::Sdpa(Box::new(cpu::SdpaBackward::new(
                    sdpa_op,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    inputs[2].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::Conv1d(spec) => Backward::Conv1d(Box::new(cpu::Conv1dBackward::new(
                self.0,
                side,
                *spec,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Conv2d(spec) => Backward::Conv2d(Box::new(cpu::Conv2dBackward::new(
                self.0,
                side,
                *spec,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::ConvTranspose1d(spec) => {
                Backward::ConvTranspose1d(Box::new(cpu::ConvTranspose1dBackward::new(
                    self.0,
                    side,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::ConvTranspose2d(spec) => {
                Backward::ConvTranspose2d(Box::new(cpu::ConvTranspose2dBackward::new(
                    self.0,
                    side,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::Stft { spec, window } => {
                Backward::Stft(Box::new(cpu::StftBackward::with_window_slice(
                    self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::CenteredStft { spec, window } => {
                Backward::CenteredStft(Box::new(cpu::CenteredStftBackward::with_window_slice(
                    self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::Istft { spec, window } => {
                Backward::Istft(Box::new(cpu::IstftBackward::with_window_slice(
                    self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::Norm(spec) => {
                Backward::Norm(Box::new(cpu::NormBackward::new(spec, a, upstream, output)?))
            }
            Operation::Softmax(spec) => Backward::Softmax(Box::new(cpu::SoftmaxBackward::new(
                spec, a, upstream, output,
            )?)),
            Operation::Rope(spec) => {
                Backward::Rope(Box::new(cpu::RopeBackward::new(spec, upstream, output)?))
            }
            Operation::Pad(spec) => {
                Backward::Pad(Box::new(cpu::PadBackward::new(spec, upstream, output)?))
            }
            Operation::Cat(spec) => Backward::Cat(Box::new(cpu::CatBackward::new(
                side, spec, upstream, output,
            )?)),
        })
    }
    fn dependencies(plan: &Backward) -> [bool; 3] {
        let pad = |d: [bool; 2]| [d[0], d[1], false];
        match plan {
            Backward::Unary(p) => [p.input_required(), false, false],
            Backward::Binary(p) => pad(p.input_dependencies()),
            Backward::Reduce(p) => [p.input_required(), false, false],
            Backward::Stft(_) | Backward::CenteredStft(_) | Backward::Istft(_) => [false; 3],
            Backward::Matmul(p) => pad(p.input_dependencies()),
            Backward::Bmm(p) => pad(p.input_dependencies()),
            Backward::Sdpa(p) => p.input_dependencies(),
            Backward::Conv1d(p) => pad(p.input_dependencies()),
            Backward::Conv2d(p) => pad(p.input_dependencies()),
            Backward::ConvTranspose1d(p) => pad(p.input_dependencies()),
            Backward::ConvTranspose2d(p) => pad(p.input_dependencies()),
            Backward::Norm(p) => [p.input_required(), false, false],
            Backward::Softmax(p) => [p.input_required(), false, false],
            Backward::Rope(p) => [p.input_required(), false, false],
            Backward::Pad(p) => [p.input_required(), false, false],
            Backward::Cat(p) => pad(p.input_dependencies()),
        }
    }
    fn forward_memory(plan: &Forward) -> PlanMemory {
        match plan {
            Forward::Unary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Binary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Reduce(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Matmul(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Bmm(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Sdpa(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Conv1d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Conv2d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::ConvTranspose1d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::ConvTranspose2d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Stft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::CenteredStft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Istft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Norm(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Softmax(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Rope(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Pad(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Forward::Cat(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
        }
    }
    fn backward_memory(plan: &Backward) -> PlanMemory {
        match plan {
            Backward::Unary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Binary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Reduce(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Matmul(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Bmm(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Sdpa(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Conv1d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Conv2d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::ConvTranspose1d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::ConvTranspose2d(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Stft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::CenteredStft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Istft(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Norm(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Softmax(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Rope(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Pad(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
            Backward::Cat(p) => memory(p.workspace_bytes(), p.host_metadata_bytes()),
        }
    }
    fn project_memory(plan: &cpu::ViewBackward) -> PlanMemory {
        PlanMemory {
            workspace: plan.workspace_bytes(),
            metadata: plan.host_metadata_bytes(),
            indices: plan.index_bytes(),
            shared: 0,
        }
    }
    fn project_plan(
        &self,
        base: &Shape,
        view: &Layout,
        upstream: &Layout,
        limit: usize,
    ) -> Result<cpu::ViewBackward> {
        cpu::ViewBackward::new(base.dims(), view, upstream, limit)
    }
    fn forward(
        &self,
        plan: &mut Forward,
        inputs: [Option<Read<'_, Tensor>>; 3],
        output: Write<'_, Tensor>,
    ) -> Result<()> {
        let a = inputs[0].map(|(t, l)| t.view(l)).transpose()?;
        let b = inputs[1].map(|(t, l)| t.view(l)).transpose()?;
        let c = inputs[2].map(|(t, l)| t.view(l)).transpose()?;
        let mut out = output.0.view_mut(output.1)?;
        match plan {
            Forward::Unary(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Binary(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
            Forward::Reduce(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Matmul(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
            Forward::Bmm(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
            Forward::Sdpa(p) => p.run(
                a.as_ref().unwrap(),
                b.as_ref().unwrap(),
                c.as_ref().unwrap(),
                &mut out,
            ),
            Forward::Conv1d(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
            Forward::Conv2d(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
            Forward::ConvTranspose1d(p) => {
                p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out)
            }
            Forward::ConvTranspose2d(p) => {
                p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out)
            }
            Forward::Stft(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::CenteredStft(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Istft(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Norm(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Softmax(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Rope(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Pad(p) => p.run(a.as_ref().unwrap(), &mut out),
            Forward::Cat(p) => p.run(a.as_ref().unwrap(), b.as_ref().unwrap(), &mut out),
        }
    }
    fn backward(
        &self,
        plan: &mut Backward,
        inputs: [Option<Read<'_, Tensor>>; 3],
        upstream: Read<'_, Tensor>,
        output: Write<'_, Tensor>,
        mode: GradientMode,
    ) -> Result<()> {
        let a = inputs[0].map(|(t, l)| t.view(l)).transpose()?;
        let b = inputs[1].map(|(t, l)| t.view(l)).transpose()?;
        let c = inputs[2].map(|(t, l)| t.view(l)).transpose()?;
        let dy = upstream.0.view(upstream.1)?;
        let mut out = output.0.view_mut(output.1)?;
        match plan {
            Backward::Unary(p) => p.run(a.as_ref(), &dy, &mut out, mode),
            Backward::Binary(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::Reduce(p) => p.run_with_input(a.as_ref(), &dy, &mut out, mode),
            Backward::Matmul(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::Bmm(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::Sdpa(p) => p.run(a.as_ref(), b.as_ref(), c.as_ref(), &dy, &mut out, mode),
            Backward::Conv1d(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::Conv2d(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::ConvTranspose1d(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::ConvTranspose2d(p) => p.run(a.as_ref(), b.as_ref(), &dy, &mut out, mode),
            Backward::Stft(p) => p.run(&dy, &mut out, mode),
            Backward::CenteredStft(p) => p.run(&dy, &mut out, mode),
            Backward::Istft(p) => p.run(&dy, &mut out, mode),
            Backward::Norm(p) => p.run(a.as_ref().unwrap(), &dy, &mut out, mode),
            Backward::Softmax(p) => p.run(a.as_ref().unwrap(), &dy, &mut out, mode),
            Backward::Rope(p) => p.run(&dy, &mut out, mode),
            Backward::Pad(p) => p.run(&dy, &mut out, mode),
            Backward::Cat(p) => p.run(&dy, &mut out, mode),
        }
    }
    fn project(
        &self,
        plan: &cpu::ViewBackward,
        upstream: Read<'_, Tensor>,
        output: &mut Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        plan.run(&upstream.0.view(upstream.1)?, output, mode)
    }
}

/// Native prepared CPU forward/reverse/update execution with guarded shared leaves.
pub struct Tape {
    core: Core<Host>,
    fused: Option<(FusedStepPlan, CpuFusedScratch)>,
    fused_checked: bool,
}
impl Tape {
    pub fn new(cpu: Cpu, program: &Program, bindings: &[Binding<Tensor>]) -> Result<Self> {
        Self::with_options(cpu, program, bindings, PrepareOptions::default())
    }
    pub fn with_options(
        cpu: Cpu,
        program: &Program,
        bindings: &[Binding<Tensor>],
        options: PrepareOptions<'_>,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::prepare(Host(cpu), program, bindings, options)?,
            fused: None,
            fused_checked: false,
        })
    }
    super::tape_methods!(Tensor);

    /// Reports whether a fused CPU whole-step execution plan is currently active.
    pub fn has_fused_plan(&self) -> bool {
        self.fused.is_some()
    }

    /// Execute forward + Overwrite backward + SGD using L1-tiled whole-step fusion when eligible.
    pub fn step(&mut self, seeds: &[Seed<'_, Tensor>], leaves: &[Value], rate: f32) -> Result<()> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(invalid("SGD rate must be finite and nonnegative"));
        }
        self.core.validate_seeds_for_capture(seeds)?;
        let selected_matches = {
            let sel = self.core.select_updated_slots(leaves)?;
            self.fused
                .as_ref()
                .is_some_and(|(p, _)| p.updated_slots.as_slice() == sel)
        };
        if !selected_matches && (!self.fused_checked || self.fused.is_some()) {
            let sel = self.core.select_updated_slots(leaves)?.to_vec();
            self.fused = self.core.analyze_fusion(&sel).and_then(|plan| {
                if plan.region.exts.len() > 16 {
                    return None;
                }
                let scratch = CpuFusedScratch::new(&plan.region, sel.len());
                Some((plan, scratch))
            });
            self.fused_checked = true;
        }
        if let Some((plan, scratch)) = self.fused.as_mut() {
            let region = &plan.region;
            if let Some(last_step) = plan.barrier_step {
                self.core.forward_up_to(last_step)?;
            }
            self.core.preflight_region(region)?;
            let seed_val = seeds[region.seed_root_pos].tensor.data()[0];
            {
                let (owners, gradients) = self.core.owners_and_gradients_mut();
                let mut guards: [Option<crate::ReadGuard<'_, Tensor>>; 16] = Default::default();
                for (i, ext) in region.exts.iter().enumerate() {
                    guards[i] = Some(owners[ext.owner_root].as_ref().unwrap().read()?);
                }
                let mut slices: [&[f32]; 16] = [&[]; 16];
                for i in 0..region.exts.len() {
                    slices[i] = guards[i].as_ref().unwrap().data();
                }
                fusion::execute_cpu_region(
                    region,
                    scratch,
                    &slices[..region.exts.len()],
                    gradients,
                    seed_val,
                );
            }
            // Apply inline SGD for leaves updated inside the fused region (exclusive write guards).
            {
                let (owners, gradients) = self.core.owners_and_gradients_mut();
                for ext in &region.exts {
                    if !ext.update_sgd {
                        continue;
                    }
                    let slot = ext.grad_slot.unwrap();
                    let mut param = owners[ext.owner_root].as_ref().unwrap().write()?;
                    let grad = &gradients[slot].data;
                    for (p, &g) in param.data_mut().iter_mut().zip(grad.data()) {
                        *p -= rate * g;
                    }
                }
            }
            if let Some(last_step) = plan.barrier_step {
                self.core
                    .backward_from(last_step, &plan.preinitialized_grad_slots)?;
                self.core.select_updated_slots(leaves)?;
                self.core
                    .sgd_selected_except(rate, &scratch.updated_in_region)?;
            } else {
                self.core.mark_captured_step_done(true);
            }
            return Ok(());
        }
        self.core.forward()?;
        self.core.backward(seeds, GradientMode::Overwrite)?;
        self.core.sgd(leaves, rate)
    }

    /// Scalar-root shorthand for `step`.
    pub fn step_scalar(&mut self, leaves: &[Value], rate: f32) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step(&[Seed::new(value, &*one, &layout)], leaves, rate)
    }

    /// Apply one `cpu::AdamW` step using currently valid gradients on this tape.
    /// Supports multi-microbatch gradient accumulation (`forward + backward(Accumulate)`).
    pub fn apply_adamw(&mut self, optimizer: &mut cpu::AdamW) -> Result<()> {
        let leaves = optimizer.leaves().to_vec();
        self.core.apply_optimizer_update(&leaves, |params, grads| {
            optimizer.apply_slices(params, grads)
        })
    }

    fn execute_forward_backward_fused(&mut self, seeds: &[Seed<'_, Tensor>]) -> Result<()> {
        let selected_matches = {
            let sel = self.core.select_updated_slots(&[])?;
            self.fused
                .as_ref()
                .is_some_and(|(p, _)| p.updated_slots.as_slice() == sel)
        };
        if !selected_matches && (!self.fused_checked || self.fused.is_some()) {
            let sel = self.core.select_updated_slots(&[])?.to_vec();
            self.fused = self.core.analyze_fusion(&sel).and_then(|plan| {
                if plan.region.exts.len() > 16 {
                    return None;
                }
                let scratch = CpuFusedScratch::new(&plan.region, sel.len());
                Some((plan, scratch))
            });
            self.fused_checked = true;
        }
        if let Some((plan, scratch)) = self.fused.as_mut() {
            let region = &plan.region;
            if let Some(last_step) = plan.barrier_step {
                self.core.forward_up_to(last_step)?;
            } else {
                self.core.record_leaf_versions()?;
            }
            self.core.preflight_region(region)?;
            let seed_val = seeds[region.seed_root_pos].tensor.data()[0];
            {
                let (owners, gradients) = self.core.owners_and_gradients_mut();
                let mut guards: [Option<crate::ReadGuard<'_, Tensor>>; 16] = Default::default();
                for (i, ext) in region.exts.iter().enumerate() {
                    guards[i] = Some(owners[ext.owner_root].as_ref().unwrap().read()?);
                }
                let mut slices: [&[f32]; 16] = [&[]; 16];
                for i in 0..region.exts.len() {
                    slices[i] = guards[i].as_ref().unwrap().data();
                }
                fusion::execute_cpu_region(
                    region,
                    scratch,
                    &slices[..region.exts.len()],
                    gradients,
                    seed_val,
                );
            }
            if let Some(last_step) = plan.barrier_step {
                self.core
                    .backward_from(last_step, &plan.preinitialized_grad_slots)?;
            } else {
                self.core.mark_gradients_ready();
            }
            return Ok(());
        }
        self.core.forward()?;
        self.core.backward(seeds, GradientMode::Overwrite)
    }

    /// Execute forward + Overwrite backward (with L1-tiled fusion when eligible) + `cpu::AdamW`.
    pub fn step_adamw(
        &mut self,
        seeds: &[Seed<'_, Tensor>],
        optimizer: &mut cpu::AdamW,
    ) -> Result<()> {
        self.core.validate_seeds_for_capture(seeds)?;
        self.execute_forward_backward_fused(seeds)?;
        self.apply_adamw(optimizer)
    }

    /// Scalar-root shorthand for `step_adamw`.
    pub fn step_scalar_adamw(&mut self, optimizer: &mut cpu::AdamW) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step_adamw(&[Seed::new(value, &*one, &layout)], optimizer)
    }
}
