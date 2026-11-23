use crate::autograd::{
    Binding, Operation, PrepareOptions, Program, Seed, Value,
    backend::{Backend, PlanMemory, Read, Write},
    engine::Core,
    fusion::{
        FusedStepPlan,
        cuda_gen::{LaunchTopology, emit_cuda_fused_source},
    },
};
use crate::{
    BinaryOperand, GradientMode, Layout, Result, SdpaOperand, Shape, StorageId, Tensor,
    cuda::{self, Cuda, DeviceTensor},
    error::invalid,
};
use std::{
    ffi::{CString, c_char, c_void},
    ptr::NonNull,
};

unsafe extern "C" {
    fn fa_cuda_tape_clear(ctx: *mut c_void, n: u64, out: *mut f32) -> i32;
    fn fa_cuda_tape_sgd(
        ctx: *mut c_void,
        n: u64,
        out: *mut f32,
        gradient: *const f32,
        rate: f32,
    ) -> i32;
    fn fa_cuda_capture_begin(ctx: *mut c_void) -> i32;
    fn fa_cuda_capture_end(ctx: *mut c_void, exec_out: *mut *mut c_void) -> i32;
    fn fa_cuda_capture_abort(ctx: *mut c_void);
    fn fa_cuda_graph_launch(ctx: *mut c_void, exec: *mut c_void) -> i32;
    fn fa_cuda_graph_destroy(ctx: *mut c_void, exec: *mut c_void);
    fn fa_cuda_nvrtc_available() -> i32;
    fn fa_cuda_nvrtc_compile(
        ctx: *mut c_void,
        source: *const c_char,
        main_name: *const c_char,
        fin_name: *const c_char,
        scratch_bytes: usize,
        out_handle: *mut *mut c_void,
        out_scratch: *mut *mut f32,
    ) -> i32;
    fn fa_cuda_nvrtc_launch(
        handle: *mut c_void,
        grid_x: u32,
        block_x: u32,
        smem_bytes: u32,
        kernel_params: *mut *mut c_void,
        is_finalize: i32,
    ) -> i32;
    fn fa_cuda_nvrtc_destroy(handle: *mut c_void);
}

pub(crate) struct Gpu(pub Cuda);
pub(crate) enum Forward {
    Unary(Box<cuda::UnaryPlan>),
    Binary(Box<cuda::BinaryPlan>),
    Reduce(Box<cuda::ReducePlan>),
    Matmul(Box<cuda::MatmulPlan>),
    Bmm(Box<cuda::BatchedMatmulPlan>),
    Sdpa(Box<cuda::SdpaPlan>),
    Conv1d(Box<cuda::Conv1dPlan>),
    Conv2d(Box<cuda::Conv2dPlan>),
    ConvTranspose1d(Box<cuda::ConvTranspose1dPlan>),
    ConvTranspose2d(Box<cuda::ConvTranspose2dPlan>),
    Stft(Box<cuda::StftPlan>),
    CenteredStft(Box<cuda::CenteredStftPlan>),
    Istft(Box<cuda::IstftPlan>),
    Norm(Box<cuda::NormPlan>),
    Softmax(Box<cuda::SoftmaxPlan>),
    Rope(Box<cuda::RopePlan>),
    Pad(Box<cuda::PadPlan>),
    Cat(Box<cuda::CatPlan>),
}
pub(crate) enum Backward {
    Unary(Box<cuda::UnaryBackward>),
    Binary(Box<cuda::BinaryBackward>),
    Reduce(Box<cuda::ReduceBackward>),
    Matmul(Box<cuda::MatmulBackward>),
    Bmm(Box<cuda::BatchedMatmulBackward>),
    Sdpa(Box<cuda::SdpaBackward>),
    Conv1d(Box<cuda::Conv1dBackward>),
    Conv2d(Box<cuda::Conv2dBackward>),
    ConvTranspose1d(Box<cuda::ConvTranspose1dBackward>),
    ConvTranspose2d(Box<cuda::ConvTranspose2dBackward>),
    Stft(Box<cuda::StftBackward>),
    CenteredStft(Box<cuda::CenteredStftBackward>),
    Istft(Box<cuda::IstftBackward>),
    Norm(Box<cuda::NormBackward>),
    Softmax(Box<cuda::SoftmaxBackward>),
    Rope(Box<cuda::RopeBackward>),
    Pad(Box<cuda::PadBackward>),
    Cat(Box<cuda::CatBackward>),
}
fn memory(workspace: usize, metadata: usize, shared: usize) -> PlanMemory {
    PlanMemory {
        workspace,
        metadata,
        shared,
        ..PlanMemory::default()
    }
}
impl Backend for Gpu {
    type Tensor = DeviceTensor;
    type Forward = Forward;
    type Backward = Backward;
    type Projection = cuda::ViewBackward;
    fn check(&self, tensor: &DeviceTensor) -> Result<()> {
        self.0.owns(tensor)
    }
    fn zeros(&self, shape: &Shape) -> Result<DeviceTensor> {
        self.0.zeros(shape.dims())
    }
    fn scalar(&self, value: f32) -> Result<DeviceTensor> {
        self.0.upload(&Tensor::from_vec([], vec![value])?)
    }
    fn clear(&self, tensor: &mut DeviceTensor) -> Result<()> {
        self.0.owns(tensor)?;
        // SAFETY: original session and initialized dense byte extent are checked;
        // exclusive access lasts through enqueue and existing CUDA ownership rules apply.
        super::check(unsafe {
            fa_cuda_tape_clear(
                self.0.session.ptr(),
                tensor.shape().len() as u64,
                tensor.write(),
            )
        })
    }
    fn update(
        &self,
        parameter: &mut DeviceTensor,
        gradient: &DeviceTensor,
        rate: f32,
    ) -> Result<()> {
        self.0.owns(parameter)?;
        self.0.owns(gradient)?;
        if parameter.shape() != gradient.shape() {
            return Err(invalid("parameter and gradient shapes differ"));
        }
        // SAFETY: equal checked shapes, same session, exclusive parameters and a
        // distinct immutable canonical gradient. The scalar is passed by value.
        super::check(unsafe {
            fa_cuda_tape_sgd(
                self.0.session.ptr(),
                parameter.shape().len() as u64,
                parameter.write(),
                gradient.read(),
                rate,
            )
        })
    }
    fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
    fn forward_plan(
        &self,
        op: &Operation,
        inputs: [Option<&Layout>; 3],
        output: &Layout,
    ) -> Result<Forward> {
        let a = inputs[0].unwrap();
        Ok(match op {
            Operation::Unary(op) => {
                Forward::Unary(Box::new(cuda::UnaryPlan::new(&self.0, *op, a, output)?))
            }
            Operation::Binary(op) => Forward::Binary(Box::new(cuda::BinaryPlan::new(
                &self.0,
                *op,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Reduce(spec) => {
                Forward::Reduce(Box::new(cuda::ReducePlan::new(&self.0, spec, a, output)?))
            }
            Operation::Matmul => Forward::Matmul(Box::new(cuda::MatmulPlan::new(
                &self.0,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Bmm(spec) => Forward::Bmm(Box::new(cuda::BatchedMatmulPlan::new(
                &self.0,
                spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Sdpa(spec) => Forward::Sdpa(Box::new(cuda::SdpaPlan::new(
                &self.0,
                *spec,
                a,
                inputs[1].unwrap(),
                inputs[2].unwrap(),
                output,
            )?)),
            Operation::Conv1d(spec) => Forward::Conv1d(Box::new(cuda::Conv1dPlan::new(
                &self.0,
                *spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::Conv2d(spec) => Forward::Conv2d(Box::new(cuda::Conv2dPlan::new(
                &self.0,
                *spec,
                a,
                inputs[1].unwrap(),
                output,
            )?)),
            Operation::ConvTranspose1d(spec) => Forward::ConvTranspose1d(Box::new(
                cuda::ConvTranspose1dPlan::new(&self.0, *spec, a, inputs[1].unwrap(), output)?,
            )),
            Operation::ConvTranspose2d(spec) => Forward::ConvTranspose2d(Box::new(
                cuda::ConvTranspose2dPlan::new(&self.0, *spec, a, inputs[1].unwrap(), output)?,
            )),
            Operation::Stft { spec, window } => Forward::Stft(Box::new(
                cuda::StftPlan::with_window_slice(&self.0, *spec, window.as_deref(), a, output)?,
            )),
            Operation::CenteredStft { spec, window } => {
                Forward::CenteredStft(Box::new(cuda::CenteredStftPlan::with_window_slice(
                    &self.0,
                    *spec,
                    window.as_deref(),
                    a,
                    output,
                )?))
            }
            Operation::Istft { spec, window } => Forward::Istft(Box::new(
                cuda::IstftPlan::with_window_slice(&self.0, *spec, window.as_deref(), a, output)?,
            )),
            Operation::Norm(spec) => {
                Forward::Norm(Box::new(cuda::NormPlan::new(&self.0, spec, a, output)?))
            }
            Operation::Softmax(spec) => {
                Forward::Softmax(Box::new(cuda::SoftmaxPlan::new(&self.0, spec, a, output)?))
            }
            Operation::Rope(spec) => {
                Forward::Rope(Box::new(cuda::RopePlan::new(&self.0, spec, a, output)?))
            }
            Operation::Pad(spec) => {
                Forward::Pad(Box::new(cuda::PadPlan::new(&self.0, spec, a, output)?))
            }
            Operation::Cat(spec) => Forward::Cat(Box::new(cuda::CatPlan::new(
                &self.0,
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
            Operation::Unary(op) => Backward::Unary(Box::new(cuda::UnaryBackward::new(
                &self.0, *op, a, upstream, output,
            )?)),
            Operation::Binary(op) => Backward::Binary(Box::new(cuda::BinaryBackward::new(
                &self.0,
                *op,
                side,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Reduce(spec) => Backward::Reduce(Box::new(
                cuda::ReduceBackward::with_input(&self.0, spec, a, upstream, output)?,
            )),
            Operation::Matmul => Backward::Matmul(Box::new(cuda::MatmulBackward::new(
                &self.0,
                side,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Bmm(spec) => Backward::Bmm(Box::new(cuda::BatchedMatmulBackward::new(
                &self.0,
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
                Backward::Sdpa(Box::new(cuda::SdpaBackward::new(
                    &self.0,
                    sdpa_op,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    inputs[2].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::Conv1d(spec) => Backward::Conv1d(Box::new(cuda::Conv1dBackward::new(
                &self.0,
                side,
                *spec,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::Conv2d(spec) => Backward::Conv2d(Box::new(cuda::Conv2dBackward::new(
                &self.0,
                side,
                *spec,
                a,
                inputs[1].unwrap(),
                upstream,
                output,
            )?)),
            Operation::ConvTranspose1d(spec) => {
                Backward::ConvTranspose1d(Box::new(cuda::ConvTranspose1dBackward::new(
                    &self.0,
                    side,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::ConvTranspose2d(spec) => {
                Backward::ConvTranspose2d(Box::new(cuda::ConvTranspose2dBackward::new(
                    &self.0,
                    side,
                    *spec,
                    a,
                    inputs[1].unwrap(),
                    upstream,
                    output,
                )?))
            }
            Operation::Stft { spec, window } => {
                Backward::Stft(Box::new(cuda::StftBackward::with_window_slice(
                    &self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::CenteredStft { spec, window } => {
                Backward::CenteredStft(Box::new(cuda::CenteredStftBackward::with_window_slice(
                    &self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::Istft { spec, window } => {
                Backward::Istft(Box::new(cuda::IstftBackward::with_window_slice(
                    &self.0,
                    *spec,
                    window.as_deref(),
                    upstream,
                    output,
                )?))
            }
            Operation::Norm(spec) => Backward::Norm(Box::new(cuda::NormBackward::new(
                &self.0, spec, a, upstream, output,
            )?)),
            Operation::Softmax(spec) => Backward::Softmax(Box::new(cuda::SoftmaxBackward::new(
                &self.0, spec, a, upstream, output,
            )?)),
            Operation::Rope(spec) => Backward::Rope(Box::new(cuda::RopeBackward::new(
                &self.0, spec, upstream, output,
            )?)),
            Operation::Pad(spec) => Backward::Pad(Box::new(cuda::PadBackward::new(
                &self.0, spec, upstream, output,
            )?)),
            Operation::Cat(spec) => Backward::Cat(Box::new(cuda::CatBackward::new(
                &self.0, side, spec, upstream, output,
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
            Forward::Unary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes(), 0),
            Forward::Binary(p) => memory(p.workspace_bytes(), p.host_metadata_bytes(), 0),
            Forward::Reduce(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Matmul(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Bmm(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Sdpa(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Conv1d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Conv2d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::ConvTranspose1d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::ConvTranspose2d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Stft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::CenteredStft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Istft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Norm(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Softmax(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Rope(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Pad(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Forward::Cat(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
        }
    }
    fn backward_memory(plan: &Backward) -> PlanMemory {
        match plan {
            Backward::Unary(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Binary(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Reduce(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Matmul(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Bmm(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Sdpa(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Conv1d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Conv2d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::ConvTranspose1d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::ConvTranspose2d(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Stft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::CenteredStft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Istft(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Norm(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Softmax(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Rope(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Pad(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
            Backward::Cat(p) => memory(
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
            ),
        }
    }
    fn project_memory(plan: &cuda::ViewBackward) -> PlanMemory {
        PlanMemory {
            workspace: plan.workspace_bytes().saturating_sub(plan.index_bytes()),
            metadata: plan.host_metadata_bytes(),
            indices: plan.index_bytes(),
            shared: plan.shared_memory_bytes_per_block(),
        }
    }
    fn project_plan(
        &self,
        base: &Shape,
        view: &Layout,
        upstream: &Layout,
        limit: usize,
    ) -> Result<cuda::ViewBackward> {
        cuda::ViewBackward::new(&self.0, base.dims(), view, upstream, limit)
    }
    fn forward(
        &self,
        plan: &mut Forward,
        inputs: [Option<Read<'_, DeviceTensor>>; 3],
        output: Write<'_, DeviceTensor>,
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
        inputs: [Option<Read<'_, DeviceTensor>>; 3],
        upstream: Read<'_, DeviceTensor>,
        output: Write<'_, DeviceTensor>,
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
        plan: &cuda::ViewBackward,
        upstream: Read<'_, DeviceTensor>,
        output: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        plan.run(&upstream.0.view(upstream.1)?, output, mode)
    }
}

struct CapturedGraph {
    cuda: Cuda,
    exec: NonNull<c_void>,
    leaves: Vec<(usize, StorageId, *const f32, bool)>,
    updated_roots: Vec<usize>,
    seeds: Vec<(Value, *const f32, Layout)>,
    rate_bits: u32,
}
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        // SAFETY: sole owner of exec; retains live CUDA session; native synchronizes before destroy.
        unsafe { fa_cuda_graph_destroy(self.cuda.session.ptr(), self.exec.as_ptr()) }
    }
}

struct CapturedAdamWGraph {
    cuda: Cuda,
    exec: NonNull<c_void>,
    leaves: Vec<(usize, StorageId, *const f32, bool)>,
    updated_leaves: Vec<Value>,
    seeds: Vec<(Value, *const f32, Layout)>,
    ctrl_ptr: *mut c_void,
    moment_ptrs: Vec<(*const f32, *const f32)>,
    non_lr_fp: (u32, u32, u32, u32, Option<u32>),
}
impl Drop for CapturedAdamWGraph {
    fn drop(&mut self) {
        // SAFETY: sole owner of exec; retains live CUDA session; native synchronizes before destroy.
        unsafe { fa_cuda_graph_destroy(self.cuda.session.ptr(), self.exec.as_ptr()) }
    }
}

struct CaptureGuard(*mut c_void, bool);
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.1 {
            // SAFETY: active stream capture is aborted cleanly on error or unwind.
            unsafe { fa_cuda_capture_abort(self.0) }
        }
    }
}

struct CudaFusedPlan {
    _cuda: Cuda,
    handle: NonNull<c_void>,
    scratch: *mut f32,
    plan: FusedStepPlan,
    topology: LaunchTopology,
    updated_in_region: Vec<bool>,
}
impl Drop for CudaFusedPlan {
    fn drop(&mut self) {
        // SAFETY: sole owner of handle; retains live CUDA session; native synchronizes before unload.
        unsafe { fa_cuda_nvrtc_destroy(self.handle.as_ptr()) }
    }
}

/// Native same-stream CUDA forward/reverse/update execution with guarded shared leaves.
pub struct Tape {
    core: Core<Gpu>,
    capture_warmed: bool,
    captured: Option<CapturedGraph>,
    adamw_capture_warmed: bool,
    captured_adamw: Option<CapturedAdamWGraph>,
    fused: Option<CudaFusedPlan>,
    fused_checked: bool,
}
impl Tape {
    pub fn new(cuda: &Cuda, program: &Program, bindings: &[Binding<DeviceTensor>]) -> Result<Self> {
        Self::with_options(cuda, program, bindings, PrepareOptions::default())
    }
    pub fn with_options(
        cuda: &Cuda,
        program: &Program,
        bindings: &[Binding<DeviceTensor>],
        options: PrepareOptions<'_>,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::prepare(Gpu(cuda.clone()), program, bindings, options)?,
            capture_warmed: false,
            captured: None,
            adamw_capture_warmed: false,
            captured_adamw: None,
            fused: None,
            fused_checked: false,
        })
    }
    crate::autograd::tape_methods!(DeviceTensor);

    /// Discard any cached CUDA Graph executable after waiting for pending stream work.
    pub fn invalidate_capture(&mut self) {
        self.captured = None;
        self.captured_adamw = None;
    }

    /// Reports whether a captured CUDA Graph executable is currently active.
    pub fn has_captured_graph(&self) -> bool {
        self.captured.is_some() || self.captured_adamw.is_some()
    }

    /// Reports whether an NVRTC fused whole-step kernel plan is currently active.
    pub fn has_fused_plan(&self) -> bool {
        self.fused.is_some()
    }

    fn ensure_fused_plan(&mut self, leaves: &[Value]) -> Result<()> {
        let selected_matches = {
            let sel = self.core.select_updated_slots(leaves)?;
            self.fused
                .as_ref()
                .is_some_and(|f| f.plan.updated_slots.as_slice() == sel)
        };
        if selected_matches || (self.fused_checked && self.fused.is_none()) {
            return Ok(());
        }
        let sel = self.core.select_updated_slots(leaves)?.to_vec();
        self.fused = None;
        self.fused_checked = true;
        // SAFETY: side-effect-free availability probe via dlopen/dlsym once_flag.
        if unsafe { fa_cuda_nvrtc_available() } == 0 {
            return Ok(());
        }
        let Some(plan) = self.core.analyze_fusion(&sel) else {
            return Ok(());
        };
        if plan.region.exts.len() > 16 {
            return Ok(());
        }
        let generated = emit_cuda_fused_source(&plan);
        let src_c = CString::new(generated.source)
            .map_err(|_| invalid("generated CUDA source contains NUL"))?;
        let main_c = CString::new(generated.main_name)
            .map_err(|_| invalid("generated kernel name contains NUL"))?;
        let fin_c = CString::new(generated.fin_name)
            .map_err(|_| invalid("generated finalize name contains NUL"))?;
        let mut raw_handle = std::ptr::null_mut();
        let mut scratch_ptr = std::ptr::null_mut();
        let ctx = self.core.backend().0.session.ptr();
        // SAFETY: live CUDA context and NUL-terminated strings; native owns returned handle & scratch.
        let code = unsafe {
            fa_cuda_nvrtc_compile(
                ctx,
                src_c.as_ptr(),
                main_c.as_ptr(),
                fin_c.as_ptr(),
                generated.scratch_bytes,
                &mut raw_handle,
                &mut scratch_ptr,
            )
        };
        if code != 0 {
            return super::check(code);
        }
        let handle = super::nonnull(raw_handle)?;
        let mut updated_in_region = vec![false; sel.len()];
        for ext in &plan.region.exts {
            if ext.update_sgd {
                if let Some(slot) = ext.grad_slot {
                    updated_in_region[slot] = true;
                }
            }
        }
        self.fused = Some(CudaFusedPlan {
            _cuda: self.core.backend().0.clone(),
            handle,
            scratch: scratch_ptr,
            plan,
            topology: generated.topology,
            updated_in_region,
        });
        Ok(())
    }

    fn execute_step_body(
        &mut self,
        seeds: &[Seed<'_, DeviceTensor>],
        leaves: &[Value],
        rate: f32,
    ) -> Result<()> {
        if let Some(fused) = self.fused.as_ref() {
            let barrier_step = fused.plan.barrier_step;
            if let Some(last_step) = barrier_step {
                self.core.forward_up_to(last_step)?;
            }
            let fused = self.fused.as_ref().unwrap();
            let region = &fused.plan.region;
            self.core.preflight_region(region)?;
            let mut ext_ptrs = [std::ptr::null_mut::<c_void>(); 16];
            for (i, ext) in region.exts.iter().enumerate() {
                if ext.update_sgd {
                    let mut w = self.core.value_owner(ext.owner_root).write()?;
                    ext_ptrs[i] = w.write().cast();
                } else {
                    let r = self.core.value_owner(ext.owner_root).read()?;
                    ext_ptrs[i] = r.read().cast_mut().cast();
                }
            }
            let mut ptr_storage = [std::ptr::null_mut::<c_void>(); 40];
            let mut n_ptrs = 0usize;
            for ptr in ext_ptrs.iter().take(region.exts.len()) {
                ptr_storage[n_ptrs] = *ptr;
                n_ptrs += 1;
            }
            for ext in &region.exts {
                if let Some(slot) = ext.grad_slot {
                    ptr_storage[n_ptrs] = self.core.gradient_mut(slot).write().cast();
                    n_ptrs += 1;
                }
            }
            ptr_storage[n_ptrs] = seeds[region.seed_root_pos].tensor.read().cast_mut().cast();
            let seed_idx = n_ptrs;
            n_ptrs += 1;
            let mut rate_arg = rate;
            let (main_grid, main_block, has_fin, with_scratch) = match fused.topology {
                LaunchTopology::BarrierTailOnly { grid, block } => (grid, block, false, false),
                LaunchTopology::SingleBlockAll { block, .. } => (1, block, false, false),
                LaunchTopology::WarpPerGroup {
                    grid,
                    block,
                    has_finalize,
                    ..
                } => (grid, block, has_finalize, has_finalize),
                LaunchTopology::SplitBlocksPerGroup { grid, block, .. } => {
                    (grid, block, true, true)
                }
            };
            let scratch_idx = n_ptrs;
            if with_scratch {
                ptr_storage[n_ptrs] = fused.scratch.cast();
                n_ptrs += 1;
            }
            let _ = n_ptrs;
            let mut params = [std::ptr::null_mut::<c_void>(); 40];
            let mut n_params = 0usize;
            for slot in ptr_storage.iter_mut().take(seed_idx + 1) {
                params[n_params] = (slot as *mut *mut c_void).cast();
                n_params += 1;
            }
            params[n_params] = (&mut rate_arg as *mut f32).cast();
            n_params += 1;
            if with_scratch {
                params[n_params] = (&mut ptr_storage[scratch_idx] as *mut *mut c_void).cast();
            }
            let handle_ptr = fused.handle.as_ptr();
            // SAFETY: all input/gradient/parameter/seed/scratch pointers are validated for this
            // live CUDA session and match the compiled kernel signature; versions were advanced.
            super::check(unsafe {
                fa_cuda_nvrtc_launch(handle_ptr, main_grid, main_block, 0, params.as_mut_ptr(), 0)
            })?;
            if has_fin {
                // SAFETY: finalize kernel executes on the same stream after main_fn with matching args.
                super::check(unsafe {
                    fa_cuda_nvrtc_launch(handle_ptr, 1, 256, 0, params.as_mut_ptr(), 1)
                })?;
            }
            if let Some(last_step) = barrier_step {
                let fused = self.fused.as_ref().unwrap();
                self.core
                    .backward_from(last_step, &fused.plan.preinitialized_grad_slots)?;
                self.core.select_updated_slots(leaves)?;
                self.core
                    .sgd_selected_except(rate, &fused.updated_in_region)?;
            } else {
                self.core.mark_captured_step_done(true);
            }
            return Ok(());
        }
        self.core.forward()?;
        self.core.backward(seeds, GradientMode::Overwrite)?;
        self.core.sgd(leaves, rate)
    }

    /// Execute forward + Overwrite backward + SGD using NVRTC whole-step fusion when eligible.
    pub fn step(
        &mut self,
        seeds: &[Seed<'_, DeviceTensor>],
        leaves: &[Value],
        rate: f32,
    ) -> Result<()> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(invalid("SGD rate must be finite and nonnegative"));
        }
        self.core.validate_seeds_for_capture(seeds)?;
        self.ensure_fused_plan(leaves)?;
        self.execute_step_body(seeds, leaves, rate)
    }

    /// Scalar-root shorthand for `step`.
    pub fn step_scalar(&mut self, leaves: &[Value], rate: f32) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step(&[Seed::new(value, &*one, &layout)], leaves, rate)
    }

    /// Execute forward + Overwrite backward + SGD with automatic NVRTC fusion and CUDA Graph
    /// capture/replay. The first call compiles/warms uncaptured; the second call captures the
    /// stream into a CUDA Graph and launches it; subsequent matching calls replay with a single
    /// launch, zero Rust allocations, full pointer/session/borrow validation, and version advance.
    pub fn step_captured(
        &mut self,
        seeds: &[Seed<'_, DeviceTensor>],
        leaves: &[Value],
        rate: f32,
    ) -> Result<()> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(invalid("SGD rate must be finite and nonnegative"));
        }
        if let Some(cap) = self.captured.as_ref() {
            let mut reusable = cap.rate_bits == rate.to_bits()
                && cap.seeds.len() == seeds.len()
                && self
                    .core
                    .matches_updated_leaves(leaves, &cap.updated_roots)?;
            if reusable {
                for (seed, (sv, sp, sl)) in seeds.iter().zip(&cap.seeds) {
                    if seed.value != *sv || seed.layout != sl {
                        reusable = false;
                        break;
                    }
                    self.core.backend().0.owns(seed.tensor)?;
                    sl.validate_storage(seed.tensor.shape().len())?;
                    if seed.tensor.read() != *sp {
                        reusable = false;
                        break;
                    }
                }
            }
            if reusable {
                for &(root, id, ptr, updated) in &cap.leaves {
                    let owner = self.core.leaf_owner(root);
                    let expected_shape = self.core.program().0.values[root].layout.shape();
                    {
                        let read = owner.read()?;
                        self.core.backend().0.owns(&read)?;
                        if read.shape() != expected_shape {
                            return Err(invalid("bound training storage shape changed"));
                        }
                        if read.version().storage_id() != id || read.read() != ptr {
                            reusable = false;
                            break;
                        }
                    }
                    if updated {
                        owner.check_writable()?;
                    }
                }
            }
            if reusable {
                let mut changed = false;
                for &(root, _id, ptr, updated) in &cap.leaves {
                    if !updated {
                        continue;
                    }
                    let expected_shape = self.core.program().0.values[root].layout.shape();
                    let write = self.core.leaf_owner(root).write()?;
                    if write.shape() != expected_shape || write.read() != ptr {
                        changed = true;
                        break;
                    }
                }
                if changed {
                    self.captured = None;
                    self.core.mark_captured_step_done(false);
                    return Err(invalid(
                        "training leaf storage changed concurrently during captured replay",
                    ));
                }
                let ctx = self.core.backend().0.session.ptr();
                let exec = cap.exec.as_ptr();
                // SAFETY: all bound leaf/seed pointers, shapes, sessions, and exclusive parameter
                // write leases were verified; internal tape storage is owned exclusively by self.
                let status = super::check(unsafe { fa_cuda_graph_launch(ctx, exec) });
                self.core.mark_captured_step_done(status.is_ok());
                return status;
            }
            self.captured = None;
        }
        self.core.validate_seeds_for_capture(seeds)?;
        self.ensure_fused_plan(leaves)?;
        if !self.capture_warmed {
            self.execute_step_body(seeds, leaves, rate)?;
            self.capture_warmed = true;
            return Ok(());
        }
        let leaf_flags = self.core.capture_leaf_roots(leaves)?;
        let mut leaf_records = Vec::with_capacity(leaf_flags.len());
        let mut updated_roots = Vec::new();
        for (root, updated) in leaf_flags {
            let read = self.core.leaf_owner(root).read()?;
            self.core.backend().0.owns(&read)?;
            leaf_records.push((root, read.version().storage_id(), read.read(), updated));
            if updated {
                updated_roots.push(root);
            }
        }
        let seed_records = seeds
            .iter()
            .map(|s| (s.value, s.tensor.read(), s.layout.clone()))
            .collect();
        self.core.synchronize()?;
        let ctx = self.core.backend().0.session.ptr();
        // SAFETY: live session on idle stream; CaptureGuard aborts capture on any error/unwind.
        super::check(unsafe { fa_cuda_capture_begin(ctx) })?;
        let mut guard = CaptureGuard(ctx, true);
        let record_res = self.execute_step_body(seeds, leaves, rate);
        if record_res.is_err() {
            drop(guard);
            self.core.mark_captured_step_done(false);
            return record_res;
        }
        guard.1 = false;
        let mut raw_exec = std::ptr::null_mut();
        // SAFETY: matching end-capture on the same stream; native instantiates and owns exec.
        let end_res = super::check(unsafe { fa_cuda_capture_end(ctx, &mut raw_exec) });
        if end_res.is_err() {
            self.core.mark_captured_step_done(false);
            return end_res;
        }
        let exec = super::nonnull(raw_exec)?;
        // SAFETY: freshly instantiated CUDA Graph executable on the same live stream.
        let launch_res = super::check(unsafe { fa_cuda_graph_launch(ctx, exec.as_ptr()) });
        self.captured = Some(CapturedGraph {
            cuda: self.core.backend().0.clone(),
            exec,
            leaves: leaf_records,
            updated_roots,
            seeds: seed_records,
            rate_bits: rate.to_bits(),
        });
        if launch_res.is_err() {
            self.core.mark_captured_step_done(false);
        }
        launch_res
    }

    /// Scalar-root shorthand for `step_captured`.
    pub fn step_scalar_captured(&mut self, leaves: &[Value], rate: f32) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step_captured(&[Seed::new(value, &*one, &layout)], leaves, rate)
    }

    fn apply_adamw_inner(&mut self, optimizer: &mut cuda::AdamW, bump_host: bool) -> Result<()> {
        if !optimizer.same_session(&self.core.backend().0) {
            return Err(invalid("CUDA AdamW belongs to a different CUDA session"));
        }
        let leaves = optimizer.leaves().to_vec();
        self.core.apply_optimizer_update(&leaves, |params, grads| {
            optimizer.enqueue_apply_slices(params, grads, bump_host)
        })
    }

    /// Apply one `cuda::AdamW` step using currently valid gradients on this tape.
    /// Supports multi-microbatch gradient accumulation (`forward + backward(Accumulate)`).
    pub fn apply_adamw(&mut self, optimizer: &mut cuda::AdamW) -> Result<()> {
        self.apply_adamw_inner(optimizer, true)
    }

    fn execute_forward_backward_fused(&mut self, seeds: &[Seed<'_, DeviceTensor>]) -> Result<()> {
        self.ensure_fused_plan(&[])?;
        if let Some(fused) = self.fused.as_ref() {
            let barrier_step = fused.plan.barrier_step;
            if let Some(last_step) = barrier_step {
                self.core.forward_up_to(last_step)?;
            } else {
                self.core.record_leaf_versions()?;
            }
            let fused = self.fused.as_ref().unwrap();
            let region = &fused.plan.region;
            self.core.preflight_region(region)?;
            let mut ext_ptrs = [std::ptr::null_mut::<c_void>(); 16];
            for (i, ext) in region.exts.iter().enumerate() {
                let r = self.core.value_owner(ext.owner_root).read()?;
                ext_ptrs[i] = r.read().cast_mut().cast();
            }
            let mut ptr_storage = [std::ptr::null_mut::<c_void>(); 40];
            let mut n_ptrs = 0usize;
            for ptr in ext_ptrs.iter().take(region.exts.len()) {
                ptr_storage[n_ptrs] = *ptr;
                n_ptrs += 1;
            }
            for ext in &region.exts {
                if let Some(slot) = ext.grad_slot {
                    ptr_storage[n_ptrs] = self.core.gradient_mut(slot).write().cast();
                    n_ptrs += 1;
                }
            }
            ptr_storage[n_ptrs] = seeds[region.seed_root_pos].tensor.read().cast_mut().cast();
            let seed_idx = n_ptrs;
            n_ptrs += 1;
            let mut rate_arg = 0.0f32;
            let (main_grid, main_block, has_fin, with_scratch) = match fused.topology {
                LaunchTopology::BarrierTailOnly { grid, block } => (grid, block, false, false),
                LaunchTopology::SingleBlockAll { block, .. } => (1, block, false, false),
                LaunchTopology::WarpPerGroup {
                    grid,
                    block,
                    has_finalize,
                    ..
                } => (grid, block, has_finalize, has_finalize),
                LaunchTopology::SplitBlocksPerGroup { grid, block, .. } => {
                    (grid, block, true, true)
                }
            };
            let scratch_idx = n_ptrs;
            if with_scratch {
                ptr_storage[n_ptrs] = fused.scratch.cast();
                n_ptrs += 1;
            }
            let _ = n_ptrs;
            let mut params = [std::ptr::null_mut::<c_void>(); 40];
            let mut n_params = 0usize;
            for slot in ptr_storage.iter_mut().take(seed_idx + 1) {
                params[n_params] = (slot as *mut *mut c_void).cast();
                n_params += 1;
            }
            params[n_params] = (&mut rate_arg as *mut f32).cast();
            n_params += 1;
            if with_scratch {
                params[n_params] = (&mut ptr_storage[scratch_idx] as *mut *mut c_void).cast();
            }
            let handle_ptr = fused.handle.as_ptr();
            // SAFETY: all pointers validated for live CUDA session; fused kernel writes gradients only.
            super::check(unsafe {
                fa_cuda_nvrtc_launch(handle_ptr, main_grid, main_block, 0, params.as_mut_ptr(), 0)
            })?;
            if has_fin {
                // SAFETY: finalize kernel executes on the same stream after main_fn with matching args.
                super::check(unsafe {
                    fa_cuda_nvrtc_launch(handle_ptr, 1, 256, 0, params.as_mut_ptr(), 1)
                })?;
            }
            if let Some(last_step) = barrier_step {
                let fused = self.fused.as_ref().unwrap();
                self.core
                    .backward_from(last_step, &fused.plan.preinitialized_grad_slots)?;
            } else {
                self.core.mark_gradients_ready();
            }
            return Ok(());
        }
        self.core.forward()?;
        self.core.backward(seeds, GradientMode::Overwrite)
    }

    /// Execute forward + Overwrite backward (with NVRTC fusion when eligible) + `cuda::AdamW`.
    pub fn step_adamw(
        &mut self,
        seeds: &[Seed<'_, DeviceTensor>],
        optimizer: &mut cuda::AdamW,
    ) -> Result<()> {
        if !optimizer.same_session(&self.core.backend().0) {
            return Err(invalid("CUDA AdamW belongs to a different CUDA session"));
        }
        self.core.validate_seeds_for_capture(seeds)?;
        self.execute_forward_backward_fused(seeds)?;
        self.apply_adamw_inner(optimizer, true)
    }

    /// Scalar-root shorthand for `step_adamw`.
    pub fn step_scalar_adamw(&mut self, optimizer: &mut cuda::AdamW) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step_adamw(&[Seed::new(value, &*one, &layout)], optimizer)
    }

    /// Execute forward + backward + deterministic global L2 grad clipping + `cuda::AdamW` +
    /// on-device `d_step++` captured into ONE single CUDA Graph. Dynamic `optimizer.set_lr(...)`
    /// schedules replay the existing graph with zero re-captures and zero device-to-host syncs.
    pub fn step_adamw_captured(
        &mut self,
        seeds: &[Seed<'_, DeviceTensor>],
        optimizer: &mut cuda::AdamW,
    ) -> Result<()> {
        if !optimizer.same_session(&self.core.backend().0) {
            return Err(invalid("CUDA AdamW belongs to a different CUDA session"));
        }
        if let Some(cap) = self.captured_adamw.as_ref() {
            let mut reusable = cap.ctrl_ptr == optimizer.ctrl_ptr()
                && cap.non_lr_fp == optimizer.config().non_lr_fingerprint()
                && cap.updated_leaves.as_slice() == optimizer.leaves()
                && optimizer.matches_moment_ptrs(&cap.moment_ptrs)
                && cap.seeds.len() == seeds.len();
            if reusable {
                for (seed, (sv, sp, sl)) in seeds.iter().zip(&cap.seeds) {
                    if seed.value != *sv || seed.layout != sl {
                        reusable = false;
                        break;
                    }
                    self.core.backend().0.owns(seed.tensor)?;
                    sl.validate_storage(seed.tensor.shape().len())?;
                    if seed.tensor.read() != *sp {
                        reusable = false;
                        break;
                    }
                }
            }
            if reusable {
                for &(root, id, ptr, updated) in &cap.leaves {
                    let owner = self.core.leaf_owner(root);
                    let expected_shape = self.core.program().0.values[root].layout.shape();
                    {
                        let read = owner.read()?;
                        self.core.backend().0.owns(&read)?;
                        if read.shape() != expected_shape {
                            return Err(invalid("bound training storage shape changed"));
                        }
                        if read.version().storage_id() != id || read.read() != ptr {
                            reusable = false;
                            break;
                        }
                    }
                    if updated {
                        owner.check_writable()?;
                    }
                }
            }
            if reusable {
                let mut changed = false;
                for &(root, _id, ptr, updated) in &cap.leaves {
                    if !updated {
                        continue;
                    }
                    let expected_shape = self.core.program().0.values[root].layout.shape();
                    let write = self.core.leaf_owner(root).write()?;
                    if write.shape() != expected_shape || write.read() != ptr {
                        changed = true;
                        break;
                    }
                }
                if changed {
                    self.captured_adamw = None;
                    self.core.mark_captured_step_done(false);
                    return Err(invalid(
                        "training leaf storage changed concurrently during captured AdamW replay",
                    ));
                }
                let ctx = self.core.backend().0.session.ptr();
                let exec = cap.exec.as_ptr();
                // SAFETY: all leaf/seed/moment/control pointers, shapes, sessions, and exclusive
                // parameter write leases were verified above.
                let status = super::check(unsafe { fa_cuda_graph_launch(ctx, exec) });
                if status.is_ok() {
                    optimizer.bump_host_step()?;
                }
                self.core.mark_captured_step_done(status.is_ok());
                return status;
            }
            self.captured_adamw = None;
        }

        self.core.validate_seeds_for_capture(seeds)?;
        if !self.adamw_capture_warmed {
            self.execute_forward_backward_fused(seeds)?;
            self.apply_adamw_inner(optimizer, true)?;
            self.adamw_capture_warmed = true;
            return Ok(());
        }

        // Pre-compile any NVRTC fusion plan outside stream capture.
        self.ensure_fused_plan(&[])?;
        let updated_leaves = optimizer.leaves().to_vec();
        let leaf_flags = self.core.capture_leaf_roots(&updated_leaves)?;
        let mut leaf_records = Vec::with_capacity(leaf_flags.len());
        for (root, updated) in leaf_flags {
            let read = self.core.leaf_owner(root).read()?;
            self.core.backend().0.owns(&read)?;
            leaf_records.push((root, read.version().storage_id(), read.read(), updated));
        }
        let seed_records = seeds
            .iter()
            .map(|s| (s.value, s.tensor.read(), s.layout.clone()))
            .collect();
        let ctrl_ptr = optimizer.ctrl_ptr();
        let moment_ptrs = optimizer.moment_ptrs();
        let non_lr_fp = optimizer.config().non_lr_fingerprint();

        self.core.synchronize()?;
        let ctx = self.core.backend().0.session.ptr();
        // SAFETY: live session on idle stream; CaptureGuard aborts capture on any error/unwind.
        super::check(unsafe { fa_cuda_capture_begin(ctx) })?;
        let mut guard = CaptureGuard(ctx, true);
        let record_res = self
            .execute_forward_backward_fused(seeds)
            .and_then(|()| self.apply_adamw_inner(optimizer, false));
        if record_res.is_err() {
            drop(guard);
            self.core.mark_captured_step_done(false);
            return record_res;
        }
        guard.1 = false;
        let mut raw_exec = std::ptr::null_mut();
        // SAFETY: matching end-capture on the same stream; native instantiates and owns exec.
        let end_res = super::check(unsafe { fa_cuda_capture_end(ctx, &mut raw_exec) });
        if end_res.is_err() {
            self.core.mark_captured_step_done(false);
            return end_res;
        }
        let exec = super::nonnull(raw_exec)?;
        // SAFETY: freshly instantiated CUDA Graph executable on the same live stream.
        let launch_res = super::check(unsafe { fa_cuda_graph_launch(ctx, exec.as_ptr()) });
        self.captured_adamw = Some(CapturedAdamWGraph {
            cuda: self.core.backend().0.clone(),
            exec,
            leaves: leaf_records,
            updated_leaves,
            seeds: seed_records,
            ctrl_ptr,
            moment_ptrs,
            non_lr_fp,
        });
        if launch_res.is_ok() {
            optimizer.bump_host_step()?;
        } else {
            self.core.mark_captured_step_done(false);
        }
        launch_res
    }

    /// Scalar-root shorthand for `step_adamw_captured`.
    pub fn step_scalar_adamw_captured(&mut self, optimizer: &mut cuda::AdamW) -> Result<()> {
        let (value, one, layout) = self.core.scalar_seed_info()?;
        self.step_adamw_captured(&[Seed::new(value, &*one, &layout)], optimizer)
    }
}
