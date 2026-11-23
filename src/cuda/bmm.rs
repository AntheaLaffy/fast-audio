use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    BinaryOperand, BmmSpec, GradientMode, Layout, Result,
    bmm::{BmmKernelDesc, build_backward_desc, build_forward_desc},
    error::{int, invalid},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeBmmDims {
    m: u64,
    n: u64,
    k: u64,
    out_batches: u64,
    red_batches: u64,
    origins: [i64; 3],
    strides: [i64; 6],
    batch_strides: [i64; 3],
    leading: [i32; 3],
    transposes: [i32; 2],
    out_rank: i32,
    red_rank: i32,
    route: i32,
    swapped: i32,
    blas_m: i32,
    blas_n: i32,
    blas_k: i32,
    batch_count: i32,
    reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeBmmAxis {
    dim: u64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

const _: () = assert!(
    size_of::<NativeBmmDims>() == 192
        && std::mem::offset_of!(NativeBmmDims, origins) == 40
        && std::mem::offset_of!(NativeBmmDims, strides) == 64
        && std::mem::offset_of!(NativeBmmDims, batch_strides) == 112
        && std::mem::offset_of!(NativeBmmDims, leading) == 136
        && size_of::<NativeBmmAxis>() == 32
);

unsafe extern "C" {
    fn fa_cuda_bmm_create(
        ctx: *mut c_void,
        dims: NativeBmmDims,
        out_axes: *const NativeBmmAxis,
        red_axes: *const NativeBmmAxis,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_bmm_run(
        plan: *mut c_void,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bmm_destroy(plan: *mut c_void);
}

struct Core {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}

impl Core {
    fn from_desc(cuda: &Cuda, desc: &BmmKernelDesc) -> Result<Self> {
        let route = if desc.out_batches == 0 || desc.m == 0 || desc.n == 0 {
            0
        } else if desc.k == 0 || desc.red_batches == 0 {
            1
        } else if let Some(blas) = desc.blas {
            if blas.collapsed_2d {
                2
            } else if blas.strided_batched {
                3
            } else {
                4
            }
        } else {
            4
        };
        let (leading, transposes, swapped, blas_m, blas_n, blas_k, batch_count, batch_strides) =
            if let Some(blas) = desc.blas {
                (
                    [blas.lda, blas.ldb, blas.ldc],
                    [blas.ta, blas.tb],
                    i32::from(blas.swapped),
                    blas.m,
                    blas.n,
                    blas.k,
                    blas.batch_count,
                    [
                        blas.batch_stride_a,
                        blas.batch_stride_b,
                        blas.batch_stride_c,
                    ],
                )
            } else {
                ([0; 3], [0; 2], 0, 0, 0, 0, 0, [0; 3])
            };
        let out_axes: Vec<NativeBmmAxis> = desc
            .out_axes
            .iter()
            .map(|a| NativeBmmAxis {
                dim: a.dim as u64,
                stride_a: a.stride_a as i64,
                stride_b: a.stride_b as i64,
                stride_c: a.stride_c as i64,
            })
            .collect();
        let red_axes: Vec<NativeBmmAxis> = desc
            .red_axes
            .iter()
            .map(|a| NativeBmmAxis {
                dim: a.dim as u64,
                stride_a: a.stride_a as i64,
                stride_b: a.stride_b as i64,
                stride_c: a.stride_c as i64,
            })
            .collect();
        let dims = NativeBmmDims {
            m: desc.m as u64,
            n: desc.n as u64,
            k: desc.k as u64,
            out_batches: desc.out_batches as u64,
            red_batches: desc.red_batches as u64,
            origins: [
                desc.origins[0] as i64,
                desc.origins[1] as i64,
                desc.origins[2] as i64,
            ],
            strides: [
                desc.mat_a.row_stride as i64,
                desc.mat_a.col_stride as i64,
                desc.mat_b.row_stride as i64,
                desc.mat_b.col_stride as i64,
                desc.mat_c.row_stride as i64,
                desc.mat_c.col_stride as i64,
            ],
            batch_strides,
            leading,
            transposes,
            out_rank: int(out_axes.len())?,
            red_rank: int(red_axes.len())?,
            route,
            swapped,
            blas_m,
            blas_n,
            blas_k,
            batch_count,
            reserved: 0,
        };
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: asserted ABI sizes and validated geometry; native syncs metadata uploads.
        check(unsafe {
            fa_cuda_bmm_create(
                cuda.session.ptr(),
                dims,
                out_axes.as_ptr(),
                red_axes.as_ptr(),
                &mut workspace,
                &mut shared,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            ptr: nonnull(ptr)?,
            workspace,
            shared,
        })
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: sole native plan owner retains CUDA session; native syncs before destroying.
        unsafe { fa_cuda_bmm_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared arbitrary-rank broadcast batched matmul forward plan on CUDA.
pub struct BatchedMatmulPlan {
    spec: BmmSpec,
    left_layout: Layout,
    right_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl BatchedMatmulPlan {
    pub fn new(
        cuda: &Cuda,
        spec: &BmmSpec,
        left: &Layout,
        right: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let desc = build_forward_desc(spec, left, right, output)?;
        let core = Core::from_desc(cuda, &desc)?;
        Ok(Self {
            spec: spec.clone(),
            left_layout: left.clone(),
            right_layout: right.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &BmmSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }

    pub fn run(
        &self,
        left: &DeviceView<'_>,
        right: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(left.tensor)?;
        self.core.cuda.owns(right.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if left.layout() != &self.left_layout
            || right.layout() != &self.right_layout
            || output.layout() != &self.output_layout
        {
            return Err(invalid(
                "CUDA batched matmul layouts differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_bmm_run(
                self.core.ptr.as_ptr(),
                left.tensor.read(),
                right.tensor.read(),
                output.tensor.write(),
                0,
            )
        })
    }
}

/// Prepared arbitrary-rank broadcast batched matmul local VJP on CUDA.
pub struct BatchedMatmulBackward {
    spec: BmmSpec,
    operand: BinaryOperand,
    left_layout: Layout,
    right_layout: Layout,
    upstream_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl BatchedMatmulBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: &BmmSpec,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let desc = build_backward_desc(spec, operand, left, right, upstream, output)?;
        let core = Core::from_desc(cuda, &desc)?;
        Ok(Self {
            spec: spec.clone(),
            operand,
            left_layout: left.clone(),
            right_layout: right.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &BmmSpec {
        &self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        match self.operand {
            BinaryOperand::Left => [false, true],
            BinaryOperand::Right => [true, false],
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }

    pub fn run(
        &self,
        left: Option<&DeviceView<'_>>,
        right: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA batched matmul backward layouts differ from prepared plan",
            ));
        }
        let acc = i32::from(mode == GradientMode::Accumulate);
        match self.operand {
            BinaryOperand::Left => {
                let r = right.ok_or_else(|| {
                    invalid("CUDA batched matmul Left backward requires saved right input")
                })?;
                self.core.cuda.owns(r.tensor)?;
                if r.layout() != &self.right_layout {
                    return Err(invalid(
                        "CUDA batched matmul backward right layout differs from prepared plan",
                    ));
                }
                // SAFETY: validated session, layouts, exclusive injective output; enqueues on stream.
                check(unsafe {
                    fa_cuda_bmm_run(
                        self.core.ptr.as_ptr(),
                        upstream.tensor.read(),
                        r.tensor.read(),
                        output.tensor.write(),
                        acc,
                    )
                })
            }
            BinaryOperand::Right => {
                let l = left.ok_or_else(|| {
                    invalid("CUDA batched matmul Right backward requires saved left input")
                })?;
                self.core.cuda.owns(l.tensor)?;
                if l.layout() != &self.left_layout {
                    return Err(invalid(
                        "CUDA batched matmul backward left layout differs from prepared plan",
                    ));
                }
                // SAFETY: validated session, layouts, exclusive injective output; enqueues on stream.
                check(unsafe {
                    fa_cuda_bmm_run(
                        self.core.ptr.as_ptr(),
                        l.tensor.read(),
                        upstream.tensor.read(),
                        output.tensor.write(),
                        acc,
                    )
                })
            }
        }
    }
}
