use super::{Cuda, DeviceView, DeviceViewMut, check};
use crate::{
    BinaryOperand, GradientMode, Layout, MatmulStrategy, Result,
    matmul::{BackwardGeometry, ForwardGeometry, Geometry},
};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Native {
    m: u64,
    n: u64,
    k: u64,
    origins: [i64; 3],
    strides: [i64; 6],
    leading: [i32; 3],
    transposes: [i32; 2],
    kind: i32,
    reserved: [i32; 2],
}
const _: () = assert!(
    size_of::<Native>() == 128
        && std::mem::offset_of!(Native, origins) == 24
        && std::mem::offset_of!(Native, strides) == 48
        && std::mem::offset_of!(Native, leading) == 96
        && std::mem::offset_of!(Native, kind) == 116
);
unsafe extern "C" {
    fn fa_cuda_matmul_shared(ctx: *mut c_void, kind: i32, bytes: *mut usize) -> i32;
    fn fa_cuda_matmul_layout(
        ctx: *mut c_void,
        dims: Native,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        accumulate: i32,
    ) -> i32;
}
struct Core {
    cuda: Cuda,
    dims: Native,
    swapped: bool,
    strategy: MatmulStrategy,
    shared: usize,
}
impl Core {
    fn new(cuda: &Cuda, g: &Geometry) -> Result<Self> {
        let strategy = if g.empty() {
            MatmulStrategy::Empty
        } else if g.zero() {
            MatmulStrategy::Zero
        } else if g.blas.is_some() {
            MatmulStrategy::Cublas
        } else {
            MatmulStrategy::StridedCuda
        };
        let [m, n, k] = g.dimensions();
        let mut strides = [0; 6];
        for (i, matrix) in g.matrices.iter().enumerate() {
            strides[2 * i] = matrix.steps[0] as i64;
            strides[2 * i + 1] = matrix.steps[1] as i64;
        }
        let dims = Native {
            m: m as u64,
            n: n as u64,
            k: k as u64,
            origins: g.matrices.map(|a| a.origin as i64),
            strides,
            leading: g.blas.map_or([0; 3], |b| b.leading),
            transposes: g.blas.map_or([0; 2], |b| b.transposes),
            kind: strategy as i32,
            reserved: [0; 2],
        };
        let mut shared = 0;
        // SAFETY: validated route and originating session; the native query reads only
        // the selected kernel's actual static shared-memory property.
        check(unsafe { fa_cuda_matmul_shared(cuda.session.ptr(), dims.kind, &mut shared) })?;
        Ok(Self {
            cuda: cuda.clone(),
            dims,
            swapped: g.swapped,
            strategy,
            shared,
        })
    }
    fn run(&self, mut data: [*const f32; 2], out: *mut f32, mode: GradientMode) -> Result<()> {
        if self.strategy == MatmulStrategy::Empty
            || (self.strategy == MatmulStrategy::Zero && mode == GradientMode::Accumulate)
        {
            return Ok(());
        }
        if self.swapped {
            data.swap(0, 1);
        }
        // SAFETY: wrappers checked all exact layouts, initialized spans, required payloads,
        // original sessions and exclusive injective destinations. Axis swaps preserve maps;
        // by-value descriptors need no borrowed metadata upload or native plan lifetime.
        check(unsafe {
            fa_cuda_matmul_layout(
                self.cuda.session.ptr(),
                self.dims,
                data[0],
                data[1],
                out,
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
/// Prepared rank-two strided multiplication on the originating CUDA stream.
pub struct MatmulPlan {
    geometry: ForwardGeometry,
    core: Core,
}
impl MatmulPlan {
    pub fn new(cuda: &Cuda, left: &Layout, right: &Layout, destination: &Layout) -> Result<Self> {
        let geometry = ForwardGeometry::new(left, right, destination)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self { geometry, core })
    }
    pub fn left_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn strategy(&self) -> MatmulStrategy {
        self.core.strategy
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(
        &self,
        left: &DeviceView<'_>,
        right: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(left.tensor)?;
        self.core.cuda.owns(right.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry
            .validate(left.layout(), right.layout(), out.layout())?;
        self.core.run(
            [left.tensor.read(), right.tensor.read()],
            out.tensor.write(),
            GradientMode::Overwrite,
        )
    }
}
/// One matrix operand's VJP without retaining or physically transposing parameters.
pub struct MatmulBackward {
    geometry: BackwardGeometry,
    core: Core,
}
impl MatmulBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let geometry = BackwardGeometry::new(operand, left, right, upstream, gradient)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self { geometry, core })
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn left_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[3]
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.geometry.dependencies
    }
    pub fn strategy(&self) -> MatmulStrategy {
        self.core.strategy
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(
        &self,
        left: Option<&DeviceView<'_>>,
        right: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for input in [left, right].into_iter().flatten() {
            self.core.cuda.owns(input.tensor)?;
        }
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry.validate(
            [left.map(|v| v.layout()), right.map(|v| v.layout())],
            upstream.layout(),
            out.layout(),
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [
                upstream.tensor.read(),
                right.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        } else {
            [
                left.map_or(std::ptr::null(), |v| v.tensor.read()),
                upstream.tensor.read(),
            ]
        };
        self.core.run(data, out.tensor.write(), mode)
    }
}
