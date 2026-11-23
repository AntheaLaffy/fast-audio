use super::{Cuda, DeviceView, DeviceViewMut, check};
use crate::{
    BinaryOperand, Conv1dSpec, Conv2dSpec, ConvStrategy, ConvTranspose1dSpec, GradientMode, Layout,
    Result,
    conv::{BackwardGeometry, ForwardGeometry, SpatialCore},
};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Native {
    m: u64,
    n: u64,
    k: u64,
    origins: [i64; 3],
    strides: [i64; 12],
    batch: i32,
    groups: i32,
    cin_per_group: i32,
    cout_per_group: i32,
    input_hw: [i32; 2],
    kernel_hw: [i32; 2],
    output_hw: [i32; 2],
    stride_hw: [i32; 2],
    padding_hw: [i32; 2],
    dilation_hw: [i32; 2],
    leading: [i32; 3],
    transposes: [i32; 2],
    swapped: i32,
    kind: i32,
    op_mode: i32,
}

const _: () = assert!(
    size_of::<Native>() == 240
        && std::mem::offset_of!(Native, origins) == 24
        && std::mem::offset_of!(Native, strides) == 48
        && std::mem::offset_of!(Native, batch) == 144
        && std::mem::offset_of!(Native, leading) == 208
        && std::mem::offset_of!(Native, kind) == 232
);

unsafe extern "C" {
    fn fa_cuda_conv_shared(ctx: *mut c_void, kind: i32, bytes: *mut usize) -> i32;
    fn fa_cuda_conv_layout(
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
    strategy: ConvStrategy,
    shared: usize,
}
impl Core {
    fn new(cuda: &Cuda, core: &SpatialCore) -> Result<Self> {
        let strategy = if core.blas.is_some() {
            ConvStrategy::Cublas
        } else {
            ConvStrategy::TiledCuda
        };
        let [m, n, k] = core.dims;
        let mut strides = [0i64; 12];
        for (slot, tensor) in core.tensors.iter().enumerate() {
            for axis in 0..4 {
                strides[4 * slot + axis] = tensor.strides[axis] as i64;
            }
        }
        let dims = Native {
            m: m as u64,
            n: n as u64,
            k: k as u64,
            origins: core.tensors.map(|t| t.origin as i64),
            strides,
            batch: core.batch as i32,
            groups: core.groups as i32,
            cin_per_group: core.cin_per_group as i32,
            cout_per_group: core.cout_per_group as i32,
            input_hw: core.input_hw.map(|v| v as i32),
            kernel_hw: core.kernel_hw.map(|v| v as i32),
            output_hw: core.output_hw.map(|v| v as i32),
            stride_hw: core.stride_hw.map(|v| v as i32),
            padding_hw: core.padding_hw.map(|v| v as i32),
            dilation_hw: core.dilation_hw.map(|v| v as i32),
            leading: core.blas.map_or([0; 3], |b| b.leading),
            transposes: core.blas.map_or([0; 2], |b| b.transposes),
            swapped: i32::from(core.swapped),
            kind: strategy as i32,
            op_mode: core.op_mode as i32,
        };
        let mut shared = 0;
        // SAFETY: validated route and originating session; queries only static shared-memory size.
        check(unsafe { fa_cuda_conv_shared(cuda.session.ptr(), dims.kind, &mut shared) })?;
        Ok(Self {
            cuda: cuda.clone(),
            dims,
            strategy,
            shared,
        })
    }

    fn run(&self, data: [*const f32; 2], out: *mut f32, mode: GradientMode) -> Result<()> {
        // SAFETY: callers validate exact layouts, initialized storage spans, required payloads,
        // originating CUDA sessions, and exclusive injective destinations. The 240-byte descriptor
        // is passed by value with no execution-time allocation or metadata upload.
        check(unsafe {
            fa_cuda_conv_layout(
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

/// Prepared rank-three strided 1D cross-correlation on the originating CUDA stream.
pub struct Conv1dPlan {
    geometry: ForwardGeometry,
    spec: Conv1dSpec,
    core: Core,
}
impl Conv1dPlan {
    pub fn new(
        cuda: &Cuda,
        spec: Conv1dSpec,
        input: &Layout,
        weight: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = ForwardGeometry::new_1d(spec, input, weight, destination)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv1dSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn weight_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: &DeviceView<'_>,
        weight: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(weight.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry
            .validate(input.layout(), weight.layout(), out.layout())?;
        self.core.run(
            [input.tensor.read(), weight.tensor.read()],
            out.tensor.write(),
            GradientMode::Overwrite,
        )
    }
}

/// Local VJP for either 1D convolution operand on the originating CUDA stream.
pub struct Conv1dBackward {
    geometry: BackwardGeometry,
    spec: Conv1dSpec,
    core: Core,
}
impl Conv1dBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: Conv1dSpec,
        input: &Layout,
        weight: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let geometry = BackwardGeometry::new_1d(operand, spec, input, weight, upstream, gradient)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv1dSpec {
        self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn input_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn weight_layout(&self) -> &Layout {
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
    pub fn strategy(&self) -> ConvStrategy {
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
        input: Option<&DeviceView<'_>>,
        weight: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for view in [input, weight].into_iter().flatten() {
            self.core.cuda.owns(view.tensor)?;
        }
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry.validate(
            [input.map(|v| v.layout()), weight.map(|v| v.layout())],
            upstream.layout(),
            out.layout(),
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [
                upstream.tensor.read(),
                weight.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        } else {
            [
                input.map_or(std::ptr::null(), |v| v.tensor.read()),
                upstream.tensor.read(),
            ]
        };
        self.core.run(data, out.tensor.write(), mode)
    }
}

/// Prepared rank-four strided 2D cross-correlation on the originating CUDA stream.
pub struct Conv2dPlan {
    geometry: ForwardGeometry,
    spec: Conv2dSpec,
    core: Core,
}
impl Conv2dPlan {
    pub fn new(
        cuda: &Cuda,
        spec: Conv2dSpec,
        input: &Layout,
        weight: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = ForwardGeometry::new_2d(spec, input, weight, destination)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv2dSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn weight_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: &DeviceView<'_>,
        weight: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(weight.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry
            .validate(input.layout(), weight.layout(), out.layout())?;
        self.core.run(
            [input.tensor.read(), weight.tensor.read()],
            out.tensor.write(),
            GradientMode::Overwrite,
        )
    }
}

/// Local VJP for either 2D convolution operand on the originating CUDA stream.
pub struct Conv2dBackward {
    geometry: BackwardGeometry,
    spec: Conv2dSpec,
    core: Core,
}
impl Conv2dBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: Conv2dSpec,
        input: &Layout,
        weight: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let geometry = BackwardGeometry::new_2d(operand, spec, input, weight, upstream, gradient)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv2dSpec {
        self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn input_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn weight_layout(&self) -> &Layout {
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
    pub fn strategy(&self) -> ConvStrategy {
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
        input: Option<&DeviceView<'_>>,
        weight: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for view in [input, weight].into_iter().flatten() {
            self.core.cuda.owns(view.tensor)?;
        }
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry.validate(
            [input.map(|v| v.layout()), weight.map(|v| v.layout())],
            upstream.layout(),
            out.layout(),
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [
                upstream.tensor.read(),
                weight.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        } else {
            [
                input.map_or(std::ptr::null(), |v| v.tensor.read()),
                upstream.tensor.read(),
            ]
        };
        self.core.run(data, out.tensor.write(), mode)
    }
}

/// Prepared rank-three strided 1D transposed convolution on CUDA.
pub struct ConvTranspose1dPlan {
    geometry: ForwardGeometry,
    spec: ConvTranspose1dSpec,
    core: Core,
}
impl ConvTranspose1dPlan {
    pub fn new(
        cuda: &Cuda,
        spec: ConvTranspose1dSpec,
        input: &Layout,
        weight: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = ForwardGeometry::new_transpose_1d(spec, input, weight, destination)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> ConvTranspose1dSpec {
        self.spec
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: &DeviceView<'_>,
        weight: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(weight.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry
            .validate(input.layout(), weight.layout(), out.layout())?;
        self.core.run(
            [input.tensor.read(), weight.tensor.read()],
            out.tensor.write(),
            GradientMode::Overwrite,
        )
    }
}

/// Local VJP for 1D transposed convolution (`Left` = `dX`, `Right` = `dW`) on CUDA.
pub struct ConvTranspose1dBackward {
    geometry: BackwardGeometry,
    spec: ConvTranspose1dSpec,
    core: Core,
}
impl ConvTranspose1dBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: ConvTranspose1dSpec,
        input: &Layout,
        weight: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let geometry =
            BackwardGeometry::new_transpose_1d(operand, spec, input, weight, upstream, gradient)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> ConvTranspose1dSpec {
        self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.geometry.dependencies
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: Option<&DeviceView<'_>>,
        weight: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for view in [input, weight].into_iter().flatten() {
            self.core.cuda.owns(view.tensor)?;
        }
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry.validate(
            [input.map(|v| v.layout()), weight.map(|v| v.layout())],
            upstream.layout(),
            out.layout(),
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [
                upstream.tensor.read(),
                weight.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        } else {
            [
                upstream.tensor.read(),
                input.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        };
        self.core.run(data, out.tensor.write(), mode)
    }
}

/// Prepared rank-four strided 2D transposed convolution on CUDA.
pub struct ConvTranspose2dPlan {
    geometry: ForwardGeometry,
    spec: Conv2dSpec,
    core: Core,
}
impl ConvTranspose2dPlan {
    pub fn new(
        cuda: &Cuda,
        spec: Conv2dSpec,
        input: &Layout,
        weight: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = ForwardGeometry::new_transpose_2d(spec, input, weight, destination)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv2dSpec {
        self.spec
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: &DeviceView<'_>,
        weight: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(weight.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry
            .validate(input.layout(), weight.layout(), out.layout())?;
        self.core.run(
            [input.tensor.read(), weight.tensor.read()],
            out.tensor.write(),
            GradientMode::Overwrite,
        )
    }
}

/// Local VJP for 2D transposed convolution (`Left` = `dX`, `Right` = `dW`) on CUDA.
pub struct ConvTranspose2dBackward {
    geometry: BackwardGeometry,
    spec: Conv2dSpec,
    core: Core,
}
impl ConvTranspose2dBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: Conv2dSpec,
        input: &Layout,
        weight: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let geometry =
            BackwardGeometry::new_transpose_2d(operand, spec, input, weight, upstream, gradient)?;
        let core = Core::new(cuda, &geometry.core)?;
        Ok(Self {
            geometry,
            spec,
            core,
        })
    }
    pub fn spec(&self) -> Conv2dSpec {
        self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.geometry.dependencies
    }
    pub fn strategy(&self) -> ConvStrategy {
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
        input: Option<&DeviceView<'_>>,
        weight: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        out: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for view in [input, weight].into_iter().flatten() {
            self.core.cuda.owns(view.tensor)?;
        }
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(out.tensor)?;
        self.geometry.validate(
            [input.map(|v| v.layout()), weight.map(|v| v.layout())],
            upstream.layout(),
            out.layout(),
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [
                upstream.tensor.read(),
                weight.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        } else {
            [
                upstream.tensor.read(),
                input.map_or(std::ptr::null(), |v| v.tensor.read()),
            ]
        };
        self.core.run(data, out.tensor.write(), mode)
    }
}
