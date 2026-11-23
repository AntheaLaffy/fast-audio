use crate::{Error, Layout, Result, Shape, Tensor, TensorView, copy::MappingSpec};

/// Explicit initialization policy for a gradient destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientMode {
    Overwrite,
    Accumulate,
}

/// Geometry-selected execution route. Index-based routes obey the preparation budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ViewBackwardStrategy {
    Empty = 0,
    Contiguous = 1,
    InjectiveStrided = 2,
    Broadcast = 3,
    Indexed = 4,
    Grouped = 5,
}

pub(crate) fn index_bytes(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<u64>())
        .ok_or_else(|| Error::Unsupported("view backward index bytes overflow".into()))
}
pub(crate) fn limit_index(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        Err(Error::Unsupported(format!(
            "view backward index preparation requires {bytes} bytes, limit is {limit}"
        )))
    } else {
        Ok(())
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|e| Error::Allocation(e.to_string()))?;
    Ok(out)
}

pub(crate) struct Reduction {
    pub axes: Vec<(usize, isize)>,
    pub len: usize,
}
impl Reduction {
    pub fn offset(&self, mut flat: usize) -> isize {
        let mut offset = 0;
        for &(dim, stride) in self.axes.iter().skip(1).rev() {
            offset += (flat % dim) as isize * stride;
            flat /= dim;
        }
        if let Some(&(_, stride)) = self.axes.first() {
            offset += flat as isize * stride;
        }
        offset
    }
}
pub(crate) enum Route {
    Empty,
    Injective,
    Broadcast {
        unique: Box<MappingSpec>,
        reduction: Reduction,
        lanes: usize,
    },
    Indexed {
        targets: Vec<u64>,
        sources: Vec<u64>,
    },
    Grouped {
        targets: Vec<u64>,
        offsets: Vec<u64>,
        sources: Vec<u64>,
        lanes: usize,
    },
}
pub(crate) struct BackwardSpec {
    pub base: Shape,
    pub map: MappingSpec,
    pub route: Route,
    pub index_bytes: usize,
    pub preparation_index_bytes: usize,
    pub written: Option<(usize, usize)>,
}
fn lanes(count: usize) -> usize {
    if count <= 32 { 32 } else { 256 }
}

impl BackwardSpec {
    pub fn new(
        base: impl AsRef<[usize]>,
        view: &Layout,
        upstream: &Layout,
        limit: usize,
    ) -> Result<Self> {
        let base = Shape::new(base)?;
        view.validate_storage(base.len())?;
        let map = MappingSpec::new(upstream, view)?;
        let (route, index_bytes, preparation_index_bytes) = if map.len == 0 {
            (Route::Empty, 0, 0)
        } else if view.is_non_overlapping() {
            (Route::Injective, 0, 0)
        } else if let Some(route) = Self::broadcast(view, upstream)? {
            (route, 0, 0)
        } else {
            Self::indexed(&map, limit)?
        };
        let groups = match &route {
            Route::Empty => 0,
            Route::Injective => map.len,
            Route::Broadcast { unique, .. } => unique.len,
            Route::Indexed { targets, .. } | Route::Grouped { targets, .. } => targets.len(),
        };
        // Bounds plus a proved/exact unique count establish complete interval coverage.
        let written = view
            .span()
            .filter(|span| span.len() == groups)
            .map(|span| (span.start, span.end));
        Ok(Self {
            base,
            map,
            route,
            index_bytes,
            preparation_index_bytes,
            written,
        })
    }
    fn broadcast(view: &Layout, upstream: &Layout) -> Result<Option<Route>> {
        if !view
            .shape()
            .dims()
            .iter()
            .zip(view.strides())
            .any(|(&d, &s)| d > 1 && s == 0)
        {
            return Ok(None);
        }
        let dims: Vec<_> = view
            .shape()
            .dims()
            .iter()
            .zip(view.strides())
            .map(|(&d, &s)| if s == 0 { 1 } else { d })
            .collect();
        let unique_view = Layout::from_strides(&dims, view.strides(), view.offset())?;
        if !unique_view.is_non_overlapping() {
            return Ok(None);
        }
        let unique_upstream = Layout::from_strides(dims, upstream.strides(), upstream.offset())?;
        let unique = MappingSpec::new(&unique_upstream, &unique_view)?;
        let len = view.shape().len() / unique.len;
        let mut axes: Vec<(usize, isize)> = Vec::new();
        for ((&dim, &target), &source) in view
            .shape()
            .dims()
            .iter()
            .zip(view.strides())
            .zip(upstream.strides())
            .rev()
        {
            if dim <= 1 || target != 0 {
                continue;
            }
            if let Some((inner, step)) = axes.last_mut() {
                if step.checked_mul(*inner as isize) == Some(source) {
                    *inner *= dim;
                    continue;
                }
            }
            axes.push((dim, source));
        }
        axes.reverse();
        Ok(Some(Route::Broadcast {
            unique: Box::new(unique),
            reduction: Reduction { axes, len },
            lanes: lanes(len),
        }))
    }
    fn indexed(map: &MappingSpec, limit: usize) -> Result<(Route, usize, usize)> {
        let pair_bytes = map
            .len
            .checked_mul(size_of::<(u64, u64)>())
            .ok_or_else(|| Error::Unsupported("view backward pair bytes overflow".into()))?;
        limit_index(pair_bytes, limit)?;
        let mut pairs = reserve(map.len)?;
        for j in 0..map.len {
            pairs.push((map.destination.linear_valid(j) as u64, j as u64));
        }
        pairs.sort_unstable(); // Logical index is the secondary key: no stable-sort scratch.
        let (mut groups, mut degree, mut max_degree, mut previous) = (0usize, 0usize, 0usize, None);
        for &(target, _) in &pairs {
            if previous != Some(target) {
                groups += 1;
                degree = 0;
                previous = Some(target);
            }
            degree += 1;
            max_degree = max_degree.max(degree);
        }
        let count = groups
            .checked_mul(2)
            .and_then(|n| n.checked_add(map.len))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Unsupported("view backward table size overflows".into()))?;
        let table_bytes = index_bytes(if groups == map.len {
            groups
                .checked_mul(2)
                .ok_or_else(|| Error::Unsupported("view backward table size overflows".into()))?
        } else {
            count
        })?;
        let preparation = pair_bytes
            .checked_add(table_bytes)
            .ok_or_else(|| Error::Unsupported("view backward preparation bytes overflow".into()))?;
        limit_index(preparation, limit)?;
        let mut targets = reserve(groups)?;
        let mut sources = reserve(map.len)?;
        if groups == map.len {
            for (target, j) in pairs {
                targets.push(target);
                sources.push(map.source.linear_valid(j as usize) as u64);
            }
            Ok((
                Route::Indexed { targets, sources },
                table_bytes,
                preparation,
            ))
        } else {
            let mut offsets = reserve(groups + 1)?;
            let mut previous = None;
            for (target, j) in pairs {
                if previous != Some(target) {
                    offsets.push(sources.len() as u64);
                    targets.push(target);
                    previous = Some(target);
                }
                sources.push(map.source.linear_valid(j as usize) as u64);
            }
            offsets.push(sources.len() as u64);
            Ok((
                Route::Grouped {
                    targets,
                    offsets,
                    sources,
                    lanes: lanes(max_degree),
                },
                table_bytes,
                preparation,
            ))
        }
    }
    pub fn validate(&self, upstream: &Layout, out: &Shape) -> Result<()> {
        out.expect(self.base.dims())?;
        self.map.validate(upstream, &self.map.destination)
    }
    pub fn strategy(&self) -> ViewBackwardStrategy {
        match &self.route {
            Route::Empty => ViewBackwardStrategy::Empty,
            Route::Injective if self.map.contiguous => ViewBackwardStrategy::Contiguous,
            Route::Injective => ViewBackwardStrategy::InjectiveStrided,
            Route::Broadcast { .. } => ViewBackwardStrategy::Broadcast,
            Route::Indexed { .. } => ViewBackwardStrategy::Indexed,
            Route::Grouped { .. } => ViewBackwardStrategy::Grouped,
        }
    }
    pub fn reduction_lanes(&self) -> usize {
        match &self.route {
            Route::Empty => 0,
            Route::Injective | Route::Indexed { .. } => 1,
            Route::Broadcast { lanes, .. } | Route::Grouped { lanes, .. } => *lanes,
        }
    }
    #[cfg(feature = "cuda")]
    pub fn groups(&self) -> usize {
        match &self.route {
            Route::Empty => 0,
            Route::Injective => self.map.len,
            Route::Broadcast { unique, .. } => unique.len,
            Route::Indexed { targets, .. } | Route::Grouped { targets, .. } => targets.len(),
        }
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of_val(self.base.dims())
            + self.map.metadata_bytes()
            + match &self.route {
                Route::Broadcast {
                    unique, reduction, ..
                } => {
                    size_of::<MappingSpec>()
                        + unique.metadata_bytes()
                        + reduction.axes.capacity() * size_of::<(usize, isize)>()
                }
                Route::Indexed { targets, sources } => {
                    (targets.capacity() + sources.capacity()) * size_of::<u64>()
                }
                Route::Grouped {
                    targets,
                    offsets,
                    sources,
                    ..
                } => {
                    (targets.capacity() + offsets.capacity() + sources.capacity())
                        * size_of::<u64>()
                }
                _ => 0,
            }
    }
    fn clear(&self, out: &mut [f32]) {
        let (begin, end) = self.written.unwrap_or((0, 0));
        out[..begin].fill(0.0);
        out[end..].fill(0.0);
    }
}

fn tree<const LANES: usize>(count: usize, value: impl Fn(usize) -> f32) -> f32 {
    let mut partial = [0.0f32; LANES];
    for i in 0..count {
        partial[i & (LANES - 1)] += value(i);
    }
    let mut step = LANES / 2;
    while step > 0 {
        for lane in 0..step {
            partial[lane] += partial[lane + step];
        }
        step /= 2;
    }
    partial[0]
}
fn sum(lanes: usize, count: usize, value: impl Fn(usize) -> f32) -> f32 {
    if lanes == 32 {
        tree::<32>(count, value)
    } else {
        tree::<256>(count, value)
    }
}
fn write(out: &mut f32, value: f32, mode: GradientMode) {
    match mode {
        GradientMode::Overwrite => *out = value,
        GradientMode::Accumulate => *out += value,
    }
}

/// Prepared adjoint of a read view. No original forward payload is retained.
///
/// ```
/// use fast_audio::{GradientMode, Layout, Tensor, cpu::ViewBackward};
/// let view = Layout::from_strides([2, 2], [1, 1], 0)?;
/// let upstream = Layout::contiguous([2, 2])?;
/// let plan = ViewBackward::new([3], &view, &upstream, 1024)?;
/// let dy = Tensor::from_vec([2, 2], vec![1.; 4])?;
/// let mut dx = Tensor::zeros([3])?;
/// plan.run(&dy.view(plan.upstream_layout())?, &mut dx, GradientMode::Overwrite)?;
/// assert_eq!(dx.data(), &[1., 2., 1.]);
/// # Ok::<(), fast_audio::Error>(())
/// ```
pub struct ViewBackward {
    spec: BackwardSpec,
}
impl ViewBackward {
    pub fn new(
        base_shape: impl AsRef<[usize]>,
        view: &Layout,
        upstream: &Layout,
        index_limit_bytes: usize,
    ) -> Result<Self> {
        Ok(Self {
            spec: BackwardSpec::new(base_shape, view, upstream, index_limit_bytes)?,
        })
    }
    pub fn base_shape(&self) -> &[usize] {
        self.spec.base.dims()
    }
    pub fn view_layout(&self) -> &Layout {
        &self.spec.map.destination
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.spec.map.source
    }
    pub fn strategy(&self) -> ViewBackwardStrategy {
        self.spec.strategy()
    }
    pub fn reduction_lanes(&self) -> usize {
        self.spec.reduction_lanes()
    }
    pub fn index_bytes(&self) -> usize {
        self.spec.index_bytes
    }
    pub fn preparation_index_bytes(&self) -> usize {
        self.spec.preparation_index_bytes
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.host_metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        upstream: &TensorView<'_>,
        out: &mut Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        let spec = &self.spec;
        spec.validate(upstream.layout(), out.shape())?;
        if mode == GradientMode::Overwrite {
            spec.clear(out.data_mut());
        }
        match &spec.route {
            Route::Empty => Ok(()),
            Route::Injective => {
                let mut destination = out.view_mut(&spec.map.destination)?;
                match mode {
                    GradientMode::Overwrite => spec.map.apply::<false>(upstream, &mut destination),
                    GradientMode::Accumulate => spec.map.apply::<true>(upstream, &mut destination),
                }
            }
            Route::Broadcast {
                unique,
                reduction,
                lanes,
            } => {
                for group in 0..unique.len {
                    let (base, target) = unique.element_offsets(group);
                    let value = sum(*lanes, reduction.len, |i| {
                        upstream.data[(base as isize + reduction.offset(i)) as usize]
                    });
                    write(&mut out.data_mut()[target], value, mode);
                }
                Ok(())
            }
            Route::Indexed { targets, sources } => {
                for (&target, &source) in targets.iter().zip(sources) {
                    write(
                        &mut out.data_mut()[target as usize],
                        upstream.data[source as usize],
                        mode,
                    );
                }
                Ok(())
            }
            Route::Grouped {
                targets,
                offsets,
                sources,
                lanes,
            } => {
                for (group, &target) in targets.iter().enumerate() {
                    let begin = offsets[group] as usize;
                    let count = (offsets[group + 1] - offsets[group]) as usize;
                    let value = sum(*lanes, count, |i| {
                        upstream.data[sources[begin + i] as usize]
                    });
                    write(&mut out.data_mut()[target as usize], value, mode);
                }
                Ok(())
            }
        }
    }
}
