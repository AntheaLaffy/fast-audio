use super::{DefinitionGraph, Kind, Operation};
use crate::{BinaryOp, Layout, ReduceOp, UnaryOp};

#[derive(Clone, Debug)]
pub(crate) struct AffineMap {
    pub offset: isize,
    pub strides: Vec<isize>,
    pub has_k: bool,
    pub k_stride: isize,
}

impl AffineMap {
    fn is_unit_inner(&self) -> bool {
        !self.has_k && self.strides.last().copied().unwrap_or(0) == 1
    }
    fn is_k_contiguous(&self, k_dim: usize) -> bool {
        self.has_k
            && self.k_stride == 1
            && self.strides.last().copied().unwrap_or(0) == k_dim as isize
    }
    fn is_zero_inner(&self) -> bool {
        self.strides.last().copied().unwrap_or(0) == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExtRole {
    ReadOnly,
    BarrierOut,
    GroupKept,
    SmallGlobal,
    FullElementwise,
}

#[derive(Clone, Debug)]
pub(crate) struct ExtInput {
    pub owner_root: usize,
    pub grad_slot: Option<usize>,
    pub param_len: usize,
    pub update_sgd: bool,
    pub role: ExtRole,
    pub param_map: Option<AffineMap>,
}

#[derive(Clone, Debug)]
pub(crate) enum FwdInstr {
    LoadExt {
        ext: usize,
        map: AffineMap,
    },
    Unary {
        op: UnaryOp,
        src: usize,
    },
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
    },
    SmallReduce {
        op: ReduceOp,
        src: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct RegionPlan {
    pub domain: Vec<usize>,
    pub total_domain: usize,
    pub k_dim: usize,
    pub loss_op: ReduceOp,
    pub seed_root_pos: usize,
    pub exts: Vec<ExtInput>,
    pub instrs: Vec<FwdInstr>,
    pub is_vec: Vec<bool>,
    pub req_grad: Vec<bool>,
    pub kept_axes: Vec<usize>,
    pub red_axes: Vec<usize>,
    pub group_len: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub red_len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FusedStepPlan {
    /// None => entire graph is fused; Some(idx) => steps[0..=idx] run via prepared plans,
    /// and the region fuses everything after step idx (writing d(barrier_out) + tail params).
    pub barrier_step: Option<usize>,
    pub preinitialized_grad_slots: Vec<usize>,
    pub region: RegionPlan,
    pub updated_slots: Vec<bool>,
}

struct Bijection {
    /// For each view axis `a`, `(base_axis, reversed)`.
    view_to_base: Vec<(usize, bool)>,
    base_dims: Vec<usize>,
}

impl Bijection {
    fn identity(dims: &[usize]) -> Self {
        Self {
            view_to_base: (0..dims.len()).map(|a| (a, false)).collect(),
            base_dims: dims.to_vec(),
        }
    }
    fn from_alias_layout(base_dims: &[usize], view_layout: &Layout) -> Option<Self> {
        let view_dims = view_layout.shape().dims();
        if base_dims.len() != view_dims.len() || base_dims.is_empty() {
            return None;
        }
        let rank = base_dims.len();
        let mut base_strides = vec![1isize; rank];
        for a in (0..rank.saturating_sub(1)).rev() {
            base_strides[a] = base_strides[a + 1].checked_mul(base_dims[a + 1] as isize)?;
        }
        let mut used = vec![false; rank];
        let mut view_to_base = Vec::with_capacity(rank);
        let mut expected_offset = 0isize;
        for (a, &dim) in view_dims.iter().enumerate() {
            let s = view_layout.strides()[a];
            let abs_s = s.checked_abs()?;
            let mut found = None;
            for b in 0..rank {
                if !used[b] && base_dims[b] == dim && (dim == 1 || base_strides[b] == abs_s) {
                    found = Some(b);
                    break;
                }
            }
            let b = found?;
            used[b] = true;
            let rev = dim > 1 && s < 0;
            if rev {
                expected_offset += (dim as isize - 1) * base_strides[b];
            }
            view_to_base.push((b, rev));
        }
        if view_layout.offset() as isize != expected_offset {
            return None;
        }
        Some(Self {
            view_to_base,
            base_dims: base_dims.to_vec(),
        })
    }
    fn compose(&self, view_layout: &Layout) -> Option<AffineMap> {
        let rank = self.base_dims.len();
        if view_layout.shape().dims().len() != rank {
            return None;
        }
        let mut offset = view_layout.offset() as isize;
        let mut strides = vec![0isize; rank];
        for (a, &(b, rev)) in self.view_to_base.iter().enumerate() {
            let vs = view_layout.strides()[a];
            if rev {
                offset += (self.base_dims[b] as isize - 1) * vs;
                strides[b] = -vs;
            } else {
                strides[b] = vs;
            }
        }
        Some(AffineMap {
            offset,
            strides,
            has_k: false,
            k_stride: 0,
        })
    }
}

fn affine_from_layout(layout: &Layout, base_rank: usize, k_dim: usize) -> Option<AffineMap> {
    let dims = layout.shape().dims();
    if k_dim > 1 && dims.len() == base_rank + 1 && dims[base_rank] == k_dim {
        let k_stride = layout.strides()[base_rank];
        Some(AffineMap {
            offset: layout.offset() as isize,
            strides: layout.strides()[..base_rank].to_vec(),
            has_k: k_stride != 0,
            k_stride,
        })
    } else if dims.len() == base_rank {
        Some(AffineMap {
            offset: layout.offset() as isize,
            strides: layout.strides().to_vec(),
            has_k: false,
            k_stride: 0,
        })
    } else {
        None
    }
}

fn is_barrier_op(op: &Operation) -> bool {
    match op {
        Operation::Matmul
        | Operation::Bmm(_)
        | Operation::Sdpa(_)
        | Operation::Conv1d(_)
        | Operation::Conv2d(_)
        | Operation::ConvTranspose1d(_)
        | Operation::ConvTranspose2d(_)
        | Operation::Pad(_)
        | Operation::Cat(_)
        | Operation::Stft { .. }
        | Operation::CenteredStft { .. }
        | Operation::Istft { .. }
        | Operation::Norm(_)
        | Operation::Softmax(_)
        | Operation::Rope(_) => true,
        Operation::Reduce(spec) => spec.operation().requires_input(),
        _ => false,
    }
}

/// Analyze a prepared training DAG and build a general fused whole-step plan if eligible.
pub(crate) fn analyze_step_fusion(
    graph: &DefinitionGraph,
    value_gradients: &[Option<usize>],
    updated_slots: &[bool],
) -> Option<FusedStepPlan> {
    // 1. Identify single differentiable scalar root (all other roots must be detached).
    let mut active_root = None;
    for (pos, &root) in graph.roots.iter().enumerate() {
        let def = &graph.values[root.index];
        if def.requires_grad {
            if active_root.is_some() {
                return None;
            }
            if !def.layout.shape().dims().is_empty() || def.layout.offset() != 0 {
                return None;
            }
            active_root = Some((pos, root.index));
        }
    }
    let (seed_root_pos, loss_val_idx) = active_root?;
    let Kind::Compute {
        operation: Operation::Reduce(loss_spec),
        inputs: [Some(loss_in_idx), None, None],
    } = &graph.values[loss_val_idx].kind
    else {
        return None;
    };
    let loss_in_idx = *loss_in_idx;
    let loss_in_dims = graph.values[loss_in_idx].layout.shape().dims();
    if loss_in_dims.is_empty() || graph.values[loss_in_idx].layout.shape().is_empty() {
        return None;
    }
    if loss_spec.axes().len() != loss_in_dims.len() {
        return None;
    }
    let loss_op = loss_spec.operation();
    if loss_op.requires_input() {
        return None;
    }

    // 2. Partition at the last heavy linear barrier (Matmul/Conv1d/Conv2d), if any.
    let mut last_barrier_pos = None;
    for (pos, &val_idx) in graph.forward.iter().enumerate() {
        if let Kind::Compute { operation, .. } = &graph.values[val_idx].kind {
            if is_barrier_op(operation) {
                last_barrier_pos = Some(pos);
            }
        }
    }
    let region_start_pos = last_barrier_pos.map_or(0, |p| p + 1);
    let barrier_out_val = last_barrier_pos.map(|p| graph.forward[p]);
    let barrier_out_grad_slot = if let Some(b_val) = barrier_out_val {
        let def = &graph.values[b_val];
        if !def.requires_grad || !def.layout.is_contiguous() || def.layout.offset() != 0 {
            return None;
        }
        Some(value_gradients[def.root]?)
    } else {
        None
    };

    // Track roots and gradient slots touched by pre-barrier steps so residual skip connections
    // never mutate a shared leaf before pre-barrier backward finishes and accumulate gradients.
    let mut pre_barrier_roots = vec![false; graph.values.len()];
    let mut pre_barrier_grad_slots = vec![false; updated_slots.len()];
    if let Some(last_b) = last_barrier_pos {
        for &val_idx in &graph.forward[..=last_b] {
            pre_barrier_roots[graph.values[val_idx].root] = true;
            if let Kind::Compute { inputs, .. } = &graph.values[val_idx].kind {
                for inp in inputs.iter().flatten().copied() {
                    let r = graph.values[inp].root;
                    pre_barrier_roots[r] = true;
                    if let Some(s) = value_gradients[r] {
                        if s < pre_barrier_grad_slots.len() {
                            pre_barrier_grad_slots[s] = true;
                        }
                    }
                }
            }
        }
    }

    // Mark which compute values belong to the fusible region (excluding final loss reduce).
    let mut in_region = vec![false; graph.values.len()];
    let mut region_compute = Vec::new();
    let mut small_reduce_val = None;
    for &val_idx in &graph.forward[region_start_pos..] {
        if val_idx == loss_val_idx {
            continue;
        }
        let Kind::Compute { operation, .. } = &graph.values[val_idx].kind else {
            return None;
        };
        match operation {
            Operation::Unary(_) | Operation::Binary(_) => {}
            Operation::Reduce(spec) => {
                if spec.operation().requires_input() || small_reduce_val.is_some() {
                    return None;
                }
                if spec.axes().len() != 1 || spec.keep_dims() {
                    return None;
                }
                let src_idx = match graph.values[val_idx].kind {
                    Kind::Compute {
                        inputs: [Some(s), None, None],
                        ..
                    } => s,
                    _ => return None,
                };
                let src_dims = graph.values[src_idx].layout.shape().dims();
                let axis = spec.axes()[0];
                if axis + 1 != src_dims.len() || src_dims[axis] < 2 || src_dims[axis] > 8 {
                    return None;
                }
                small_reduce_val = Some((val_idx, src_dims[axis]));
            }
            _ => return None,
        }
        in_region[val_idx] = true;
        region_compute.push(val_idx);
    }

    // Ensure the final loss node is the last forward step in the region.
    if *graph.forward.last()? != loss_val_idx {
        return None;
    }

    let k_dim = small_reduce_val.map_or(1, |(_, k)| k);
    let small_reduce_idx = small_reduce_val.map(|(v, _)| v);

    // 3. Detect if a bijective view alias (Slice(±1)/Permute) connects a contiguous base domain
    //    to the loss domain. Pull everything back into base domain so reads stay coalesced!
    let mut post_view = vec![false; graph.values.len()];
    let mut bijection: Option<(Vec<usize>, Bijection)> = None;

    // Find any alias inside the region whose root is an in-region compute node or barrier_out
    // and whose layout is non-contiguous (e.g. reversed + permuted prediction).
    for &val_idx in &region_compute {
        let Kind::Compute { inputs, .. } = &graph.values[val_idx].kind else {
            continue;
        };
        for inp in inputs.iter().flatten().copied() {
            let def = &graph.values[inp];
            if matches!(def.kind, Kind::Alias(_))
                && (in_region[def.root] || barrier_out_val == Some(def.root))
                && (!def.layout.is_contiguous() || def.layout.offset() != 0)
            {
                let base_dims = graph.values[def.root].layout.shape().dims();
                let bij = Bijection::from_alias_layout(base_dims, &def.layout)?;
                if let Some((ref existing_base, _)) = bijection {
                    if existing_base != base_dims {
                        return None;
                    }
                } else {
                    bijection = Some((base_dims.to_vec(), bij));
                }
            }
        }
    }

    let (domain, bij) = match bijection {
        Some((base_dims, bij)) => (base_dims, bij),
        None => (loss_in_dims.to_vec(), Bijection::identity(loss_in_dims)),
    };
    let base_rank = domain.len();
    let total_domain: usize = domain.iter().product();
    if total_domain == 0 {
        return None;
    }

    // Determine which compute nodes live in pre-k (`[domain, k]`), pre-view (`domain`),
    // or post-view (`loss_in_dims`) spaces.
    let mut is_pre_k = vec![false; graph.values.len()];
    for &val_idx in &region_compute {
        let dims = graph.values[val_idx].layout.shape().dims();
        if k_dim > 1
            && dims.len() == base_rank + 1
            && dims[..base_rank] == domain[..]
            && dims[base_rank] == k_dim
        {
            if small_reduce_idx == Some(val_idx) {
                return None;
            }
            is_pre_k[val_idx] = true;
        } else if dims == domain.as_slice() {
            let Kind::Compute { inputs, .. } = &graph.values[val_idx].kind else {
                return None;
            };
            let any_post = inputs.iter().flatten().copied().any(|inp| {
                let d = &graph.values[inp];
                post_view[inp]
                    || (matches!(d.kind, Kind::Alias(_))
                        && (in_region[d.root] || barrier_out_val == Some(d.root))
                        && (!d.layout.is_contiguous() || d.layout.offset() != 0))
            });
            post_view[val_idx] = any_post;
        } else if dims == loss_in_dims {
            post_view[val_idx] = true;
        } else {
            return None;
        }
    }

    // Verify barrier_out (if present) matches `domain`.
    if let Some(b_val) = barrier_out_val {
        let b_dims = graph.values[b_val].layout.shape().dims();
        if b_dims != domain.as_slice() {
            return None;
        }
    }

    // 4. Lower all region nodes into SSA `FwdInstr`s and external inputs.
    let mut exts: Vec<ExtInput> = Vec::new();
    let mut val_to_reg: Vec<Option<usize>> = vec![None; graph.values.len()];
    let mut instrs: Vec<FwdInstr> = Vec::new();
    let mut is_vec: Vec<bool> = Vec::new();
    let mut req_grad: Vec<bool> = Vec::new();

    let get_or_load = |inp_idx: usize,
                       consumer_pre_k: bool,
                       consumer_post_view: bool,
                       consumer_dims: &[usize],
                       val_to_reg: &[Option<usize>],
                       exts: &mut Vec<ExtInput>,
                       instrs: &mut Vec<FwdInstr>,
                       is_vec: &mut Vec<bool>,
                       req_grad: &mut Vec<bool>|
     -> Option<usize> {
        let def = &graph.values[inp_idx];
        let root = def.root;
        // Case 1: internal region compute value (possibly via a view alias)
        if in_region[root] {
            if matches!(def.kind, Kind::Alias(_)) {
                if !def.layout.is_contiguous() || def.layout.offset() != 0 {
                    // Must match the discovered bijection into `post_view`.
                    if !consumer_post_view {
                        return None;
                    }
                    let check_bij = Bijection::from_alias_layout(&domain, &def.layout)?;
                    if check_bij.view_to_base != bij.view_to_base {
                        return None;
                    }
                } else if def.layout.shape().dims() != graph.values[root].layout.shape().dims() {
                    return None;
                }
            }
            return val_to_reg[root];
        }
        // Case 2: barrier output or external leaf / pre-barrier read-only value
        let is_barrier_out = barrier_out_val == Some(root);
        if !is_barrier_out && !matches!(graph.values[root].kind, Kind::Leaf) {
            // Pre-barrier non-leaf value: only allowed if non-differentiable from tail.
            if def.requires_grad {
                return None;
            }
        }
        let bcast = def.layout.broadcast_to(consumer_dims).ok()?;
        let map = if consumer_post_view {
            if consumer_pre_k {
                return None;
            }
            bij.compose(&bcast)?
        } else {
            affine_from_layout(&bcast, base_rank, if consumer_pre_k { k_dim } else { 1 })?
        };
        let grad_slot = if def.requires_grad {
            value_gradients[root]
        } else {
            None
        };
        let param_len = graph.values[root].layout.shape().len();
        let used_pre_barrier = pre_barrier_roots[root]
            || grad_slot.is_some_and(|s| pre_barrier_grad_slots.get(s).copied().unwrap_or(false));
        let update_sgd = grad_slot.is_some_and(|s| updated_slots.get(s).copied().unwrap_or(false))
            && !is_barrier_out
            && !used_pre_barrier;

        // Reuse existing ExtInput entry if same owner_root (handles tied parameters cleanly!).
        let ext_id = if let Some(existing) = exts.iter().position(|e| e.owner_root == root) {
            if exts[existing].grad_slot.is_some() {
                // Tied trainable parameter: require identical map for clean single-pass reduction.
                let same_map = exts[existing].param_map.as_ref().is_some_and(|m| {
                    m.offset == map.offset
                        && m.strides == map.strides
                        && m.has_k == map.has_k
                        && m.k_stride == map.k_stride
                });
                if !same_map {
                    return None;
                }
            }
            existing
        } else {
            let role = if is_barrier_out {
                ExtRole::BarrierOut
            } else if grad_slot.is_none() {
                ExtRole::ReadOnly
            } else {
                ExtRole::SmallGlobal // classified precisely in step 5
            };
            exts.push(ExtInput {
                owner_root: root,
                grad_slot,
                param_len,
                update_sgd,
                role,
                param_map: if grad_slot.is_some() && !is_barrier_out {
                    Some(map.clone())
                } else {
                    None
                },
            });
            exts.len() - 1
        };

        let reg = instrs.len();
        instrs.push(FwdInstr::LoadExt { ext: ext_id, map });
        is_vec.push(consumer_pre_k);
        req_grad.push(def.requires_grad);
        Some(reg)
    };

    for &val_idx in &region_compute {
        let def = &graph.values[val_idx];
        let Kind::Compute { operation, inputs } = &def.kind else {
            return None;
        };
        let c_pre_k = is_pre_k[val_idx];
        let c_post_view = post_view[val_idx];
        let c_dims = def.layout.shape().dims();
        let reg = match operation {
            Operation::Unary(op) => {
                let src = get_or_load(
                    inputs[0]?,
                    c_pre_k,
                    c_post_view,
                    c_dims,
                    &val_to_reg,
                    &mut exts,
                    &mut instrs,
                    &mut is_vec,
                    &mut req_grad,
                )?;
                let r = instrs.len();
                instrs.push(FwdInstr::Unary { op: *op, src });
                is_vec.push(c_pre_k);
                req_grad.push(def.requires_grad);
                r
            }
            Operation::Binary(op) => {
                let lhs = get_or_load(
                    inputs[0]?,
                    c_pre_k,
                    c_post_view,
                    c_dims,
                    &val_to_reg,
                    &mut exts,
                    &mut instrs,
                    &mut is_vec,
                    &mut req_grad,
                )?;
                let rhs = get_or_load(
                    inputs[1]?,
                    c_pre_k,
                    c_post_view,
                    c_dims,
                    &val_to_reg,
                    &mut exts,
                    &mut instrs,
                    &mut is_vec,
                    &mut req_grad,
                )?;
                let r = instrs.len();
                instrs.push(FwdInstr::Binary { op: *op, lhs, rhs });
                is_vec.push(c_pre_k);
                req_grad.push(def.requires_grad);
                r
            }
            Operation::Reduce(spec) => {
                let src_idx = inputs[0]?;
                if !in_region[graph.values[src_idx].root] || !is_pre_k[graph.values[src_idx].root] {
                    return None;
                }
                let src = val_to_reg[graph.values[src_idx].root]?;
                let r = instrs.len();
                instrs.push(FwdInstr::SmallReduce {
                    op: spec.operation(),
                    src,
                });
                is_vec.push(false);
                req_grad.push(def.requires_grad);
                r
            }
            _ => return None,
        };
        val_to_reg[val_idx] = Some(reg);
    }

    // Ensure the last SSA register produces `loss_in`.
    let loss_in_root = graph.values[loss_in_idx].root;
    if val_to_reg[loss_in_root] != Some(instrs.len().checked_sub(1)?) {
        return None;
    }
    if is_vec[instrs.len() - 1] {
        return None;
    }

    // Ensure no unused differentiable leaves exist outside the region when in WholeRegion mode.
    if last_barrier_pos.is_none() {
        let mut covered_slots = vec![false; updated_slots.len()];
        for ext in &exts {
            if let Some(s) = ext.grad_slot {
                covered_slots[s] = true;
            }
        }
        for (slot, &upd) in updated_slots.iter().enumerate() {
            if upd && !covered_slots[slot] {
                return None;
            }
        }
    }

    // 5. Classify trainable leaf parameter reductions (GroupKept vs SmallGlobal vs FullElementwise).
    //    Check that every trainable parameter's AffineMap maps its active axes bijectively
    //    with contiguous positive strides onto 0..param_len.
    let mut best_group_mask: Option<(Vec<bool>, usize)> = None;
    let mut ext_masks: Vec<Option<(Vec<bool>, bool)>> = vec![None; exts.len()];

    for (eid, ext) in exts.iter().enumerate() {
        if ext.grad_slot.is_none() || ext.role == ExtRole::BarrierOut {
            continue;
        }
        let map = ext.param_map.as_ref()?;
        if map.offset != 0 {
            return None;
        }
        let mut mask = vec![false; base_rank];
        let mut active_prod = if map.has_k {
            if map.k_stride != 1 {
                return None;
            }
            k_dim
        } else {
            1
        };
        // Verify contiguous row-major parameter indexing over active axes.
        let mut expected_stride = active_prod as isize;
        for a in (0..base_rank).rev() {
            let s = map.strides[a];
            if s == 0 {
                continue;
            }
            if domain[a] == 1 {
                continue;
            }
            if s != expected_stride {
                return None;
            }
            mask[a] = true;
            active_prod *= domain[a];
            expected_stride *= domain[a] as isize;
        }
        if active_prod != ext.param_len {
            return None;
        }
        ext_masks[eid] = Some((mask.clone(), map.has_k));

        let is_full = !map.has_k && active_prod == total_domain && !mask.iter().any(|&m| !m);
        let touches_inner = domain[base_rank - 1] > 1 && mask[base_rank - 1];
        if !is_full
            && !map.has_k
            && !touches_inner
            && active_prod > 1
            && best_group_mask
                .as_ref()
                .is_none_or(|(_, g)| active_prod > *g)
        {
            best_group_mask = Some((mask, active_prod));
        }
    }

    let (kept_axes, red_axes, group_len) = if let Some((mask, g_len)) = best_group_mask {
        let kept: Vec<usize> = (0..base_rank).filter(|&a| mask[a]).collect();
        let red: Vec<usize> = (0..base_rank).filter(|&a| !mask[a]).collect();
        (kept, red, g_len)
    } else {
        (Vec::new(), (0..base_rank).collect(), 1)
    };
    let red_len = total_domain / group_len;

    // Assign final ExtRole to every trainable parameter.
    for (eid, ext) in exts.iter_mut().enumerate() {
        if ext.grad_slot.is_none() || ext.role == ExtRole::BarrierOut {
            continue;
        }
        let (mask, has_k) = ext_masks[eid].as_ref()?;
        let is_full = !has_k && ext.param_len == total_domain;
        let matches_group =
            !has_k && group_len > 1 && (0..base_rank).all(|a| mask[a] == kept_axes.contains(&a));
        if matches_group {
            ext.role = ExtRole::GroupKept;
        } else if is_full && last_barrier_pos.is_some() {
            ext.role = ExtRole::FullElementwise;
        } else if ext.param_len <= 32 {
            ext.role = ExtRole::SmallGlobal;
        } else {
            return None;
        }
    }

    let mut preinitialized_grad_slots = Vec::new();
    if let Some(slot) = barrier_out_grad_slot {
        preinitialized_grad_slots.push(slot);
    }
    for ext in &exts {
        if let Some(slot) = ext.grad_slot {
            if !preinitialized_grad_slots.contains(&slot) {
                preinitialized_grad_slots.push(slot);
            }
        }
    }

    Some(FusedStepPlan {
        barrier_step: last_barrier_pos,
        preinitialized_grad_slots,
        region: RegionPlan {
            domain,
            total_domain,
            k_dim,
            loss_op,
            seed_root_pos,
            exts,
            instrs,
            is_vec,
            req_grad,
            kept_axes,
            red_axes,
            group_len,
            red_len,
        },
        updated_slots: updated_slots.to_vec(),
    })
}

// ================================================================================================
// General CPU L1-Tiled Fused Executor (with optional oneMKL VML_HA on 256-float L1 stack tiles)
// ================================================================================================

const CPU_TILE: usize = 256;

#[inline(always)]
fn eval_unary_scalar(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Neg => f32::from_bits(x.to_bits() ^ 0x8000_0000),
        UnaryOp::Abs => f32::from_bits(x.to_bits() & 0x7fff_ffff),
        UnaryOp::Square => x * x,
        UnaryOp::Reciprocal => x.recip(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Relu => {
            if x < 0.0 {
                0.0
            } else {
                x
            }
        }
        UnaryOp::Sigmoid => crate::pointwise::sigmoid(x),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Silu => {
            if x == f32::NEG_INFINITY {
                -0.0
            } else {
                x * crate::pointwise::sigmoid(x)
            }
        }
        UnaryOp::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
    }
}

#[inline(always)]
fn eval_unary_bwd_scalar(op: UnaryOp, x: f32, y: f32, d: f32) -> f32 {
    match op {
        UnaryOp::Neg => f32::from_bits(d.to_bits() ^ 0x8000_0000),
        UnaryOp::Abs => {
            if x.is_nan() {
                f32::from_bits(0x7fc0_0000)
            } else if x > 0.0 {
                d
            } else if x < 0.0 {
                f32::from_bits(d.to_bits() ^ 0x8000_0000)
            } else {
                0.0
            }
        }
        UnaryOp::Square => (2.0 * x) * d,
        UnaryOp::Reciprocal => f32::from_bits(((d / x) / x).to_bits() ^ 0x8000_0000),
        UnaryOp::Sqrt => d / (2.0 * y),
        UnaryOp::Exp => y * d,
        UnaryOp::Log => d / x,
        UnaryOp::Relu => {
            if x.is_nan() {
                f32::from_bits(0x7fc0_0000)
            } else if x > 0.0 {
                d
            } else {
                0.0
            }
        }
        UnaryOp::Sigmoid => (y * (1.0 - y)) * d,
        UnaryOp::Tanh => ((1.0 - y) * (1.0 + y)) * d,
        UnaryOp::Silu => {
            let factor = if x == f32::INFINITY {
                1.0
            } else if x == f32::NEG_INFINITY {
                -0.0
            } else {
                let s = crate::pointwise::sigmoid(x);
                s + x * (s * (1.0 - s))
            };
            factor * d
        }
        UnaryOp::Softplus => crate::pointwise::sigmoid(x) * d,
    }
}

fn eval_unary_tile(op: UnaryOp, src: &[f32], dst: &mut [f32]) {
    #[cfg(feature = "mkl")]
    if src.len() >= 32 {
        match op {
            UnaryOp::Tanh => {
                crate::pointwise::vml::tanh(src, dst);
                return;
            }
            UnaryOp::Exp => {
                crate::pointwise::vml::exp(src, dst);
                return;
            }
            UnaryOp::Log => {
                crate::pointwise::vml::ln(src, dst);
                return;
            }
            UnaryOp::Sqrt => {
                crate::pointwise::vml::sqrt(src, dst);
                return;
            }
            UnaryOp::Sigmoid => {
                let mut tmp = [0.0f32; CPU_TILE];
                let mut e = [0.0f32; CPU_TILE];
                for (xs, ys) in src.chunks(CPU_TILE).zip(dst.chunks_mut(CPU_TILE)) {
                    let n = xs.len();
                    for (t, &x) in tmp[..n].iter_mut().zip(xs) {
                        *t = -x.abs();
                    }
                    crate::pointwise::vml::exp(&tmp[..n], &mut e[..n]);
                    for ((y, &x), &ev) in ys.iter_mut().zip(xs).zip(&e[..n]) {
                        *y = if x >= 0.0 {
                            1.0 / (1.0 + ev)
                        } else {
                            ev / (1.0 + ev)
                        };
                    }
                }
                return;
            }
            _ => {}
        }
    }
    match op {
        UnaryOp::Neg => {
            for (o, &x) in dst.iter_mut().zip(src) {
                *o = f32::from_bits(x.to_bits() ^ 0x8000_0000);
            }
        }
        UnaryOp::Abs => {
            for (o, &x) in dst.iter_mut().zip(src) {
                *o = f32::from_bits(x.to_bits() & 0x7fff_ffff);
            }
        }
        UnaryOp::Square => {
            for (o, &x) in dst.iter_mut().zip(src) {
                *o = x * x;
            }
        }
        UnaryOp::Relu => {
            for (o, &x) in dst.iter_mut().zip(src) {
                *o = if x < 0.0 { 0.0 } else { x };
            }
        }
        _ => {
            for (o, &x) in dst.iter_mut().zip(src) {
                *o = eval_unary_scalar(op, x);
            }
        }
    }
}

fn eval_binary_tile(op: BinaryOp, a: &[f32], b: &[f32], dst: &mut [f32]) {
    match op {
        BinaryOp::Add => {
            for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *o = x + y;
            }
        }
        BinaryOp::Sub => {
            for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *o = x - y;
            }
        }
        BinaryOp::Mul => {
            for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *o = x * y;
            }
        }
        BinaryOp::Div => {
            for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *o = x / y;
            }
        }
    }
}

fn accum_unary_bwd_tile(op: UnaryOp, x: &[f32], y: &[f32], d: &[f32], dst: &mut [f32], init: bool) {
    match (op, init) {
        (UnaryOp::Square, true) => {
            for ((o, &xv), &dv) in dst.iter_mut().zip(x).zip(d) {
                *o = (2.0 * xv) * dv;
            }
        }
        (UnaryOp::Square, false) => {
            for ((o, &xv), &dv) in dst.iter_mut().zip(x).zip(d) {
                *o += (2.0 * xv) * dv;
            }
        }
        (UnaryOp::Tanh, true) => {
            for ((o, &yv), &dv) in dst.iter_mut().zip(y).zip(d) {
                *o = ((1.0 - yv) * (1.0 + yv)) * dv;
            }
        }
        (UnaryOp::Tanh, false) => {
            for ((o, &yv), &dv) in dst.iter_mut().zip(y).zip(d) {
                *o += ((1.0 - yv) * (1.0 + yv)) * dv;
            }
        }
        (UnaryOp::Sigmoid, true) => {
            for ((o, &yv), &dv) in dst.iter_mut().zip(y).zip(d) {
                *o = (yv * (1.0 - yv)) * dv;
            }
        }
        (UnaryOp::Sigmoid, false) => {
            for ((o, &yv), &dv) in dst.iter_mut().zip(y).zip(d) {
                *o += (yv * (1.0 - yv)) * dv;
            }
        }
        (_, true) => {
            for (((o, &xv), &yv), &dv) in dst.iter_mut().zip(x).zip(y).zip(d) {
                *o = eval_unary_bwd_scalar(op, xv, yv, dv);
            }
        }
        (_, false) => {
            for (((o, &xv), &yv), &dv) in dst.iter_mut().zip(x).zip(y).zip(d) {
                *o += eval_unary_bwd_scalar(op, xv, yv, dv);
            }
        }
    }
}

fn accum_binary_bwd_tile(
    op: BinaryOp,
    lhs: bool,
    a: &[f32],
    b: &[f32],
    d: &[f32],
    dst: &mut [f32],
    init: bool,
) {
    match (op, lhs, init) {
        (BinaryOp::Add, _, true) | (BinaryOp::Sub, true, true) => dst.copy_from_slice(d),
        (BinaryOp::Add, _, false) | (BinaryOp::Sub, true, false) => {
            for (o, &dv) in dst.iter_mut().zip(d) {
                *o += dv;
            }
        }
        (BinaryOp::Sub, false, true) => {
            for (o, &dv) in dst.iter_mut().zip(d) {
                *o = f32::from_bits(dv.to_bits() ^ 0x8000_0000);
            }
        }
        (BinaryOp::Sub, false, false) => {
            for (o, &dv) in dst.iter_mut().zip(d) {
                *o += f32::from_bits(dv.to_bits() ^ 0x8000_0000);
            }
        }
        (BinaryOp::Mul, true, true) => {
            for ((o, &bv), &dv) in dst.iter_mut().zip(b).zip(d) {
                *o = dv * bv;
            }
        }
        (BinaryOp::Mul, true, false) => {
            for ((o, &bv), &dv) in dst.iter_mut().zip(b).zip(d) {
                *o += dv * bv;
            }
        }
        (BinaryOp::Mul, false, true) => {
            for ((o, &av), &dv) in dst.iter_mut().zip(a).zip(d) {
                *o = dv * av;
            }
        }
        (BinaryOp::Mul, false, false) => {
            for ((o, &av), &dv) in dst.iter_mut().zip(a).zip(d) {
                *o += dv * av;
            }
        }
        (BinaryOp::Div, true, true) => {
            for ((o, &bv), &dv) in dst.iter_mut().zip(b).zip(d) {
                *o = dv / bv;
            }
        }
        (BinaryOp::Div, true, false) => {
            for ((o, &bv), &dv) in dst.iter_mut().zip(b).zip(d) {
                *o += dv / bv;
            }
        }
        (BinaryOp::Div, false, true) => {
            for (((o, &av), &bv), &dv) in dst.iter_mut().zip(a).zip(b).zip(d) {
                let v = (dv / bv) * (av / bv);
                *o = f32::from_bits(v.to_bits() ^ 0x8000_0000);
            }
        }
        (BinaryOp::Div, false, false) => {
            for (((o, &av), &bv), &dv) in dst.iter_mut().zip(a).zip(b).zip(d) {
                let v = (dv / bv) * (av / bv);
                *o += f32::from_bits(v.to_bits() ^ 0x8000_0000);
            }
        }
    }
}

#[inline(always)]
fn sum_slice_simd8(s: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = s.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        for j in 0..8 {
            acc[j] += c[j];
        }
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for &v in rem {
        total += v;
    }
    total
}

/// Recognized zero-scratch single-pass CPU fused kernel patterns lowered directly from `RegionPlan`.
enum CpuFastKernel {
    None,
    /// `(x * scale + bias)^2.sum(-1)` vs `target` squared-error loss (`k == 2`).
    ComplexAffinePowerMse {
        ext_x: usize,
        map_x: AffineMap,
        ext_scale: usize,
        ext_bias: usize,
        ext_tgt: usize,
        map_tgt: AffineMap,
        use_2d_tile: bool,
    },
    /// `unary(x * scale + bias)` vs `target` squared-error loss (`k == 1`).
    AffineUnaryMse {
        ext_x: usize,
        map_x: AffineMap,
        ext_scale: usize,
        ext_bias: usize,
        ext_tgt: usize,
        map_tgt: AffineMap,
        unary: UnaryOp,
        use_2d_tile: bool,
    },
}

fn is_contiguous_group_map(region: &RegionPlan, map: &AffineMap) -> bool {
    if map.offset != 0 || map.has_k {
        return false;
    }
    for &a in &region.red_axes {
        if region.domain[a] > 1 && map.strides[a] != 0 {
            return false;
        }
    }
    let mut exp = 1isize;
    for &a in region.kept_axes.iter().rev() {
        let d = region.domain[a];
        if d > 1 && map.strides[a] != exp {
            return false;
        }
        exp *= d as isize;
    }
    true
}

fn map_zero_on_nontrivial_domain(region: &RegionPlan, map: &AffineMap) -> bool {
    region
        .domain
        .iter()
        .zip(&map.strides)
        .all(|(&d, &s)| d <= 1 || s == 0)
}

fn recognize_cpu_fast_kernel(region: &RegionPlan, inner_in_red: bool) -> CpuFastKernel {
    if !inner_in_red || region.kept_axes.is_empty() || region.instrs.len() < 4 {
        return CpuFastKernel::None;
    }
    let rank = region.domain.len();
    let inner_axis = rank - 1;
    let last_kept = *region.kept_axes.last().unwrap();
    let last = region.instrs.len() - 1;

    // Must end in Square(Sub(pred, tgt)) with no BarrierOut/FullElementwise outputs.
    if region
        .exts
        .iter()
        .any(|e| matches!(e.role, ExtRole::BarrierOut | ExtRole::FullElementwise))
    {
        return CpuFastKernel::None;
    }
    let FwdInstr::Unary {
        op: UnaryOp::Square,
        src: r_diff,
    } = region.instrs[last]
    else {
        return CpuFastKernel::None;
    };
    let FwdInstr::Binary {
        op: BinaryOp::Sub,
        lhs: d_lhs,
        rhs: d_rhs,
    } = region.instrs[r_diff]
    else {
        return CpuFastKernel::None;
    };
    let (r_pred, ext_tgt, map_tgt) = match (&region.instrs[d_lhs], &region.instrs[d_rhs]) {
        (_, FwdInstr::LoadExt { ext, map })
            if region.exts[*ext].role == ExtRole::ReadOnly
                && !region.is_vec[d_rhs]
                && !map.has_k =>
        {
            (d_lhs, *ext, map.clone())
        }
        (FwdInstr::LoadExt { ext, map }, _)
            if region.exts[*ext].role == ExtRole::ReadOnly
                && !region.is_vec[d_lhs]
                && !map.has_k =>
        {
            (d_rhs, *ext, map.clone())
        }
        _ => return CpuFastKernel::None,
    };

    // Case 1: ComplexAffinePowerMse (`k == 2`)
    if region.k_dim == 2 {
        let FwdInstr::SmallReduce {
            op: ReduceOp::Sum,
            src: r_sq,
        } = region.instrs[r_pred]
        else {
            return CpuFastKernel::None;
        };
        let FwdInstr::Unary {
            op: UnaryOp::Square,
            src: r_z,
        } = region.instrs[r_sq]
        else {
            return CpuFastKernel::None;
        };
        let FwdInstr::Binary {
            op: BinaryOp::Add,
            lhs: z_lhs,
            rhs: z_rhs,
        } = region.instrs[r_z]
        else {
            return CpuFastKernel::None;
        };
        let (r_mul, ext_bias) = match (&region.instrs[z_lhs], &region.instrs[z_rhs]) {
            (_, FwdInstr::LoadExt { ext, map })
                if region.exts[*ext].role == ExtRole::SmallGlobal
                    && region.exts[*ext].param_len == 2
                    && region.exts[*ext].grad_slot.is_some()
                    && map.offset == 0
                    && map.has_k
                    && map.k_stride == 1
                    && map_zero_on_nontrivial_domain(region, map) =>
            {
                (z_lhs, *ext)
            }
            (FwdInstr::LoadExt { ext, map }, _)
                if region.exts[*ext].role == ExtRole::SmallGlobal
                    && region.exts[*ext].param_len == 2
                    && region.exts[*ext].grad_slot.is_some()
                    && map.offset == 0
                    && map.has_k
                    && map.k_stride == 1
                    && map_zero_on_nontrivial_domain(region, map) =>
            {
                (z_rhs, *ext)
            }
            _ => return CpuFastKernel::None,
        };
        let FwdInstr::Binary {
            op: BinaryOp::Mul,
            lhs: m_lhs,
            rhs: m_rhs,
        } = region.instrs[r_mul]
        else {
            return CpuFastKernel::None;
        };
        let (ext_x, map_x, ext_scale) = match (&region.instrs[m_lhs], &region.instrs[m_rhs]) {
            (FwdInstr::LoadExt { ext: ex, map: mx }, FwdInstr::LoadExt { ext: es, map: ms })
                if region.exts[*ex].role == ExtRole::ReadOnly
                    && mx.is_k_contiguous(2)
                    && region.exts[*es].role == ExtRole::GroupKept
                    && region.exts[*es].grad_slot.is_some()
                    && is_contiguous_group_map(region, ms) =>
            {
                (*ex, mx.clone(), *es)
            }
            (FwdInstr::LoadExt { ext: es, map: ms }, FwdInstr::LoadExt { ext: ex, map: mx })
                if region.exts[*ex].role == ExtRole::ReadOnly
                    && mx.is_k_contiguous(2)
                    && region.exts[*es].role == ExtRole::GroupKept
                    && region.exts[*es].grad_slot.is_some()
                    && is_contiguous_group_map(region, ms) =>
            {
                (*ex, mx.clone(), *es)
            }
            _ => return CpuFastKernel::None,
        };
        // Ensure no other trainable exts exist in the region.
        let trainable_count = region.exts.iter().filter(|e| e.grad_slot.is_some()).count();
        if trainable_count != 2
            || ext_scale == ext_bias
            || region.exts[ext_scale].grad_slot == region.exts[ext_bias].grad_slot
        {
            return CpuFastKernel::None;
        }
        let use_2d_tile = map_tgt.strides[inner_axis].unsigned_abs() >= 16
            && map_tgt.strides[last_kept] == 1
            && region.domain[last_kept] >= 8;
        return CpuFastKernel::ComplexAffinePowerMse {
            ext_x,
            map_x,
            ext_scale,
            ext_bias,
            ext_tgt,
            map_tgt,
            use_2d_tile,
        };
    }

    // Case 2: AffineUnaryMse (`k == 1`)
    if region.k_dim == 1 {
        let FwdInstr::Unary {
            op: unary,
            src: r_z,
        } = region.instrs[r_pred]
        else {
            return CpuFastKernel::None;
        };
        if !matches!(
            unary,
            UnaryOp::Tanh | UnaryOp::Relu | UnaryOp::Sigmoid | UnaryOp::Silu | UnaryOp::Square
        ) {
            return CpuFastKernel::None;
        }
        let FwdInstr::Binary {
            op: BinaryOp::Add,
            lhs: z_lhs,
            rhs: z_rhs,
        } = region.instrs[r_z]
        else {
            return CpuFastKernel::None;
        };
        let (r_mul, ext_bias) = match (&region.instrs[z_lhs], &region.instrs[z_rhs]) {
            (_, FwdInstr::LoadExt { ext, map })
                if region.exts[*ext].role == ExtRole::GroupKept
                    && region.exts[*ext].grad_slot.is_some()
                    && is_contiguous_group_map(region, map) =>
            {
                (z_lhs, *ext)
            }
            (FwdInstr::LoadExt { ext, map }, _)
                if region.exts[*ext].role == ExtRole::GroupKept
                    && region.exts[*ext].grad_slot.is_some()
                    && is_contiguous_group_map(region, map) =>
            {
                (z_rhs, *ext)
            }
            _ => return CpuFastKernel::None,
        };
        let FwdInstr::Binary {
            op: BinaryOp::Mul,
            lhs: m_lhs,
            rhs: m_rhs,
        } = region.instrs[r_mul]
        else {
            return CpuFastKernel::None;
        };
        let (ext_x, map_x, ext_scale) = match (&region.instrs[m_lhs], &region.instrs[m_rhs]) {
            (FwdInstr::LoadExt { ext: ex, map: mx }, FwdInstr::LoadExt { ext: es, map: ms })
                if region.exts[*ex].role == ExtRole::ReadOnly
                    && mx.is_unit_inner()
                    && region.exts[*es].role == ExtRole::GroupKept
                    && region.exts[*es].grad_slot.is_some()
                    && is_contiguous_group_map(region, ms) =>
            {
                (*ex, mx.clone(), *es)
            }
            (FwdInstr::LoadExt { ext: es, map: ms }, FwdInstr::LoadExt { ext: ex, map: mx })
                if region.exts[*ex].role == ExtRole::ReadOnly
                    && mx.is_unit_inner()
                    && region.exts[*es].role == ExtRole::GroupKept
                    && region.exts[*es].grad_slot.is_some()
                    && is_contiguous_group_map(region, ms) =>
            {
                (*ex, mx.clone(), *es)
            }
            _ => return CpuFastKernel::None,
        };
        let trainable_count = region.exts.iter().filter(|e| e.grad_slot.is_some()).count();
        if trainable_count != 2
            || ext_scale == ext_bias
            || region.exts[ext_scale].grad_slot == region.exts[ext_bias].grad_slot
        {
            return CpuFastKernel::None;
        }
        let use_2d_tile = map_tgt.strides[inner_axis].unsigned_abs() >= 4
            && map_tgt.strides[last_kept] == 1
            && region.domain[last_kept] >= 4;
        return CpuFastKernel::AffineUnaryMse {
            ext_x,
            map_x,
            ext_scale,
            ext_bias,
            ext_tgt,
            map_tgt,
            unary,
            use_2d_tile,
        };
    }

    CpuFastKernel::None
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_affine_power_mse_contiguous_avx2(
    x: &[f32],
    tgt: &[f32],
    sc: f32,
    b0: f32,
    b1: f32,
    d4: f32,
) -> (f32, f32, f32) {
    let n = tgt.len();
    debug_assert_eq!(x.len(), n * 2);
    let mut asc = [0.0f32; 8];
    let mut ab0 = [0.0f32; 8];
    let mut ab1 = [0.0f32; 8];
    let chunks_t = tgt.chunks_exact(8);
    let rem_t = chunks_t.remainder();
    let chunks_x = x.chunks_exact(16);
    let rem_x = chunks_x.remainder();
    for (xc, tc) in chunks_x.zip(chunks_t) {
        let mut x0 = [0.0f32; 8];
        let mut x1 = [0.0f32; 8];
        for j in 0..8 {
            x0[j] = xc[j * 2];
            x1[j] = xc[j * 2 + 1];
        }
        for j in 0..8 {
            let z0 = x0[j] * sc + b0;
            let z1 = x1[j] * sc + b1;
            let diff = (z0 * z0 + z1 * z1) - tc[j];
            let c = diff * d4;
            let dz0 = c * z0;
            let dz1 = c * z1;
            ab0[j] += dz0;
            ab1[j] += dz1;
            asc[j] += dz0 * x0[j] + dz1 * x1[j];
        }
    }
    let mut g_sc =
        ((asc[0] + asc[1]) + (asc[2] + asc[3])) + ((asc[4] + asc[5]) + (asc[6] + asc[7]));
    let mut g_b0 =
        ((ab0[0] + ab0[1]) + (ab0[2] + ab0[3])) + ((ab0[4] + ab0[5]) + (ab0[6] + ab0[7]));
    let mut g_b1 =
        ((ab1[0] + ab1[1]) + (ab1[2] + ab1[3])) + ((ab1[4] + ab1[5]) + (ab1[6] + ab1[7]));
    for (j, &tv) in rem_t.iter().enumerate() {
        let x0 = rem_x[j * 2];
        let x1 = rem_x[j * 2 + 1];
        let z0 = x0 * sc + b0;
        let z1 = x1 * sc + b1;
        let diff = (z0 * z0 + z1 * z1) - tv;
        let c = diff * d4;
        let dz0 = c * z0;
        let dz1 = c * z1;
        g_b0 += dz0;
        g_b1 += dz1;
        g_sc += dz0 * x0 + dz1 * x1;
    }
    (g_sc, g_b0, g_b1)
}

#[inline(always)]
fn complex_affine_power_mse_contiguous(
    x: &[f32],
    tgt: &[f32],
    sc: f32,
    b0: f32,
    b1: f32,
    d4: f32,
) -> (f32, f32, f32) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        return unsafe { complex_affine_power_mse_contiguous_avx2(x, tgt, sc, b0, b1, d4) };
    }
    let n = tgt.len();
    debug_assert_eq!(x.len(), n * 2);
    let mut asc = [0.0f32; 4];
    let mut ab0 = [0.0f32; 4];
    let mut ab1 = [0.0f32; 4];
    let chunks_t = tgt.chunks_exact(4);
    let rem_t = chunks_t.remainder();
    let chunks_x = x.chunks_exact(8);
    let rem_x = chunks_x.remainder();
    for (xc, tc) in chunks_x.zip(chunks_t) {
        for j in 0..4 {
            let x0 = xc[j * 2];
            let x1 = xc[j * 2 + 1];
            let z0 = x0 * sc + b0;
            let z1 = x1 * sc + b1;
            let diff = (z0 * z0 + z1 * z1) - tc[j];
            let c = diff * d4;
            let dz0 = c * z0;
            let dz1 = c * z1;
            ab0[j] += dz0;
            ab1[j] += dz1;
            asc[j] += dz0 * x0 + dz1 * x1;
        }
    }
    let mut g_sc = (asc[0] + asc[1]) + (asc[2] + asc[3]);
    let mut g_b0 = (ab0[0] + ab0[1]) + (ab0[2] + ab0[3]);
    let mut g_b1 = (ab1[0] + ab1[1]) + (ab1[2] + ab1[3]);
    for (j, &tv) in rem_t.iter().enumerate() {
        let x0 = rem_x[j * 2];
        let x1 = rem_x[j * 2 + 1];
        let z0 = x0 * sc + b0;
        let z1 = x1 * sc + b1;
        let diff = (z0 * z0 + z1 * z1) - tv;
        let c = diff * d4;
        let dz0 = c * z0;
        let dz1 = c * z1;
        g_b0 += dz0;
        g_b1 += dz1;
        g_sc += dz0 * x0 + dz1 * x1;
    }
    (g_sc, g_b0, g_b1)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn execute_complex_affine_power_mse(
    region: &RegionPlan,
    coord: &mut [usize],
    outer_red_axes: &[usize],
    outer_red_len: usize,
    inner_len: usize,
    ext_reads: &[&[f32]],
    gradients: &mut [super::engine::Gradient<crate::Tensor>],
    d_root: f32,
    ext_x: usize,
    map_x: &AffineMap,
    ext_scale: usize,
    ext_bias: usize,
    ext_tgt: usize,
    map_tgt: &AffineMap,
    use_2d_tile: bool,
) {
    let rank = region.domain.len();
    let inner_axis = rank - 1;
    let x_buf = ext_reads[ext_x];
    let scale_buf = ext_reads[ext_scale];
    let bias_buf = ext_reads[ext_bias];
    let tgt_buf = ext_reads[ext_tgt];
    let b0 = bias_buf[0];
    let b1 = bias_buf[1];
    let d4 = 4.0f32 * d_root;
    let tgt_s_inner = map_tgt.strides[inner_axis];

    let scale_slot = region.exts[ext_scale].grad_slot.unwrap();
    let bias_slot = region.exts[ext_bias].grad_slot.unwrap();
    let mut total_b0 = 0.0f64;
    let mut total_b1 = 0.0f64;

    if use_2d_tile {
        // 2D L1 cache-blocked micro-transpose: read target unit-stride along `last_kept`
        // (100% 64-byte cache-line utilization), transpose into 2KB L1 stack tile, then run
        // the 4-way AVX2 contiguous complex loop.
        const G_TILE: usize = 16;
        const T_TILE: usize = 32;
        let last_kept = *region.kept_axes.last().unwrap();
        let last_kept_dim = region.domain[last_kept];
        let outer_kept_len = region.group_len / last_kept_dim;
        let x_s_lk = map_x.strides[last_kept];
        let g_scale_buf = gradients[scale_slot].data.data_mut();
        g_scale_buf.fill(0.0);

        let mut tgt_tile = [[0.0f32; T_TILE]; G_TILE];
        for outer in 0..outer_red_len {
            let mut rem_o = outer;
            for &a in outer_red_axes.iter().rev() {
                let d = region.domain[a];
                coord[a] = rem_o % d;
                rem_o /= d;
            }
            coord[inner_axis] = 0;

            for ok in 0..outer_kept_len {
                let mut rem_ok = ok;
                for &a in region.kept_axes[..region.kept_axes.len() - 1].iter().rev() {
                    let d = region.domain[a];
                    coord[a] = rem_ok % d;
                    rem_ok /= d;
                }
                coord[last_kept] = 0;
                let mut x_base_ok = map_x.offset;
                let mut t_base_ok = map_tgt.offset;
                for a in 0..rank {
                    x_base_ok += coord[a] as isize * map_x.strides[a];
                    t_base_ok += coord[a] as isize * map_tgt.strides[a];
                }

                let mut lk0 = 0usize;
                while lk0 < last_kept_dim {
                    let ng = (last_kept_dim - lk0).min(G_TILE);
                    let mut t0 = 0usize;
                    while t0 < inner_len {
                        let nt = (inner_len - t0).min(T_TILE);
                        // Sequential cache-line loads along `last_kept` (stride == 1)
                        for dt in 0..nt {
                            let row_off = (t_base_ok
                                + (t0 + dt) as isize * tgt_s_inner
                                + lk0 as isize) as usize;
                            let src_row = &tgt_buf[row_off..row_off + ng];
                            for dg in 0..ng {
                                tgt_tile[dg][dt] = src_row[dg];
                            }
                        }
                        for dg in 0..ng {
                            let g = ok * last_kept_dim + lk0 + dg;
                            let sc = scale_buf[g];
                            let x_off =
                                (x_base_ok + (lk0 + dg) as isize * x_s_lk + (t0 * 2) as isize)
                                    as usize;
                            let (g_sc, g_b0, g_b1) = complex_affine_power_mse_contiguous(
                                &x_buf[x_off..x_off + nt * 2],
                                &tgt_tile[dg][..nt],
                                sc,
                                b0,
                                b1,
                                d4,
                            );
                            g_scale_buf[g] += g_sc;
                            total_b0 += g_b0 as f64;
                            total_b1 += g_b1 as f64;
                        }
                        t0 += nt;
                    }
                    lk0 += ng;
                }
            }
        }
    } else {
        let g_scale_buf = gradients[scale_slot].data.data_mut();
        for g in 0..region.group_len {
            let mut rem_g = g;
            for &a in region.kept_axes.iter().rev() {
                let d = region.domain[a];
                coord[a] = rem_g % d;
                rem_g /= d;
            }
            let sc = scale_buf[g];
            let mut acc_sc = 0.0f32;
            for outer in 0..outer_red_len {
                let mut rem_o = outer;
                for &a in outer_red_axes.iter().rev() {
                    let d = region.domain[a];
                    coord[a] = rem_o % d;
                    rem_o /= d;
                }
                coord[inner_axis] = 0;
                let mut x_base = map_x.offset;
                let mut t_base = map_tgt.offset;
                for a in 0..rank {
                    x_base += coord[a] as isize * map_x.strides[a];
                    t_base += coord[a] as isize * map_tgt.strides[a];
                }
                let x_row = &x_buf[x_base as usize..x_base as usize + inner_len * 2];
                if tgt_s_inner == 1 {
                    let t_row = &tgt_buf[t_base as usize..t_base as usize + inner_len];
                    let (g_sc, g_b0, g_b1) =
                        complex_affine_power_mse_contiguous(x_row, t_row, sc, b0, b1, d4);
                    acc_sc += g_sc;
                    total_b0 += g_b0 as f64;
                    total_b1 += g_b1 as f64;
                } else {
                    let mut asc = 0.0f32;
                    let mut ab0 = 0.0f32;
                    let mut ab1 = 0.0f32;
                    for t in 0..inner_len {
                        let x0 = x_row[t * 2];
                        let x1 = x_row[t * 2 + 1];
                        let tv = tgt_buf[(t_base + t as isize * tgt_s_inner) as usize];
                        let z0 = x0 * sc + b0;
                        let z1 = x1 * sc + b1;
                        let diff = (z0 * z0 + z1 * z1) - tv;
                        let c = diff * d4;
                        let dz0 = c * z0;
                        let dz1 = c * z1;
                        ab0 += dz0;
                        ab1 += dz1;
                        asc += dz0 * x0 + dz1 * x1;
                    }
                    acc_sc += asc;
                    total_b0 += ab0 as f64;
                    total_b1 += ab1 as f64;
                }
            }
            g_scale_buf[g] = acc_sc;
        }
    }

    let g_bias_buf = gradients[bias_slot].data.data_mut();
    g_bias_buf[0] = total_b0 as f32;
    g_bias_buf[1] = total_b1 as f32;
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn affine_unary_mse_tile(
    unary: UnaryOp,
    x: &[f32],
    tgt: &[f32],
    sc: f32,
    bi: f32,
    d2: f32,
    z_buf: &mut [f32],
    y_buf: &mut [f32],
) -> (f32, f32) {
    let n = x.len();
    for i in 0..n {
        z_buf[i] = x[i] * sc + bi;
    }
    eval_unary_tile(unary, &z_buf[..n], &mut y_buf[..n]);
    let mut asc = [0.0f32; 4];
    let mut abi = [0.0f32; 4];
    let mut i = 0usize;
    match unary {
        UnaryOp::Tanh => {
            while i + 4 <= n {
                for j in 0..4 {
                    let yv = y_buf[i + j];
                    let dz = ((yv - tgt[i + j]) * d2) * ((1.0 - yv) * (1.0 + yv));
                    abi[j] += dz;
                    asc[j] += dz * x[i + j];
                }
                i += 4;
            }
            while i < n {
                let yv = y_buf[i];
                let dz = ((yv - tgt[i]) * d2) * ((1.0 - yv) * (1.0 + yv));
                abi[0] += dz;
                asc[0] += dz * x[i];
                i += 1;
            }
        }
        UnaryOp::Sigmoid => {
            while i + 4 <= n {
                for j in 0..4 {
                    let yv = y_buf[i + j];
                    let dz = ((yv - tgt[i + j]) * d2) * (yv * (1.0 - yv));
                    abi[j] += dz;
                    asc[j] += dz * x[i + j];
                }
                i += 4;
            }
            while i < n {
                let yv = y_buf[i];
                let dz = ((yv - tgt[i]) * d2) * (yv * (1.0 - yv));
                abi[0] += dz;
                asc[0] += dz * x[i];
                i += 1;
            }
        }
        UnaryOp::Relu => {
            while i + 4 <= n {
                for j in 0..4 {
                    let dz = if z_buf[i + j] > 0.0 {
                        (y_buf[i + j] - tgt[i + j]) * d2
                    } else {
                        0.0
                    };
                    abi[j] += dz;
                    asc[j] += dz * x[i + j];
                }
                i += 4;
            }
            while i < n {
                let dz = if z_buf[i] > 0.0 {
                    (y_buf[i] - tgt[i]) * d2
                } else {
                    0.0
                };
                abi[0] += dz;
                asc[0] += dz * x[i];
                i += 1;
            }
        }
        _ => {
            for j in 0..n {
                let dy = (y_buf[j] - tgt[j]) * d2;
                let dz = eval_unary_bwd_scalar(unary, z_buf[j], y_buf[j], dy);
                abi[j & 3] += dz;
                asc[j & 3] += dz * x[j];
            }
        }
    }
    (
        (asc[0] + asc[1]) + (asc[2] + asc[3]),
        (abi[0] + abi[1]) + (abi[2] + abi[3]),
    )
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn execute_affine_unary_mse(
    region: &RegionPlan,
    coord: &mut [usize],
    outer_red_axes: &[usize],
    outer_red_len: usize,
    inner_len: usize,
    ext_reads: &[&[f32]],
    gradients: &mut [super::engine::Gradient<crate::Tensor>],
    d_root: f32,
    ext_x: usize,
    map_x: &AffineMap,
    ext_scale: usize,
    ext_bias: usize,
    ext_tgt: usize,
    map_tgt: &AffineMap,
    unary: UnaryOp,
    use_2d_tile: bool,
) {
    let rank = region.domain.len();
    let inner_axis = rank - 1;
    let x_buf = ext_reads[ext_x];
    let scale_buf = ext_reads[ext_scale];
    let bias_buf = ext_reads[ext_bias];
    let tgt_buf = ext_reads[ext_tgt];
    let d2 = 2.0f32 * d_root;
    let tgt_s_inner = map_tgt.strides[inner_axis];
    let scale_slot = region.exts[ext_scale].grad_slot.unwrap();
    let bias_slot = region.exts[ext_bias].grad_slot.unwrap();

    let mut z_tile = [0.0f32; CPU_TILE];
    let mut y_tile = [0.0f32; CPU_TILE];
    let mut t_scratch = [0.0f32; CPU_TILE];

    if use_2d_tile {
        const G_TILE: usize = 8;
        const T_TILE: usize = 64;
        let last_kept = *region.kept_axes.last().unwrap();
        let last_kept_dim = region.domain[last_kept];
        let outer_kept_len = region.group_len / last_kept_dim;
        let x_s_lk = map_x.strides[last_kept];
        let (d_scale, d_bias): (&mut [f32], &mut [f32]) = if scale_slot < bias_slot {
            let (left, right) = gradients.split_at_mut(bias_slot);
            (left[scale_slot].data.data_mut(), right[0].data.data_mut())
        } else {
            let (left, right) = gradients.split_at_mut(scale_slot);
            (right[0].data.data_mut(), left[bias_slot].data.data_mut())
        };
        d_scale.fill(0.0);
        d_bias.fill(0.0);
        let mut tgt_tile = [[0.0f32; T_TILE]; G_TILE];

        for outer in 0..outer_red_len {
            let mut rem_o = outer;
            for &a in outer_red_axes.iter().rev() {
                let d = region.domain[a];
                coord[a] = rem_o % d;
                rem_o /= d;
            }
            coord[inner_axis] = 0;

            for ok in 0..outer_kept_len {
                let mut rem_ok = ok;
                for &a in region.kept_axes[..region.kept_axes.len() - 1].iter().rev() {
                    let d = region.domain[a];
                    coord[a] = rem_ok % d;
                    rem_ok /= d;
                }
                coord[last_kept] = 0;
                let mut x_base_ok = map_x.offset;
                let mut t_base_ok = map_tgt.offset;
                for a in 0..rank {
                    x_base_ok += coord[a] as isize * map_x.strides[a];
                    t_base_ok += coord[a] as isize * map_tgt.strides[a];
                }

                let mut lk0 = 0usize;
                while lk0 < last_kept_dim {
                    let ng = (last_kept_dim - lk0).min(G_TILE);
                    let mut t0 = 0usize;
                    while t0 < inner_len {
                        let nt = (inner_len - t0).min(T_TILE);
                        for dt in 0..nt {
                            let row_off = (t_base_ok
                                + (t0 + dt) as isize * tgt_s_inner
                                + lk0 as isize) as usize;
                            let src_row = &tgt_buf[row_off..row_off + ng];
                            for dg in 0..ng {
                                tgt_tile[dg][dt] = src_row[dg];
                            }
                        }
                        for dg in 0..ng {
                            let g = ok * last_kept_dim + lk0 + dg;
                            let x_off =
                                (x_base_ok + (lk0 + dg) as isize * x_s_lk + t0 as isize) as usize;
                            let (gs, gb) = affine_unary_mse_tile(
                                unary,
                                &x_buf[x_off..x_off + nt],
                                &tgt_tile[dg][..nt],
                                scale_buf[g],
                                bias_buf[g],
                                d2,
                                &mut z_tile,
                                &mut y_tile,
                            );
                            d_scale[g] += gs;
                            d_bias[g] += gb;
                        }
                        t0 += nt;
                    }
                    lk0 += ng;
                }
            }
        }
    } else {
        for g in 0..region.group_len {
            let mut rem_g = g;
            for &a in region.kept_axes.iter().rev() {
                let d = region.domain[a];
                coord[a] = rem_g % d;
                rem_g /= d;
            }
            let sc = scale_buf[g];
            let bi = bias_buf[g];
            let mut acc_s = 0.0f32;
            let mut acc_b = 0.0f32;
            for outer in 0..outer_red_len {
                let mut rem_o = outer;
                for &a in outer_red_axes.iter().rev() {
                    let d = region.domain[a];
                    coord[a] = rem_o % d;
                    rem_o /= d;
                }
                coord[inner_axis] = 0;
                let mut x_base = map_x.offset;
                let mut t_base = map_tgt.offset;
                for a in 0..rank {
                    x_base += coord[a] as isize * map_x.strides[a];
                    t_base += coord[a] as isize * map_tgt.strides[a];
                }
                let mut t0 = 0usize;
                while t0 < inner_len {
                    let n = (inner_len - t0).min(CPU_TILE);
                    let x_slice = &x_buf[(x_base + t0 as isize) as usize..][..n];
                    let t_slice = if tgt_s_inner == 1 {
                        &tgt_buf[(t_base + t0 as isize) as usize..][..n]
                    } else {
                        for i in 0..n {
                            t_scratch[i] =
                                tgt_buf[(t_base + (t0 + i) as isize * tgt_s_inner) as usize];
                        }
                        &t_scratch[..n]
                    };
                    let (gs, gb) = affine_unary_mse_tile(
                        unary,
                        x_slice,
                        t_slice,
                        sc,
                        bi,
                        d2,
                        &mut z_tile,
                        &mut y_tile,
                    );
                    acc_s += gs;
                    acc_b += gb;
                    t0 += n;
                }
            }
            gradients[scale_slot].data.data_mut()[g] = acc_s;
            gradients[bias_slot].data.data_mut()[g] = acc_b;
        }
    }
}

/// Reusable L1-resident tile workspace for zero-allocation CPU fused step execution.
pub(crate) struct CpuFusedScratch {
    fwd: Vec<Vec<f32>>,
    adj: Vec<Vec<f32>>,
    adj_init: Vec<bool>,
    small_acc: Vec<Vec<f32>>,
    group_acc: Vec<f32>,
    coord: Vec<usize>,
    outer_red_axes: Vec<usize>,
    outer_red_len: usize,
    inner_in_red: bool,
    inner_len: usize,
    all_flat_contiguous: bool,
    fast_kernel: CpuFastKernel,
    mixed_dl: Vec<f32>,
    mixed_dr: Vec<f32>,
    pub updated_in_region: Vec<bool>,
}

impl CpuFusedScratch {
    pub fn new(region: &RegionPlan, num_grad_slots: usize) -> Self {
        let rank = region.domain.len();
        let k = region.k_dim;
        let fwd = region
            .is_vec
            .iter()
            .map(|&v| vec![0.0f32; CPU_TILE * if v { k } else { 1 }])
            .collect();
        let adj = region
            .is_vec
            .iter()
            .map(|&v| vec![0.0f32; CPU_TILE * if v { k } else { 1 }])
            .collect();
        let small_acc = region
            .exts
            .iter()
            .map(|e| {
                if e.role == ExtRole::SmallGlobal {
                    vec![0.0f32; e.param_len]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let inner_axis = rank - 1;
        let inner_in_red = region.red_axes.contains(&inner_axis);
        let inner_len = if inner_in_red {
            region.domain[inner_axis]
        } else {
            1
        };
        let outer_red_axes: Vec<usize> = region
            .red_axes
            .iter()
            .copied()
            .filter(|&a| !inner_in_red || a != inner_axis)
            .collect();
        let outer_red_len: usize = outer_red_axes.iter().map(|&a| region.domain[a]).product();
        let all_flat_contiguous = region.group_len == 1
            && k == 1
            && region.exts.iter().all(|e| {
                matches!(
                    e.role,
                    ExtRole::ReadOnly | ExtRole::BarrierOut | ExtRole::FullElementwise
                )
            })
            && region.instrs.iter().all(|ins| match ins {
                FwdInstr::LoadExt { map, .. } => {
                    if map.offset != 0 || map.has_k {
                        return false;
                    }
                    let mut exp = 1isize;
                    for a in (0..rank).rev() {
                        if region.domain[a] > 1 && map.strides[a] != exp {
                            return false;
                        }
                        exp *= region.domain[a] as isize;
                    }
                    true
                }
                _ => true,
            });
        let mut updated_in_region = vec![false; num_grad_slots];
        for ext in &region.exts {
            if ext.update_sgd {
                if let Some(slot) = ext.grad_slot {
                    updated_in_region[slot] = true;
                }
            }
        }
        let fast_kernel = recognize_cpu_fast_kernel(region, inner_in_red);
        let mixed_len = if k > 1 { CPU_TILE * k } else { 0 };
        Self {
            fwd,
            adj,
            adj_init: vec![false; region.instrs.len()],
            small_acc,
            group_acc: vec![0.0f32; region.exts.len()],
            coord: vec![0usize; rank],
            outer_red_axes,
            outer_red_len,
            inner_in_red,
            inner_len,
            all_flat_contiguous,
            fast_kernel,
            mixed_dl: vec![0.0f32; mixed_len],
            mixed_dr: vec![0.0f32; mixed_len],
            updated_in_region,
        }
    }
}

/// Execute the fused region on CPU in L1-resident 256-element tiles with zero heap allocations.
/// Computes and stores all region gradients into `gradients`; parameter updates run cleanly
/// afterward under exclusive write guards without aliasing reads.
pub(crate) fn execute_cpu_region(
    region: &RegionPlan,
    scratch: &mut CpuFusedScratch,
    ext_reads: &[&[f32]],
    gradients: &mut [super::engine::Gradient<crate::Tensor>],
    seed_val: f32,
) {
    let rank = region.domain.len();
    let k = region.k_dim;
    let d_root = match region.loss_op {
        ReduceOp::Sum => seed_val,
        ReduceOp::Mean => seed_val / region.total_domain as f32,
        ReduceOp::Max | ReduceOp::Min | ReduceOp::LogSumExp => {
            unreachable!("nonlinear loss reductions are excluded by analyze_step_fusion")
        }
    };

    match &scratch.fast_kernel {
        CpuFastKernel::ComplexAffinePowerMse {
            ext_x,
            map_x,
            ext_scale,
            ext_bias,
            ext_tgt,
            map_tgt,
            use_2d_tile,
        } => {
            execute_complex_affine_power_mse(
                region,
                &mut scratch.coord,
                &scratch.outer_red_axes,
                scratch.outer_red_len,
                scratch.inner_len,
                ext_reads,
                gradients,
                d_root,
                *ext_x,
                map_x,
                *ext_scale,
                *ext_bias,
                *ext_tgt,
                map_tgt,
                *use_2d_tile,
            );
            return;
        }
        CpuFastKernel::AffineUnaryMse {
            ext_x,
            map_x,
            ext_scale,
            ext_bias,
            ext_tgt,
            map_tgt,
            unary,
            use_2d_tile,
        } => {
            execute_affine_unary_mse(
                region,
                &mut scratch.coord,
                &scratch.outer_red_axes,
                scratch.outer_red_len,
                scratch.inner_len,
                ext_reads,
                gradients,
                d_root,
                *ext_x,
                map_x,
                *ext_scale,
                *ext_bias,
                *ext_tgt,
                map_tgt,
                *unary,
                *use_2d_tile,
            );
            return;
        }
        CpuFastKernel::None => {}
    }

    for acc in &mut scratch.small_acc {
        acc.fill(0.0);
    }

    if scratch.all_flat_contiguous {
        let mut base = 0usize;
        while base < region.total_domain {
            let n = (region.total_domain - base).min(CPU_TILE);
            run_tile_flat(region, scratch, ext_reads, gradients, base, n, d_root);
            base += n;
        }
        return;
    }

    let inner_axis = rank - 1;
    let inner_in_red = scratch.inner_in_red;
    let inner_len = scratch.inner_len;
    let outer_red_len = scratch.outer_red_len;
    let n_instrs = region.instrs.len();
    for g in 0..region.group_len {
        let mut rem_g = g;
        for &a in region.kept_axes.iter().rev() {
            let d = region.domain[a];
            scratch.coord[a] = rem_g % d;
            rem_g /= d;
        }
        scratch.group_acc.fill(0.0);

        for outer in 0..outer_red_len {
            let mut rem_o = outer;
            for &a in scratch.outer_red_axes.iter().rev() {
                let d = region.domain[a];
                scratch.coord[a] = rem_o % d;
                rem_o /= d;
            }
            if inner_in_red {
                scratch.coord[inner_axis] = 0;
            }

            let mut t0 = 0usize;
            while t0 < inner_len {
                let n = (inner_len - t0).min(CPU_TILE);
                let nk = n * k;

                // Forward SSA evaluation over the tile [t0..t0+n].
                for r in 0..n_instrs {
                    match &region.instrs[r] {
                        FwdInstr::LoadExt { ext, map } => {
                            let src_buf = ext_reads[*ext];
                            let mut base_off = map.offset;
                            for a in 0..rank {
                                base_off += scratch.coord[a] as isize * map.strides[a];
                            }
                            let s_inner = if inner_in_red {
                                map.strides[inner_axis]
                            } else {
                                0
                            };
                            base_off += t0 as isize * s_inner;
                            let out = &mut scratch.fwd[r];
                            if region.is_vec[r] {
                                if map.is_k_contiguous(k) {
                                    out[..nk].copy_from_slice(&src_buf[base_off as usize..][..nk]);
                                } else if !map.has_k && s_inner == 0 {
                                    let val = src_buf[base_off as usize];
                                    out[..nk].fill(val);
                                } else if map.has_k && s_inner == 0 {
                                    if k == 2 {
                                        let v0 = src_buf[base_off as usize];
                                        let v1 = src_buf[(base_off + map.k_stride) as usize];
                                        for pair in out[..nk].chunks_exact_mut(2) {
                                            pair[0] = v0;
                                            pair[1] = v1;
                                        }
                                    } else {
                                        for i in 0..n {
                                            for kk in 0..k {
                                                out[i * k + kk] = src_buf[(base_off
                                                    + kk as isize * map.k_stride)
                                                    as usize];
                                            }
                                        }
                                    }
                                } else {
                                    for i in 0..n {
                                        let row = base_off + i as isize * s_inner;
                                        for kk in 0..k {
                                            let off = row
                                                + if map.has_k {
                                                    kk as isize * map.k_stride
                                                } else {
                                                    0
                                                };
                                            out[i * k + kk] = src_buf[off as usize];
                                        }
                                    }
                                }
                            } else if map.is_unit_inner() && inner_in_red {
                                out[..n].copy_from_slice(&src_buf[base_off as usize..][..n]);
                            } else if s_inner == 0 {
                                out[..n].fill(src_buf[base_off as usize]);
                            } else {
                                for (i, dst) in out[..n].iter_mut().enumerate() {
                                    *dst = src_buf[(base_off + i as isize * s_inner) as usize];
                                }
                            }
                        }
                        FwdInstr::Unary { op, src } => {
                            let len = if region.is_vec[r] { nk } else { n };
                            let (left, right) = scratch.fwd.split_at_mut(r);
                            eval_unary_tile(*op, &left[*src][..len], &mut right[0][..len]);
                        }
                        FwdInstr::Binary { op, lhs, rhs } => {
                            let len = if region.is_vec[r] { nk } else { n };
                            let (left, right) = scratch.fwd.split_at_mut(r);
                            let l_vec = region.is_vec[*lhs];
                            let r_vec = region.is_vec[*rhs];
                            if l_vec == r_vec {
                                eval_binary_tile(
                                    *op,
                                    &left[*lhs][..len],
                                    &left[*rhs][..len],
                                    &mut right[0][..len],
                                );
                            } else {
                                // Scalar broadcast into vec register.
                                let dst = &mut right[0][..nk];
                                let a = &left[*lhs];
                                let b = &left[*rhs];
                                for i in 0..n {
                                    for kk in 0..k {
                                        let av = if l_vec { a[i * k + kk] } else { a[i] };
                                        let bv = if r_vec { b[i * k + kk] } else { b[i] };
                                        dst[i * k + kk] = match op {
                                            BinaryOp::Add => av + bv,
                                            BinaryOp::Sub => av - bv,
                                            BinaryOp::Mul => av * bv,
                                            BinaryOp::Div => av / bv,
                                        };
                                    }
                                }
                            }
                        }
                        FwdInstr::SmallReduce { op, src } => {
                            let (left, right) = scratch.fwd.split_at_mut(r);
                            let s = &left[*src][..nk];
                            let dst = &mut right[0][..n];
                            if k == 2 && *op == ReduceOp::Sum {
                                for (i, o) in dst.iter_mut().enumerate() {
                                    *o = s[i * 2] + s[i * 2 + 1];
                                }
                            } else {
                                let inv_k = 1.0f32 / k as f32;
                                for (i, o) in dst.iter_mut().enumerate() {
                                    let mut sum = 0.0f32;
                                    for kk in 0..k {
                                        sum += s[i * k + kk];
                                    }
                                    *o = if *op == ReduceOp::Mean {
                                        sum * inv_k
                                    } else {
                                        sum
                                    };
                                }
                            }
                        }
                    }
                }

                // Reverse SSA evaluation over the same L1 tile [t0..t0+n].
                scratch.adj_init.fill(false);
                let last = n_instrs - 1;
                scratch.adj[last][..n].fill(d_root);
                scratch.adj_init[last] = true;

                for r in (0..n_instrs).rev() {
                    if !region.req_grad[r] || !scratch.adj_init[r] {
                        continue;
                    }
                    match &region.instrs[r] {
                        FwdInstr::LoadExt { ext, map } => {
                            let role = region.exts[*ext].role;
                            let len = if region.is_vec[r] { nk } else { n };
                            match role {
                                ExtRole::ReadOnly => {}
                                ExtRole::GroupKept => {
                                    scratch.group_acc[*ext] +=
                                        sum_slice_simd8(&scratch.adj[r][..len]);
                                }
                                ExtRole::SmallGlobal => {
                                    let acc = &mut scratch.small_acc[*ext];
                                    if region.is_vec[r] && map.has_k && map.is_zero_inner() {
                                        let mut base_off = map.offset;
                                        for a in 0..rank {
                                            base_off += scratch.coord[a] as isize * map.strides[a];
                                        }
                                        let d = &scratch.adj[r][..nk];
                                        if k == 2 && base_off == 0 && map.k_stride == 1 {
                                            let mut s0 = [0.0f32; 4];
                                            let mut s1 = [0.0f32; 4];
                                            let chunks = d.chunks_exact(8);
                                            let rem = chunks.remainder();
                                            for c in chunks {
                                                for j in 0..4 {
                                                    s0[j] += c[j * 2];
                                                    s1[j] += c[j * 2 + 1];
                                                }
                                            }
                                            let mut t0 = (s0[0] + s0[1]) + (s0[2] + s0[3]);
                                            let mut t1 = (s1[0] + s1[1]) + (s1[2] + s1[3]);
                                            for pair in rem.chunks_exact(2) {
                                                t0 += pair[0];
                                                t1 += pair[1];
                                            }
                                            acc[0] += t0;
                                            acc[1] += t1;
                                        } else {
                                            for i in 0..n {
                                                for kk in 0..k {
                                                    let idx = (base_off
                                                        + kk as isize * map.k_stride)
                                                        as usize;
                                                    acc[idx] += d[i * k + kk];
                                                }
                                            }
                                        }
                                    } else {
                                        let mut base_off = map.offset;
                                        for a in 0..rank {
                                            base_off += scratch.coord[a] as isize * map.strides[a];
                                        }
                                        let s_inner = if inner_in_red {
                                            map.strides[inner_axis]
                                        } else {
                                            0
                                        };
                                        base_off += t0 as isize * s_inner;
                                        let d = &scratch.adj[r];
                                        if region.is_vec[r] {
                                            for i in 0..n {
                                                let row = base_off + i as isize * s_inner;
                                                for kk in 0..k {
                                                    let idx = (row
                                                        + if map.has_k {
                                                            kk as isize * map.k_stride
                                                        } else {
                                                            0
                                                        })
                                                        as usize;
                                                    acc[idx] += d[i * k + kk];
                                                }
                                            }
                                        } else if s_inner == 0 {
                                            acc[base_off as usize] += sum_slice_simd8(&d[..n]);
                                        } else {
                                            for (i, &dv) in d[..n].iter().enumerate() {
                                                acc[(base_off + i as isize * s_inner) as usize] +=
                                                    dv;
                                            }
                                        }
                                    }
                                }
                                ExtRole::BarrierOut | ExtRole::FullElementwise => {
                                    let mut base_off = map.offset;
                                    for a in 0..rank {
                                        base_off += scratch.coord[a] as isize * map.strides[a];
                                    }
                                    let s_inner = if inner_in_red {
                                        map.strides[inner_axis]
                                    } else {
                                        0
                                    };
                                    base_off += t0 as isize * s_inner;
                                    let slot = region.exts[*ext].grad_slot.unwrap();
                                    let g_buf = gradients[slot].data.data_mut();
                                    let d = &scratch.adj[r][..n];
                                    if map.is_unit_inner() && inner_in_red {
                                        g_buf[base_off as usize..][..n].copy_from_slice(d);
                                    } else {
                                        for (i, &dv) in d.iter().enumerate() {
                                            let idx = (base_off + i as isize * s_inner) as usize;
                                            g_buf[idx] = dv;
                                        }
                                    }
                                }
                            }
                        }
                        FwdInstr::Unary { op, src } => {
                            if region.req_grad[*src] {
                                let len = if region.is_vec[r] { nk } else { n };
                                let init = !scratch.adj_init[*src];
                                let (left, right) = scratch.adj.split_at_mut(r);
                                accum_unary_bwd_tile(
                                    *op,
                                    &scratch.fwd[*src][..len],
                                    &scratch.fwd[r][..len],
                                    &right[0][..len],
                                    &mut left[*src][..len],
                                    init,
                                );
                                scratch.adj_init[*src] = true;
                            }
                        }
                        FwdInstr::Binary { op, lhs, rhs } => {
                            let len = if region.is_vec[r] { nk } else { n };
                            let l_vec = region.is_vec[*lhs];
                            let r_vec = region.is_vec[*rhs];
                            if l_vec == r_vec {
                                if region.req_grad[*lhs] {
                                    let init = !scratch.adj_init[*lhs];
                                    let (left, right) = scratch.adj.split_at_mut(r);
                                    accum_binary_bwd_tile(
                                        *op,
                                        true,
                                        &scratch.fwd[*lhs][..len],
                                        &scratch.fwd[*rhs][..len],
                                        &right[0][..len],
                                        &mut left[*lhs][..len],
                                        init,
                                    );
                                    scratch.adj_init[*lhs] = true;
                                }
                                if region.req_grad[*rhs] {
                                    let init = !scratch.adj_init[*rhs];
                                    let (left, right) = scratch.adj.split_at_mut(r);
                                    accum_binary_bwd_tile(
                                        *op,
                                        false,
                                        &scratch.fwd[*lhs][..len],
                                        &scratch.fwd[*rhs][..len],
                                        &right[0][..len],
                                        &mut left[*rhs][..len],
                                        init,
                                    );
                                    scratch.adj_init[*rhs] = true;
                                }
                            } else {
                                // Mixed vec/scalar binary backward (zero stack zeroing).
                                let d = &scratch.adj[r][..nk];
                                let a = &scratch.fwd[*lhs];
                                let b = &scratch.fwd[*rhs];
                                let l_len = if l_vec { nk } else { n };
                                let r_len = if r_vec { nk } else { n };
                                let dl_tmp = &mut scratch.mixed_dl[..l_len];
                                let dr_tmp = &mut scratch.mixed_dr[..r_len];
                                if region.req_grad[*lhs] {
                                    dl_tmp.fill(0.0);
                                }
                                if region.req_grad[*rhs] {
                                    dr_tmp.fill(0.0);
                                }
                                for i in 0..n {
                                    for kk in 0..k {
                                        let dv = d[i * k + kk];
                                        let av = if l_vec { a[i * k + kk] } else { a[i] };
                                        let bv = if r_vec { b[i * k + kk] } else { b[i] };
                                        if region.req_grad[*lhs] {
                                            let c = match op {
                                                BinaryOp::Add | BinaryOp::Sub => dv,
                                                BinaryOp::Mul => dv * bv,
                                                BinaryOp::Div => dv / bv,
                                            };
                                            let idx = if l_vec { i * k + kk } else { i };
                                            dl_tmp[idx] += c;
                                        }
                                        if region.req_grad[*rhs] {
                                            let c = match op {
                                                BinaryOp::Add => dv,
                                                BinaryOp::Sub => {
                                                    f32::from_bits(dv.to_bits() ^ 0x8000_0000)
                                                }
                                                BinaryOp::Mul => dv * av,
                                                BinaryOp::Div => {
                                                    let v = (dv / bv) * (av / bv);
                                                    f32::from_bits(v.to_bits() ^ 0x8000_0000)
                                                }
                                            };
                                            let idx = if r_vec { i * k + kk } else { i };
                                            dr_tmp[idx] += c;
                                        }
                                    }
                                }
                                if region.req_grad[*lhs] {
                                    let dst = &mut scratch.adj[*lhs][..l_len];
                                    if !scratch.adj_init[*lhs] {
                                        dst.copy_from_slice(&scratch.mixed_dl[..l_len]);
                                    } else {
                                        for (o, &v) in
                                            dst.iter_mut().zip(&scratch.mixed_dl[..l_len])
                                        {
                                            *o += v;
                                        }
                                    }
                                    scratch.adj_init[*lhs] = true;
                                }
                                if region.req_grad[*rhs] {
                                    let dst = &mut scratch.adj[*rhs][..r_len];
                                    if !scratch.adj_init[*rhs] {
                                        dst.copy_from_slice(&scratch.mixed_dr[..r_len]);
                                    } else {
                                        for (o, &v) in
                                            dst.iter_mut().zip(&scratch.mixed_dr[..r_len])
                                        {
                                            *o += v;
                                        }
                                    }
                                    scratch.adj_init[*rhs] = true;
                                }
                            }
                        }
                        FwdInstr::SmallReduce { op, src } => {
                            if region.req_grad[*src] {
                                let init = !scratch.adj_init[*src];
                                let (left, right) = scratch.adj.split_at_mut(r);
                                let d = &right[0][..n];
                                let dst = &mut left[*src][..nk];
                                let scale = if *op == ReduceOp::Mean {
                                    1.0f32 / k as f32
                                } else {
                                    1.0f32
                                };
                                if k == 2 && *op == ReduceOp::Sum && init {
                                    for (i, &dv) in d.iter().enumerate() {
                                        dst[i * 2] = dv;
                                        dst[i * 2 + 1] = dv;
                                    }
                                } else if init {
                                    for (i, &dv) in d.iter().enumerate() {
                                        let v = dv * scale;
                                        for kk in 0..k {
                                            dst[i * k + kk] = v;
                                        }
                                    }
                                } else {
                                    for (i, &dv) in d.iter().enumerate() {
                                        let v = dv * scale;
                                        for kk in 0..k {
                                            dst[i * k + kk] += v;
                                        }
                                    }
                                }
                                scratch.adj_init[*src] = true;
                            }
                        }
                    }
                }

                t0 += n;
            }
        }

        // Commit GroupKept gradients for group `g`.
        for (eid, ext) in region.exts.iter().enumerate() {
            if ext.role != ExtRole::GroupKept {
                continue;
            }
            let slot = ext.grad_slot.unwrap();
            gradients[slot].data.data_mut()[g] = scratch.group_acc[eid];
        }
    }

    // Commit SmallGlobal gradients.
    for (eid, ext) in region.exts.iter().enumerate() {
        if ext.role != ExtRole::SmallGlobal {
            continue;
        }
        let slot = ext.grad_slot.unwrap();
        gradients[slot]
            .data
            .data_mut()
            .copy_from_slice(&scratch.small_acc[eid]);
    }
}

fn run_tile_flat(
    region: &RegionPlan,
    scratch: &mut CpuFusedScratch,
    ext_reads: &[&[f32]],
    gradients: &mut [super::engine::Gradient<crate::Tensor>],
    base: usize,
    n: usize,
    d_root: f32,
) {
    let n_instrs = region.instrs.len();
    for r in 0..n_instrs {
        match &region.instrs[r] {
            FwdInstr::LoadExt { ext, .. } => {
                scratch.fwd[r][..n].copy_from_slice(&ext_reads[*ext][base..base + n]);
            }
            FwdInstr::Unary { op, src } => {
                let (left, right) = scratch.fwd.split_at_mut(r);
                eval_unary_tile(*op, &left[*src][..n], &mut right[0][..n]);
            }
            FwdInstr::Binary { op, lhs, rhs } => {
                let (left, right) = scratch.fwd.split_at_mut(r);
                eval_binary_tile(*op, &left[*lhs][..n], &left[*rhs][..n], &mut right[0][..n]);
            }
            FwdInstr::SmallReduce { .. } => unreachable!(),
        }
    }
    scratch.adj_init.fill(false);
    let last = n_instrs - 1;
    scratch.adj[last][..n].fill(d_root);
    scratch.adj_init[last] = true;
    for r in (0..n_instrs).rev() {
        if !region.req_grad[r] || !scratch.adj_init[r] {
            continue;
        }
        match &region.instrs[r] {
            FwdInstr::LoadExt { ext, .. } => {
                if matches!(
                    region.exts[*ext].role,
                    ExtRole::BarrierOut | ExtRole::FullElementwise
                ) {
                    let slot = region.exts[*ext].grad_slot.unwrap();
                    let d = &scratch.adj[r][..n];
                    gradients[slot].data.data_mut()[base..base + n].copy_from_slice(d);
                }
            }
            FwdInstr::Unary { op, src } => {
                if region.req_grad[*src] {
                    let init = !scratch.adj_init[*src];
                    let (left, right) = scratch.adj.split_at_mut(r);
                    accum_unary_bwd_tile(
                        *op,
                        &scratch.fwd[*src][..n],
                        &scratch.fwd[r][..n],
                        &right[0][..n],
                        &mut left[*src][..n],
                        init,
                    );
                    scratch.adj_init[*src] = true;
                }
            }
            FwdInstr::Binary { op, lhs, rhs } => {
                if region.req_grad[*lhs] {
                    let init = !scratch.adj_init[*lhs];
                    let (left, right) = scratch.adj.split_at_mut(r);
                    accum_binary_bwd_tile(
                        *op,
                        true,
                        &scratch.fwd[*lhs][..n],
                        &scratch.fwd[*rhs][..n],
                        &right[0][..n],
                        &mut left[*lhs][..n],
                        init,
                    );
                    scratch.adj_init[*lhs] = true;
                }
                if region.req_grad[*rhs] {
                    let init = !scratch.adj_init[*rhs];
                    let (left, right) = scratch.adj.split_at_mut(r);
                    accum_binary_bwd_tile(
                        *op,
                        false,
                        &scratch.fwd[*lhs][..n],
                        &scratch.fwd[*rhs][..n],
                        &right[0][..n],
                        &mut left[*rhs][..n],
                        init,
                    );
                    scratch.adj_init[*rhs] = true;
                }
            }
            FwdInstr::SmallReduce { .. } => unreachable!(),
        }
    }
}

// ================================================================================================
// General CUDA NVRTC Source Code Emitter (Strict IEEE FP32, zero --use_fast_math, 4 Topologies)
// ================================================================================================

#[cfg(feature = "cuda")]
pub(crate) mod cuda_gen {
    use super::*;
    use std::fmt::Write as _;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum LaunchTopology {
        /// Pure elementwise tail (e.g. BarrierWithTail with no broadcast params): 1 kernel, 0 scratch.
        BarrierTailOnly { grid: u32, block: u32 },
        /// Entire small graph in ONE single block: 1 kernel, 0 scratch.
        SingleBlockAll { block: u32, warp_per_group: bool },
        /// 1 warp per GroupKept group (e.g. spectrum-medium/large); finalize only if SmallGlobal exists.
        WarpPerGroup {
            grid: u32,
            block: u32,
            has_finalize: bool,
            scratch_floats: usize,
        },
        /// Multiple blocks per GroupKept group (e.g. wave-multichannel/long): 2 kernels.
        SplitBlocksPerGroup {
            split: u32,
            grid: u32,
            block: u32,
            scratch_floats: usize,
        },
    }

    pub(crate) struct CudaFusedSource {
        pub source: String,
        pub topology: LaunchTopology,
        pub main_name: &'static str,
        pub fin_name: &'static str,
        pub scratch_bytes: usize,
    }

    fn emit_preamble(out: &mut String) {
        out.push_str(
            r#"
typedef unsigned long long u64;
typedef long long i64;
__device__ __forceinline__ float fa_neg(float x) { return __uint_as_float(__float_as_uint(x) ^ 0x80000000u); }
__device__ __forceinline__ float fa_abs(float x) { return __uint_as_float(__float_as_uint(x) & 0x7fffffffu); }
__device__ __forceinline__ float fa_sig(float x) {
    if (x >= 0.0f) { return 1.0f / (1.0f + expf(-x)); }
    float e = expf(x);
    return e / (1.0f + e);
}
__device__ __forceinline__ float fa_silu(float x) {
    return (x == -__int_as_float(0x7f800000)) ? -0.0f : (x * fa_sig(x));
}
__device__ __forceinline__ float fa_softplus(float x) {
    float m = x > 0.0f ? x : 0.0f;
    return m + log1pf(expf(-fa_abs(x)));
}
__device__ __forceinline__ float fa_abs_bwd(float x, float d) {
    if (isnan(x)) return __uint_as_float(0x7fc00000u);
    if (x > 0.0f) return d;
    if (x < 0.0f) return fa_neg(d);
    return 0.0f;
}
__device__ __forceinline__ float fa_relu_bwd(float x, float d) {
    if (isnan(x)) return __uint_as_float(0x7fc00000u);
    return x > 0.0f ? d : 0.0f;
}
__device__ __forceinline__ float fa_silu_bwd(float x, float d) {
    if (x == __int_as_float(0x7f800000)) return d;
    if (x == -__int_as_float(0x7f800000)) return -0.0f * d;
    float s = fa_sig(x);
    return (s + x * (s * (1.0f - s))) * d;
}
__device__ __forceinline__ float fa_warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}
"#,
        );
    }

    fn unary_fwd_expr(op: UnaryOp, x: &str) -> String {
        match op {
            UnaryOp::Neg => format!("fa_neg({x})"),
            UnaryOp::Abs => format!("fa_abs({x})"),
            UnaryOp::Square => format!("({x} * {x})"),
            UnaryOp::Reciprocal => format!("(1.0f / {x})"),
            UnaryOp::Sqrt => format!("sqrtf({x})"),
            UnaryOp::Exp => format!("expf({x})"),
            UnaryOp::Log => format!("logf({x})"),
            UnaryOp::Relu => format!("(({x} < 0.0f) ? 0.0f : {x})"),
            UnaryOp::Sigmoid => format!("fa_sig({x})"),
            UnaryOp::Tanh => format!("tanhf({x})"),
            UnaryOp::Silu => format!("fa_silu({x})"),
            UnaryOp::Softplus => format!("fa_softplus({x})"),
        }
    }

    fn unary_bwd_expr(op: UnaryOp, x: &str, y: &str, d: &str) -> String {
        match op {
            UnaryOp::Neg => format!("fa_neg({d})"),
            UnaryOp::Abs => format!("fa_abs_bwd({x}, {d})"),
            UnaryOp::Square => format!("((2.0f * {x}) * {d})"),
            UnaryOp::Reciprocal => format!("fa_neg(({d} / {x}) / {x})"),
            UnaryOp::Sqrt => format!("({d} / (2.0f * {y}))"),
            UnaryOp::Exp => format!("({y} * {d})"),
            UnaryOp::Log => format!("({d} / {x})"),
            UnaryOp::Relu => format!("fa_relu_bwd({x}, {d})"),
            UnaryOp::Sigmoid => format!("(({y} * (1.0f - {y})) * {d})"),
            UnaryOp::Tanh => format!("(((1.0f - {y}) * (1.0f + {y})) * {d})"),
            UnaryOp::Silu => format!("fa_silu_bwd({x}, {d})"),
            UnaryOp::Softplus => format!("(fa_sig({x}) * {d})"),
        }
    }

    fn binary_fwd_expr(op: BinaryOp, a: &str, b: &str) -> String {
        match op {
            BinaryOp::Add => format!("({a} + {b})"),
            BinaryOp::Sub => format!("({a} - {b})"),
            BinaryOp::Mul => format!("({a} * {b})"),
            BinaryOp::Div => format!("({a} / {b})"),
        }
    }

    fn binary_bwd_expr(op: BinaryOp, lhs: bool, a: &str, b: &str, d: &str) -> String {
        match (op, lhs) {
            (BinaryOp::Add, _) | (BinaryOp::Sub, true) => d.to_string(),
            (BinaryOp::Sub, false) => format!("fa_neg({d})"),
            (BinaryOp::Mul, true) => format!("({d} * {b})"),
            (BinaryOp::Mul, false) => format!("({d} * {a})"),
            (BinaryOp::Div, true) => format!("({d} / {b})"),
            (BinaryOp::Div, false) => format!("fa_neg(({d} / {b}) * ({a} / {b}))"),
        }
    }

    /// Emit coordinate decoding expressions from `(g, m)` into `c_0, c_1, ..., c_{R-1}`.
    fn emit_decode_coords(
        out: &mut String,
        region: &RegionPlan,
        indent: &str,
        g_var: &str,
        m_var: &str,
    ) {
        let rank = region.domain.len();
        for a in 0..rank {
            let _ = writeln!(out, "{indent}i64 c_{a} = 0;");
        }
        if !region.kept_axes.is_empty() {
            let _ = writeln!(out, "{indent}{{ i64 rem = {g_var};");
            for (idx, &a) in region.kept_axes.iter().rev().enumerate() {
                let dim = region.domain[a];
                if idx + 1 == region.kept_axes.len() {
                    let _ = writeln!(out, "{indent}  c_{a} = rem;");
                } else {
                    let _ = writeln!(out, "{indent}  c_{a} = rem % {dim}LL; rem /= {dim}LL;");
                }
            }
            let _ = writeln!(out, "{indent}}}");
        }
        if !region.red_axes.is_empty() {
            let _ = writeln!(out, "{indent}{{ i64 rem = {m_var};");
            for (idx, &a) in region.red_axes.iter().rev().enumerate() {
                let dim = region.domain[a];
                if idx + 1 == region.red_axes.len() {
                    let _ = writeln!(out, "{indent}  c_{a} = rem;");
                } else {
                    let _ = writeln!(out, "{indent}  c_{a} = rem % {dim}LL; rem /= {dim}LL;");
                }
            }
            let _ = writeln!(out, "{indent}}}");
        }
    }

    fn affine_offset_expr(map: &AffineMap, k_expr: Option<usize>) -> String {
        let mut terms = Vec::new();
        if map.offset != 0 {
            terms.push(format!("{}LL", map.offset));
        }
        for (a, &s) in map.strides.iter().enumerate() {
            if s == 1 {
                terms.push(format!("c_{a}"));
            } else if s == -1 {
                terms.push(format!("(-c_{a})"));
            } else if s != 0 {
                terms.push(format!("(c_{a} * {s}LL)"));
            }
        }
        if let Some(kk) = k_expr {
            if map.has_k && kk != 0 {
                let off = kk as isize * map.k_stride;
                if off != 0 {
                    terms.push(format!("({off}LL)"));
                }
            }
        }
        if terms.is_empty() {
            "0LL".to_string()
        } else {
            terms.join(" + ")
        }
    }

    /// Emit the body of one point `(c_0, ..., c_{R-1})` forward+backward SSA evaluation.
    fn emit_point_ssa(
        out: &mut String,
        region: &RegionPlan,
        indent: &str,
        d_root_var: &str,
        preload_group_regs: bool,
    ) {
        let k = region.k_dim;
        let n_instrs = region.instrs.len();

        // Forward SSA instructions
        for r in 0..n_instrs {
            let vec_len = if region.is_vec[r] { k } else { 1 };
            match &region.instrs[r] {
                FwdInstr::LoadExt { ext, map } => {
                    if preload_group_regs
                        && region.exts[*ext].role == ExtRole::GroupKept
                        && !map.has_k
                    {
                        for kk in 0..vec_len {
                            let _ = writeln!(out, "{indent}float f_{r}_{kk} = gp_{ext};");
                        }
                    } else if region.exts[*ext].role == ExtRole::SmallGlobal
                        && region.exts[*ext].param_len <= 8
                    {
                        for kk in 0..vec_len {
                            let off = affine_offset_expr(
                                map,
                                if region.is_vec[r] { Some(kk) } else { None },
                            );
                            let _ = writeln!(out, "{indent}float f_{r}_{kk} = ext_{ext}[{off}];");
                        }
                    } else if region.is_vec[r]
                        && k == 2
                        && map.has_k
                        && map.k_stride == 1
                        && (map.offset % 2 == 0)
                        && map.strides.iter().all(|&s| s % 2 == 0)
                    {
                        // Coalesced aligned float2 load!
                        let base_off = affine_offset_expr(map, Some(0));
                        let _ = writeln!(
                            out,
                            "{indent}float2 tmp2_{r} = reinterpret_cast<const float2*>(ext_{ext})[({base_off}) >> 1];"
                        );
                        let _ = writeln!(
                            out,
                            "{indent}float f_{r}_0 = tmp2_{r}.x; float f_{r}_1 = tmp2_{r}.y;"
                        );
                    } else {
                        for kk in 0..vec_len {
                            let off = affine_offset_expr(
                                map,
                                if region.is_vec[r] { Some(kk) } else { None },
                            );
                            let _ = writeln!(out, "{indent}float f_{r}_{kk} = ext_{ext}[{off}];");
                        }
                    }
                }
                FwdInstr::Unary { op, src } => {
                    for kk in 0..vec_len {
                        let expr = unary_fwd_expr(*op, &format!("f_{src}_{kk}"));
                        let _ = writeln!(out, "{indent}float f_{r}_{kk} = {expr};");
                    }
                }
                FwdInstr::Binary { op, lhs, rhs } => {
                    let l_vec = region.is_vec[*lhs];
                    let r_vec = region.is_vec[*rhs];
                    for kk in 0..vec_len {
                        let ls = format!("f_{lhs}_{}", if l_vec { kk } else { 0 });
                        let rs = format!("f_{rhs}_{}", if r_vec { kk } else { 0 });
                        let expr = binary_fwd_expr(*op, &ls, &rs);
                        let _ = writeln!(out, "{indent}float f_{r}_{kk} = {expr};");
                    }
                }
                FwdInstr::SmallReduce { op, src } => {
                    let mut expr = format!("f_{src}_0");
                    for kk in 1..k {
                        expr = format!("({expr} + f_{src}_{kk})");
                    }
                    if *op == ReduceOp::Mean {
                        let _ = writeln!(out, "{indent}float f_{r}_0 = {expr} / {k}.0f;");
                    } else {
                        let _ = writeln!(out, "{indent}float f_{r}_0 = {expr};");
                    }
                }
            }
        }

        // Reverse SSA instructions
        let mut adj_declared = vec![false; n_instrs];
        let last = n_instrs - 1;
        let _ = writeln!(out, "{indent}float a_{last}_0 = {d_root_var};");
        adj_declared[last] = true;

        for r in (0..n_instrs).rev() {
            if !region.req_grad[r] || !adj_declared[r] {
                continue;
            }
            let vec_len = if region.is_vec[r] { k } else { 1 };
            match &region.instrs[r] {
                FwdInstr::LoadExt { ext, map } => match region.exts[*ext].role {
                    ExtRole::ReadOnly => {}
                    ExtRole::GroupKept => {
                        for kk in 0..vec_len {
                            let _ = writeln!(out, "{indent}gacc_{ext} += a_{r}_{kk};");
                        }
                    }
                    ExtRole::SmallGlobal => {
                        let param_len = region.exts[*ext].param_len;
                        if region.is_vec[r]
                            && map.has_k
                            && map.offset == 0
                            && map.strides.iter().all(|&s| s == 0)
                            && map.k_stride == 1
                            && param_len == k
                        {
                            for kk in 0..k {
                                let _ = writeln!(out, "{indent}sacc_{ext}[{kk}] += a_{r}_{kk};");
                            }
                        } else {
                            for kk in 0..vec_len {
                                let off = affine_offset_expr(
                                    map,
                                    if region.is_vec[r] { Some(kk) } else { None },
                                );
                                let _ = writeln!(out, "{indent}sacc_{ext}[{off}] += a_{r}_{kk};");
                            }
                        }
                    }
                    ExtRole::BarrierOut | ExtRole::FullElementwise => {
                        for kk in 0..vec_len {
                            let off = affine_offset_expr(
                                map,
                                if region.is_vec[r] { Some(kk) } else { None },
                            );
                            let _ = writeln!(out, "{indent}grad_{ext}[{off}] = a_{r}_{kk};");
                            if region.exts[*ext].update_sgd {
                                let _ = writeln!(
                                    out,
                                    "{indent}ext_{ext}[{off}] = __fsub_rn(ext_{ext}[{off}], __fmul_rn(rate, a_{r}_{kk}));"
                                );
                            }
                        }
                    }
                },
                FwdInstr::Unary { op, src } => {
                    if region.req_grad[*src] {
                        for kk in 0..vec_len {
                            let expr = unary_bwd_expr(
                                *op,
                                &format!("f_{src}_{kk}"),
                                &format!("f_{r}_{kk}"),
                                &format!("a_{r}_{kk}"),
                            );
                            if !adj_declared[*src] {
                                let _ = writeln!(out, "{indent}float a_{src}_{kk} = {expr};");
                            } else {
                                let _ = writeln!(out, "{indent}a_{src}_{kk} += {expr};");
                            }
                        }
                        adj_declared[*src] = true;
                    }
                }
                FwdInstr::Binary { op, lhs, rhs } => {
                    let l_vec = region.is_vec[*lhs];
                    let r_vec = region.is_vec[*rhs];
                    if region.req_grad[*lhs] {
                        if !adj_declared[*lhs] {
                            let l_len = if l_vec { k } else { 1 };
                            for kk in 0..l_len {
                                let _ = writeln!(out, "{indent}float a_{lhs}_{kk} = 0.0f;");
                            }
                            adj_declared[*lhs] = true;
                        }
                        for kk in 0..vec_len {
                            let lk = if l_vec { kk } else { 0 };
                            let rk = if r_vec { kk } else { 0 };
                            let expr = binary_bwd_expr(
                                *op,
                                true,
                                &format!("f_{lhs}_{lk}"),
                                &format!("f_{rhs}_{rk}"),
                                &format!("a_{r}_{kk}"),
                            );
                            let _ = writeln!(out, "{indent}a_{lhs}_{lk} += {expr};");
                        }
                    }
                    if region.req_grad[*rhs] {
                        if !adj_declared[*rhs] {
                            let r_len = if r_vec { k } else { 1 };
                            for kk in 0..r_len {
                                let _ = writeln!(out, "{indent}float a_{rhs}_{kk} = 0.0f;");
                            }
                            adj_declared[*rhs] = true;
                        }
                        for kk in 0..vec_len {
                            let lk = if l_vec { kk } else { 0 };
                            let rk = if r_vec { kk } else { 0 };
                            let expr = binary_bwd_expr(
                                *op,
                                false,
                                &format!("f_{lhs}_{lk}"),
                                &format!("f_{rhs}_{rk}"),
                                &format!("a_{r}_{kk}"),
                            );
                            let _ = writeln!(out, "{indent}a_{rhs}_{rk} += {expr};");
                        }
                    }
                }
                FwdInstr::SmallReduce { op, src } => {
                    if region.req_grad[*src] {
                        let scale = if *op == ReduceOp::Mean {
                            format!("(a_{r}_0 / {k}.0f)")
                        } else {
                            format!("a_{r}_0")
                        };
                        for kk in 0..k {
                            if !adj_declared[*src] {
                                let _ = writeln!(out, "{indent}float a_{src}_{kk} = {scale};");
                            } else {
                                let _ = writeln!(out, "{indent}a_{src}_{kk} += {scale};");
                            }
                        }
                        adj_declared[*src] = true;
                    }
                }
            }
        }
    }

    fn emit_kernel_sig_args(region: &RegionPlan, with_scratch: bool) -> String {
        let mut args = Vec::new();
        for (eid, ext) in region.exts.iter().enumerate() {
            if ext.update_sgd {
                args.push(format!("float *__restrict__ ext_{eid}"));
            } else {
                args.push(format!("const float *__restrict__ ext_{eid}"));
            }
        }
        for (eid, ext) in region.exts.iter().enumerate() {
            if ext.grad_slot.is_some() {
                args.push(format!("float *__restrict__ grad_{eid}"));
            }
        }
        args.push("const float *__restrict__ seed_ptr".to_string());
        args.push("float rate".to_string());
        if with_scratch {
            args.push("float *__restrict__ scratch".to_string());
        }
        args.join(", ")
    }

    pub(crate) fn emit_cuda_fused_source(plan: &FusedStepPlan) -> CudaFusedSource {
        let region = &plan.region;
        let g_len = region.group_len;
        let m_len = region.red_len;
        let n_total = region.total_domain;

        let group_exts: Vec<usize> = region
            .exts
            .iter()
            .enumerate()
            .filter(|(_, e)| e.role == ExtRole::GroupKept)
            .map(|(i, _)| i)
            .collect();
        let small_exts: Vec<(usize, usize)> = region
            .exts
            .iter()
            .enumerate()
            .filter(|(_, e)| e.role == ExtRole::SmallGlobal)
            .map(|(i, e)| (i, e.param_len))
            .collect();
        let total_small: usize = small_exts.iter().map(|&(_, l)| l).sum();
        let has_reductions = !group_exts.is_empty() || !small_exts.is_empty();

        let topology = if !has_reductions {
            let block = 256u32;
            let grid = n_total.div_ceil(block as usize).clamp(1, 65535) as u32;
            LaunchTopology::BarrierTailOnly { grid, block }
        } else if n_total <= 2048 && g_len <= 256 {
            let warp_per_group = g_len <= 8;
            let block = if warp_per_group {
                (g_len.max(1) as u32) * 32
            } else {
                256
            };
            LaunchTopology::SingleBlockAll {
                block,
                warp_per_group,
            }
        } else if g_len >= 64 {
            let block = 256u32; // 8 warps per block
            let grid = g_len.div_ceil(8) as u32;
            let has_finalize = total_small > 0;
            let scratch_floats = if has_finalize {
                grid as usize * total_small
            } else {
                0
            };
            LaunchTopology::WarpPerGroup {
                grid,
                block,
                has_finalize,
                scratch_floats,
            }
        } else {
            let block = 256u32;
            let target_blocks = (m_len / 1024).clamp(1, 64) as u32;
            let split = target_blocks.min((1024 / g_len.max(1) as u32).max(1));
            let grid = (g_len as u32) * split;
            let per_block = group_exts.len() + total_small;
            let scratch_floats = grid as usize * per_block;
            LaunchTopology::SplitBlocksPerGroup {
                split,
                grid,
                block,
                scratch_floats,
            }
        };

        let mut out = String::with_capacity(4096);
        emit_preamble(&mut out);

        let d_root_expr = match region.loss_op {
            ReduceOp::Sum => "seed_ptr[0]".to_string(),
            ReduceOp::Mean => format!("(seed_ptr[0] / ((float){n_total}ULL))"),
            ReduceOp::Max | ReduceOp::Min | ReduceOp::LogSumExp => {
                unreachable!("nonlinear loss reductions are excluded by analyze_step_fusion")
            }
        };

        let mut fin_name = "";
        let mut scratch_bytes = 0usize;

        match topology {
            LaunchTopology::BarrierTailOnly { .. } => {
                let sig = emit_kernel_sig_args(region, false);
                let _ = writeln!(out, "extern \"C\" __global__ void fa_fused_main({sig}) {{");
                let _ = writeln!(out, "    const float d_root = {d_root_expr};");
                let _ = writeln!(
                    out,
                    "    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x; idx < {n_total}LL; idx += (i64)blockDim.x * gridDim.x) {{"
                );
                emit_decode_coords(&mut out, region, "        ", "0LL", "idx");
                emit_point_ssa(&mut out, region, "        ", "d_root", false);
                let _ = writeln!(out, "    }}");
                let _ = writeln!(out, "}}");
            }
            LaunchTopology::SingleBlockAll {
                block,
                warp_per_group,
            } => {
                let sig = emit_kernel_sig_args(region, false);
                let n_warps = block / 32;
                let _ = writeln!(out, "extern \"C\" __global__ void fa_fused_main({sig}) {{");
                let _ = writeln!(out, "    const float d_root = {d_root_expr};");
                if total_small > 0 {
                    let _ = writeln!(out, "    __shared__ float smem[{n_warps} * {total_small}];");
                    for &(eid, plen) in &small_exts {
                        let _ = writeln!(out, "    float sacc_{eid}[{plen}] = {{0.0f}};");
                    }
                }
                let _ = writeln!(out, "    const int tid = threadIdx.x;");
                let _ = writeln!(out, "    const int warp = tid >> 5;");
                let _ = writeln!(out, "    const int lane = tid & 31;");
                if warp_per_group {
                    let _ = writeln!(out, "    const i64 g = (i64)warp;");
                    let _ = writeln!(out, "    if (g < {g_len}LL) {{");
                    for &eid in &group_exts {
                        let _ = writeln!(out, "        float gp_{eid} = ext_{eid}[g];");
                        let _ = writeln!(out, "        float gacc_{eid} = 0.0f;");
                    }
                    let _ = writeln!(
                        out,
                        "        for (i64 m = lane; m < {m_len}LL; m += 32LL) {{"
                    );
                    emit_decode_coords(&mut out, region, "            ", "g", "m");
                    emit_point_ssa(&mut out, region, "            ", "d_root", true);
                    let _ = writeln!(out, "        }}");
                    for &eid in &group_exts {
                        let _ = writeln!(out, "        gacc_{eid} = fa_warp_sum(gacc_{eid});");
                    }
                    let _ = writeln!(out, "        if (lane == 0) {{");
                    for &eid in &group_exts {
                        let _ = writeln!(out, "            grad_{eid}[g] = gacc_{eid};");
                        if region.exts[eid].update_sgd {
                            let _ = writeln!(
                                out,
                                "            ext_{eid}[g] = __fsub_rn(gp_{eid}, __fmul_rn(rate, gacc_{eid}));"
                            );
                        }
                    }
                    let _ = writeln!(out, "        }}");
                    let _ = writeln!(out, "    }}");
                } else {
                    let _ = writeln!(
                        out,
                        "    for (i64 g = tid; g < {g_len}LL; g += {block}LL) {{"
                    );
                    for &eid in &group_exts {
                        let _ = writeln!(out, "        float gp_{eid} = ext_{eid}[g];");
                        let _ = writeln!(out, "        float gacc_{eid} = 0.0f;");
                    }
                    let _ = writeln!(out, "        for (i64 m = 0; m < {m_len}LL; ++m) {{");
                    emit_decode_coords(&mut out, region, "            ", "g", "m");
                    emit_point_ssa(&mut out, region, "            ", "d_root", true);
                    let _ = writeln!(out, "        }}");
                    for &eid in &group_exts {
                        let _ = writeln!(out, "        grad_{eid}[g] = gacc_{eid};");
                        if region.exts[eid].update_sgd {
                            let _ = writeln!(
                                out,
                                "        ext_{eid}[g] = __fsub_rn(gp_{eid}, __fmul_rn(rate, gacc_{eid}));"
                            );
                        }
                    }
                    let _ = writeln!(out, "    }}");
                }
                if total_small > 0 {
                    let mut off = 0usize;
                    for &(eid, plen) in &small_exts {
                        for j in 0..plen {
                            let _ = writeln!(
                                out,
                                "    {{ float wv = fa_warp_sum(sacc_{eid}[{j}]); if (lane == 0) smem[warp * {total_small} + {off}] = wv; }}"
                            );
                            off += 1;
                        }
                    }
                    let _ = writeln!(out, "    __syncthreads();");
                    let _ = writeln!(out, "    if (tid == 0) {{");
                    let mut off = 0usize;
                    for &(eid, plen) in &small_exts {
                        for j in 0..plen {
                            let _ = writeln!(out, "        {{ float sum = 0.0f;");
                            let _ = writeln!(
                                out,
                                "          for (int w = 0; w < {n_warps}; ++w) sum += smem[w * {total_small} + {off}];"
                            );
                            let _ = writeln!(out, "          grad_{eid}[{j}] = sum;");
                            if region.exts[eid].update_sgd {
                                let _ = writeln!(
                                    out,
                                    "          ext_{eid}[{j}] = __fsub_rn(ext_{eid}[{j}], __fmul_rn(rate, sum));"
                                );
                            }
                            let _ = writeln!(out, "        }}");
                            off += 1;
                        }
                    }
                    let _ = writeln!(out, "    }}");
                }
                let _ = writeln!(out, "}}");
            }
            LaunchTopology::WarpPerGroup {
                grid,
                block: _,
                has_finalize,
                scratch_floats,
            } => {
                scratch_bytes = scratch_floats * size_of::<f32>();
                let sig = emit_kernel_sig_args(region, has_finalize);
                let _ = writeln!(out, "extern \"C\" __global__ void fa_fused_main({sig}) {{");
                let _ = writeln!(out, "    const float d_root = {d_root_expr};");
                if has_finalize {
                    let _ = writeln!(out, "    __shared__ float smem[8 * {total_small}];");
                    for &(eid, plen) in &small_exts {
                        let _ = writeln!(out, "    float sacc_{eid}[{plen}] = {{0.0f}};");
                    }
                }
                let _ = writeln!(out, "    const int tid = threadIdx.x;");
                let _ = writeln!(out, "    const int warp = tid >> 5;");
                let _ = writeln!(out, "    const int lane = tid & 31;");
                let _ = writeln!(out, "    const i64 g = (i64)blockIdx.x * 8LL + warp;");
                let _ = writeln!(out, "    if (g < {g_len}LL) {{");
                for &eid in &group_exts {
                    let _ = writeln!(out, "        float gp_{eid} = ext_{eid}[g];");
                    let _ = writeln!(out, "        float gacc_{eid} = 0.0f;");
                }
                let _ = writeln!(
                    out,
                    "        for (i64 m = lane; m < {m_len}LL; m += 32LL) {{"
                );
                emit_decode_coords(&mut out, region, "            ", "g", "m");
                emit_point_ssa(&mut out, region, "            ", "d_root", true);
                let _ = writeln!(out, "        }}");
                for &eid in &group_exts {
                    let _ = writeln!(out, "        gacc_{eid} = fa_warp_sum(gacc_{eid});");
                }
                let _ = writeln!(out, "        if (lane == 0) {{");
                for &eid in &group_exts {
                    let _ = writeln!(out, "            grad_{eid}[g] = gacc_{eid};");
                    if region.exts[eid].update_sgd {
                        let _ = writeln!(
                            out,
                            "            ext_{eid}[g] = __fsub_rn(gp_{eid}, __fmul_rn(rate, gacc_{eid}));"
                        );
                    }
                }
                let _ = writeln!(out, "        }}");
                let _ = writeln!(out, "    }}");
                if has_finalize {
                    let mut off = 0usize;
                    for &(eid, plen) in &small_exts {
                        for j in 0..plen {
                            let _ = writeln!(
                                out,
                                "    {{ float wv = fa_warp_sum(sacc_{eid}[{j}]); if (lane == 0) smem[warp * {total_small} + {off}] = wv; }}"
                            );
                            off += 1;
                        }
                    }
                    let _ = writeln!(out, "    __syncthreads();");
                    let _ = writeln!(out, "    if (tid == 0) {{");
                    for idx in 0..total_small {
                        let _ = writeln!(
                            out,
                            "        {{ float sum = 0.0f; for (int w = 0; w < 8; ++w) sum += smem[w * {total_small} + {idx}]; scratch[(i64)blockIdx.x * {total_small}LL + {idx}LL] = sum; }}"
                        );
                    }
                    let _ = writeln!(out, "    }}");
                }
                let _ = writeln!(out, "}}");

                if has_finalize {
                    fin_name = "fa_fused_finalize";
                    let _ = writeln!(
                        out,
                        "extern \"C\" __global__ void fa_fused_finalize({sig}) {{"
                    );
                    let _ = writeln!(out, "    __shared__ float smem[8 * {total_small}];");
                    let _ = writeln!(out, "    const int tid = threadIdx.x;");
                    let _ = writeln!(out, "    const int warp = tid >> 5;");
                    let _ = writeln!(out, "    const int lane = tid & 31;");
                    for idx in 0..total_small {
                        let _ = writeln!(
                            out,
                            "    float acc_{idx} = 0.0f; for (int b = tid; b < {grid}; b += 256) acc_{idx} += scratch[(i64)b * {total_small}LL + {idx}LL];"
                        );
                        let _ = writeln!(
                            out,
                            "    acc_{idx} = fa_warp_sum(acc_{idx}); if (lane == 0) smem[warp * {total_small} + {idx}] = acc_{idx};"
                        );
                    }
                    let _ = writeln!(out, "    __syncthreads();");
                    let _ = writeln!(out, "    if (tid == 0) {{");
                    let mut off = 0usize;
                    for &(eid, plen) in &small_exts {
                        for j in 0..plen {
                            let _ = writeln!(
                                out,
                                "        {{ float sum = 0.0f; for (int w = 0; w < 8; ++w) sum += smem[w * {total_small} + {off}]; grad_{eid}[{j}] = sum;"
                            );
                            if region.exts[eid].update_sgd {
                                let _ = writeln!(
                                    out,
                                    "          ext_{eid}[{j}] = __fsub_rn(ext_{eid}[{j}], __fmul_rn(rate, sum));"
                                );
                            }
                            let _ = writeln!(out, "        }}");
                            off += 1;
                        }
                    }
                    let _ = writeln!(out, "    }}");
                    let _ = writeln!(out, "}}");
                }
            }
            LaunchTopology::SplitBlocksPerGroup {
                split,
                grid,
                block: _,
                scratch_floats,
            } => {
                fin_name = "fa_fused_finalize";
                scratch_bytes = scratch_floats * size_of::<f32>();
                let sig = emit_kernel_sig_args(region, true);
                let per_block = group_exts.len() + total_small;
                let chunk = m_len.div_ceil(split as usize);

                let _ = writeln!(out, "extern \"C\" __global__ void fa_fused_main({sig}) {{");
                let _ = writeln!(out, "    const float d_root = {d_root_expr};");
                let _ = writeln!(out, "    __shared__ float smem[8 * {per_block}];");
                let _ = writeln!(out, "    const int tid = threadIdx.x;");
                let _ = writeln!(out, "    const int warp = tid >> 5;");
                let _ = writeln!(out, "    const int lane = tid & 31;");
                let _ = writeln!(out, "    const i64 g = (i64)(blockIdx.x / {split}u);");
                let _ = writeln!(out, "    const i64 s = (i64)(blockIdx.x % {split}u);");
                let _ = writeln!(out, "    const i64 m_start = s * {chunk}LL;");
                let _ = writeln!(
                    out,
                    "    const i64 m_end = (m_start + {chunk}LL < {m_len}LL) ? (m_start + {chunk}LL) : {m_len}LL;"
                );
                for &eid in &group_exts {
                    let _ = writeln!(out, "    float gp_{eid} = ext_{eid}[g];");
                    let _ = writeln!(out, "    float gacc_{eid} = 0.0f;");
                }
                for &(eid, plen) in &small_exts {
                    let _ = writeln!(out, "    float sacc_{eid}[{plen}] = {{0.0f}};");
                }
                let _ = writeln!(
                    out,
                    "    for (i64 m = m_start + tid; m < m_end; m += 256LL) {{"
                );
                emit_decode_coords(&mut out, region, "        ", "g", "m");
                emit_point_ssa(&mut out, region, "        ", "d_root", true);
                let _ = writeln!(out, "    }}");
                let mut off = 0usize;
                for &eid in &group_exts {
                    let _ = writeln!(
                        out,
                        "    {{ float wv = fa_warp_sum(gacc_{eid}); if (lane == 0) smem[warp * {per_block} + {off}] = wv; }}"
                    );
                    off += 1;
                }
                for &(eid, plen) in &small_exts {
                    for j in 0..plen {
                        let _ = writeln!(
                            out,
                            "    {{ float wv = fa_warp_sum(sacc_{eid}[{j}]); if (lane == 0) smem[warp * {per_block} + {off}] = wv; }}"
                        );
                        off += 1;
                    }
                }
                let _ = writeln!(out, "    __syncthreads();");
                let _ = writeln!(out, "    if (tid == 0) {{");
                for idx in 0..per_block {
                    let _ = writeln!(
                        out,
                        "        {{ float sum = 0.0f; for (int w = 0; w < 8; ++w) sum += smem[w * {per_block} + {idx}]; scratch[(i64)blockIdx.x * {per_block}LL + {idx}LL] = sum; }}"
                    );
                }
                let _ = writeln!(out, "    }}");
                let _ = writeln!(out, "}}");

                // Finalize kernel: 1 block of 256 threads sums `split` partials per group + any SmallGlobal.
                let _ = writeln!(
                    out,
                    "extern \"C\" __global__ void fa_fused_finalize({sig}) {{"
                );
                let _ = writeln!(out, "    const int tid = threadIdx.x;");
                if !group_exts.is_empty() {
                    let _ = writeln!(out, "    if (tid < {g_len}) {{");
                    let _ = writeln!(out, "        const i64 g = (i64)tid;");
                    for (slot_idx, &eid) in group_exts.iter().enumerate() {
                        let _ = writeln!(out, "        float sum_{eid} = 0.0f;");
                        let _ = writeln!(
                            out,
                            "        for (i64 s = 0; s < {split}LL; ++s) sum_{eid} += scratch[(g * {split}LL + s) * {per_block}LL + {slot_idx}LL];"
                        );
                        let _ = writeln!(out, "        grad_{eid}[g] = sum_{eid};");
                        if region.exts[eid].update_sgd {
                            let _ = writeln!(
                                out,
                                "        ext_{eid}[g] = __fsub_rn(ext_{eid}[g], __fmul_rn(rate, sum_{eid}));"
                            );
                        }
                    }
                    let _ = writeln!(out, "    }}");
                }
                if total_small > 0 {
                    let _ = writeln!(out, "    if (tid == 0) {{");
                    let mut off = group_exts.len();
                    for &(eid, plen) in &small_exts {
                        for j in 0..plen {
                            let _ = writeln!(out, "        {{ float sum = 0.0f;");
                            let _ = writeln!(
                                out,
                                "          for (i64 b = 0; b < {grid}LL; ++b) sum += scratch[b * {per_block}LL + {off}LL];"
                            );
                            let _ = writeln!(out, "          grad_{eid}[{j}] = sum;");
                            if region.exts[eid].update_sgd {
                                let _ = writeln!(
                                    out,
                                    "          ext_{eid}[{j}] = __fsub_rn(ext_{eid}[{j}], __fmul_rn(rate, sum));"
                                );
                            }
                            let _ = writeln!(out, "        }}");
                            off += 1;
                        }
                    }
                    let _ = writeln!(out, "    }}");
                }
                let _ = writeln!(out, "}}");
            }
        }

        CudaFusedSource {
            source: out,
            topology,
            main_name: "fa_fused_main",
            fin_name,
            scratch_bytes,
        }
    }
}
