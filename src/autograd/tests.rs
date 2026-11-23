use super::{
    Binding, Builder, GradientStatus, Operation, PrepareOptions, Seed,
    backend::{Backend, PlanMemory, Read, Write},
    engine::Core,
    host::Host,
};
use crate::{
    BinaryOp, Error, GradientMode as G, Layout, ReduceOp, Result, Shape, Shared, Tensor, UnaryOp,
    cpu::Cpu,
};
use std::cell::Cell;

/// Delegate all arithmetic to the real CPU plans. Inject an actual external owner
/// write at a deterministic boundary after native gradient/parameter work, without
/// timing a thread or replacing any numerical operation with a mocked result.
struct AliasWriter {
    host: Host,
    owner: Shared<Tensor>,
    trigger: Cell<Option<(bool, usize)>>,
    panic_after_write: Cell<bool>,
    forward_write_after: Cell<Option<usize>>,
}
impl AliasWriter {
    fn inject(&self, projection: bool) -> Result<()> {
        if let Some((kind, left)) = self.trigger.get() {
            if kind == projection {
                if left == 1 {
                    self.trigger.set(None);
                    self.owner.write()?.data_mut()[0] += 8.;
                    assert!(
                        !self.panic_after_write.replace(false),
                        "injected execution unwind"
                    );
                } else {
                    self.trigger.set(Some((kind, left - 1)));
                }
            }
        }
        Ok(())
    }
}
impl Backend for AliasWriter {
    type Tensor = Tensor;
    type Forward = <Host as Backend>::Forward;
    type Backward = <Host as Backend>::Backward;
    type Projection = <Host as Backend>::Projection;
    fn check(&self, t: &Tensor) -> Result<()> {
        self.host.check(t)
    }
    fn zeros(&self, s: &Shape) -> Result<Tensor> {
        self.host.zeros(s)
    }
    fn scalar(&self, v: f32) -> Result<Tensor> {
        self.host.scalar(v)
    }
    fn clear(&self, t: &mut Tensor) -> Result<()> {
        self.host.clear(t)
    }
    fn update(&self, p: &mut Tensor, g: &Tensor, r: f32) -> Result<()> {
        self.host.update(p, g, r)?;
        self.inject(false)
    }
    fn synchronize(&self) -> Result<()> {
        self.host.synchronize()
    }
    fn forward_plan(
        &self,
        o: &Operation,
        i: [Option<&Layout>; 3],
        d: &Layout,
    ) -> Result<Self::Forward> {
        self.host.forward_plan(o, i, d)
    }
    fn backward_plan(
        &self,
        o: &Operation,
        s: usize,
        i: [Option<&Layout>; 3],
        u: &Layout,
        d: &Layout,
    ) -> Result<Self::Backward> {
        self.host.backward_plan(o, s, i, u, d)
    }
    fn dependencies(p: &Self::Backward) -> [bool; 3] {
        Host::dependencies(p)
    }
    fn forward_memory(p: &Self::Forward) -> PlanMemory {
        Host::forward_memory(p)
    }
    fn backward_memory(p: &Self::Backward) -> PlanMemory {
        Host::backward_memory(p)
    }
    fn project_memory(p: &Self::Projection) -> PlanMemory {
        Host::project_memory(p)
    }
    fn project_plan(
        &self,
        b: &Shape,
        v: &Layout,
        u: &Layout,
        l: usize,
    ) -> Result<Self::Projection> {
        self.host.project_plan(b, v, u, l)
    }
    fn forward(
        &self,
        p: &mut Self::Forward,
        i: [Option<Read<'_, Tensor>>; 3],
        o: Write<'_, Tensor>,
    ) -> Result<()> {
        self.host.forward(p, i, o)?;
        if let Some(left) = self.forward_write_after.get() {
            if left == 1 {
                self.forward_write_after.set(None);
                self.owner.write()?.data_mut()[0] += 8.;
            } else {
                self.forward_write_after.set(Some(left - 1));
            }
        }
        Ok(())
    }
    fn backward(
        &self,
        p: &mut Self::Backward,
        i: [Option<Read<'_, Tensor>>; 3],
        u: Read<'_, Tensor>,
        o: Write<'_, Tensor>,
        m: G,
    ) -> Result<()> {
        self.host.backward(p, i, u, o, m)
    }
    fn project(
        &self,
        p: &Self::Projection,
        u: Read<'_, Tensor>,
        o: &mut Tensor,
        m: G,
    ) -> Result<()> {
        self.host.project(p, u, o, m)?;
        self.inject(true)
    }
}

#[test]
fn partial_backward_rejects_queries_updates_and_further_accumulation() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let q = b.unary(UnaryOp::Square, x).unwrap();
    let loss = b.reduce(ReduceOp::Sum, q, &[0], false).unwrap();
    let program = b.finish(&[x, loss]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let backend = AliasWriter {
        host: Host(Cpu::Reference),
        owner: owner.clone(),
        trigger: Cell::new(None),
        panic_after_write: Cell::new(false),
        forward_write_after: Cell::new(None),
    };
    let mut core = Core::prepare(
        backend,
        &program,
        &[Binding::new(x, owner.clone())],
        PrepareOptions::default(),
    )
    .unwrap();
    let vector = Layout::contiguous([2]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let dx = Tensor::from_vec([2], vec![2., 2.]).unwrap();
    let dl = Tensor::from_vec([], vec![1.]).unwrap();
    let seeds = [Seed::new(x, &dx, &vector), Seed::new(loss, &dl, &scalar)];
    core.forward().unwrap();
    core.backward(&seeds, G::Overwrite).unwrap();
    assert_eq!(core.gradient(x).unwrap().unwrap().data(), &[4., 6.]);
    core.backend.trigger.set(Some((true, 2)));
    assert!(matches!(
        core.backward(&seeds, G::Accumulate),
        Err(Error::StorageModified { .. })
    ));
    // The first root really changed leaf bytes before the later save failed.
    let slot = core.value_gradients[x.index].unwrap();
    assert_eq!(core.gradients[slot].data.data(), &[6., 8.]);
    assert_eq!(core.gradient_status(x).unwrap(), GradientStatus::Invalid);
    assert!(core.gradient(x).is_err());
    assert!(core.backward(&seeds, G::Accumulate).is_err());
    let version = owner.version().unwrap();
    assert!(core.sgd(&[x], 0.1).is_err());
    assert_eq!(owner.version().unwrap(), version);
    core.zero_grad().unwrap();
    core.forward().unwrap();
    core.backward(&seeds, G::Accumulate).unwrap();
    assert_eq!(core.gradient(x).unwrap().unwrap().data(), &[20., 6.]);
    core.sgd(&[x], 0.1).unwrap();
}

#[test]
fn partial_sgd_rechecks_each_writer_and_invalidates_all_gradients() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let y = b.leaf([2], true).unwrap();
    let a = b.unary(UnaryOp::Square, x).unwrap();
    let c = b.unary(UnaryOp::Square, y).unwrap();
    let sum = b.binary(BinaryOp::Add, a, c).unwrap();
    let loss = b.reduce(ReduceOp::Sum, sum, &[0], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let first = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let second = Shared::new(Tensor::from_vec([2], vec![3., 4.]).unwrap()).unwrap();
    let backend = AliasWriter {
        host: Host(Cpu::Reference),
        owner: second.clone(),
        trigger: Cell::new(None),
        panic_after_write: Cell::new(false),
        forward_write_after: Cell::new(None),
    };
    let mut core = Core::prepare(
        backend,
        &program,
        &[
            Binding::new(x, first.clone()),
            Binding::new(y, second.clone()),
        ],
        PrepareOptions::default(),
    )
    .unwrap();
    core.forward().unwrap();
    core.backward_scalar(G::Overwrite).unwrap();
    core.backend.trigger.set(Some((false, 1)));
    assert!(matches!(
        core.sgd(&[x, y], 0.1),
        Err(Error::StorageModified { .. })
    ));
    assert_eq!(first.read().unwrap().data(), &[0.8, 1.6]);
    assert_eq!(second.read().unwrap().data(), &[11., 4.]);
    assert_eq!(core.gradient_status(x).unwrap(), GradientStatus::Invalid);
    assert_eq!(core.gradient_status(y).unwrap(), GradientStatus::Invalid);
    assert!(core.gradient(x).is_err());
    assert!(core.gradient(y).is_err());
    assert!(core.output(loss).is_err());
    assert!(core.sgd(&[x, y], 0.1).is_err());
    core.forward().unwrap();
    assert_eq!(core.gradient_status(x).unwrap(), GradientStatus::Invalid);
    core.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(core.gradient(x).unwrap().unwrap().data(), &[1.6, 3.2]);
    assert_eq!(core.gradient(y).unwrap().unwrap().data(), &[22., 8.]);
}

#[test]
fn interleaved_forward_writes_are_rejected_even_without_saved_payloads() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let a = b.unary(UnaryOp::Neg, x).unwrap();
    let same = b.unary(UnaryOp::Neg, a).unwrap();
    let changed = b.unary(UnaryOp::Neg, x).unwrap();
    let total = b.binary(BinaryOp::Add, same, changed).unwrap();
    let loss = b.reduce(ReduceOp::Sum, total, &[0], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let backend = AliasWriter {
        host: Host(Cpu::Reference),
        owner: owner.clone(),
        trigger: Cell::new(None),
        panic_after_write: Cell::new(false),
        forward_write_after: Cell::new(Some(2)),
    };
    let mut core = Core::prepare(
        backend,
        &program,
        &[Binding::new(x, owner)],
        PrepareOptions::default(),
    )
    .unwrap();
    assert!(matches!(core.forward(), Err(Error::StorageModified { .. })));
    assert_eq!(
        core.owners[loss.index]
            .as_ref()
            .unwrap()
            .read()
            .unwrap()
            .data(),
        &[-8.]
    );
    assert!(core.output(loss).is_err());
    assert_eq!(core.gradient_status(x).unwrap(), GradientStatus::Uncomputed);
    assert!(core.backward_scalar(G::Overwrite).is_err());
    core.forward().unwrap();
    assert_eq!(core.output(loss).unwrap().storage().data(), &[0.]);
    core.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(core.gradient(x).unwrap().unwrap().data(), &[0., 0.]);
}

#[test]
fn scalar_seed_and_gradient_recovery_survive_execution_unwinding() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let q = b.unary(UnaryOp::Square, x).unwrap();
    let loss = b.reduce(ReduceOp::Sum, q, &[0], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let backend = AliasWriter {
        host: Host(Cpu::Reference),
        owner: owner.clone(),
        trigger: Cell::new(None),
        panic_after_write: Cell::new(false),
        forward_write_after: Cell::new(None),
    };
    let mut core = Core::prepare(
        backend,
        &program,
        &[Binding::new(x, owner)],
        PrepareOptions::default(),
    )
    .unwrap();
    core.forward().unwrap();
    core.backend.trigger.set(Some((true, 1)));
    core.backend.panic_after_write.set(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        core.backward_scalar(G::Overwrite)
    }));
    assert!(result.is_err());
    assert_eq!(core.gradient_status(x).unwrap(), GradientStatus::Invalid);
    assert!(core.gradient(x).is_err());
    assert!(core.sgd(&[x], 0.1).is_err());
    core.zero_grad().unwrap();
    core.forward().unwrap();
    core.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(core.gradient(x).unwrap().unwrap().data(), &[18., 4.]);
}
