#![cfg(feature = "cuda")]
use fast_audio::{
    BinaryOperand as Side, GradientMode as G, Layout, MatmulStrategy as Route, Shared, Tensor,
    cuda::{Cuda, MatmulBackward, MatmulPlan},
};
#[path = "support/matmul.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Matrix tests use their own independent references."
)]
mod support;
use contract::*;

fn resources(
    c: &Case,
    route: Route,
    workspace: usize,
    metadata: usize,
    shared: usize,
    backward: bool,
) {
    assert_eq!(metadata, if backward { 128 } else { 96 });
    assert_eq!(workspace, 0);
    assert_eq!(shared > 0, route == Route::StridedCuda);
    assert!(shared <= 2048);
    if c.direct {
        assert_eq!(route, Route::Cublas);
    }
}
#[test]
fn cuda_prepared_matmul_and_vjps_match_independent_oracles() {
    let cuda = Cuda::new(0).unwrap();
    let mut export = Exporter::new(Some("matmul-cuda"));
    for (i, c) in cases().iter().enumerate() {
        let p = MatmulPlan::new(&cuda, &c.a, &c.b, &c.y).unwrap();
        assert_eq!(p.left_layout(), &c.a);
        assert_eq!(p.right_layout(), &c.b);
        assert_eq!(p.destination_layout(), &c.y);
        resources(
            c,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            p.shared_memory_bytes_per_block(),
            false,
        );
        let backward = SIDES.map(|side| {
            MatmulBackward::new(&cuda, side, &c.a, &c.b, &c.dy, c.gradient(side)).unwrap()
        });
        for (p, side) in backward.iter().zip(SIDES) {
            assert_eq!(p.operand(), side);
            assert_eq!(p.left_layout(), &c.a);
            assert_eq!(p.right_layout(), &c.b);
            assert_eq!(p.upstream_layout(), &c.dy);
            assert_eq!(p.destination_layout(), c.gradient(side));
            assert_eq!(p.input_dependencies(), c.dependencies(side));
            resources(
                c,
                p.strategy(),
                p.workspace_bytes(),
                p.host_metadata_bytes(),
                p.shared_memory_bytes_per_block(),
                true,
            );
        }
        for change in 0..2 {
            let a = buffer(&c.a, i + change);
            let b = buffer(&c.b, i + change + 1);
            let dy = buffer(&c.dy, i + change + 2);
            let aa = logical(&c.a, &a);
            let bb = logical(&c.b, &b);
            let dd = logical(&c.dy, &dy);
            let left = cuda.upload(&a).unwrap();
            let right = cuda.upload(&b).unwrap();
            let incoming = cuda.upload(&dy).unwrap();
            let initial = seed(&c.y, false);
            let mut out = cuda.upload(&initial).unwrap();
            support::no_allocations(|| {
                p.run(
                    &left.view(&c.a).unwrap(),
                    &right.view(&c.b).unwrap(),
                    &mut out.view_mut(&c.y).unwrap(),
                )
                .unwrap()
            });
            let y = cuda.download(&out).unwrap();
            verify(
                &c.y,
                &forward(c.dims, &aa, &bb),
                &initial,
                &y,
                G::Overwrite,
                c.dims[2] == 0,
            );
            p.run(
                &left.view(&c.a).unwrap(),
                &right.view(&c.b).unwrap(),
                &mut out.view_mut(&c.y).unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&y));
            let av = left.view(&c.a).unwrap();
            let bv = right.view(&c.b).unwrap();
            let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
            for (side_index, (p, side)) in backward.iter().zip(SIDES).enumerate() {
                let grad = c.gradient(side);
                let expected = derivative(c.dims, side, &aa, &bb, &dd);
                let deps = p.input_dependencies();
                let inputs = [
                    if deps[0] || change == 1 {
                        Some(&av)
                    } else {
                        None
                    },
                    if deps[1] || change == 1 {
                        Some(&bv)
                    } else {
                        None
                    },
                ];
                for (mode_index, mode) in [G::Overwrite, G::Accumulate].into_iter().enumerate() {
                    let initial = seed(grad, mode == G::Accumulate);
                    let mut dx = cuda.upload(&initial).unwrap();
                    support::no_allocations(|| {
                        p.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dx.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap()
                    });
                    let actual = cuda.download(&dx).unwrap();
                    verify(
                        grad,
                        &expected,
                        &initial,
                        &actual,
                        mode,
                        c.dims.contains(&0),
                    );
                    let mut repeat = cuda.upload(&initial).unwrap();
                    p.run(
                        inputs[0],
                        inputs[1],
                        &incoming.view(&c.dy).unwrap(),
                        &mut repeat.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap();
                    assert_eq!(bits(&cuda.download(&repeat).unwrap()), bits(&actual));
                    if export.enabled() {
                        results[mode_index][side_index] = Some(actual.clone());
                    }
                    if mode == G::Overwrite {
                        identities(c, side, &a, &b, &y, &dy, &actual);
                    } else {
                        p.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dx.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap();
                        verify(
                            grad,
                            &expected,
                            &actual,
                            &cuda.download(&dx).unwrap(),
                            mode,
                            false,
                        );
                    }
                }
            }
            if export.enabled() {
                export.matrix(Sample {
                    index: i,
                    binding: change,
                    backend: "cuda",
                    case: c,
                    inputs: [&a, &b, &dy],
                    output: &y,
                    gradients: [
                        [
                            results[0][0].as_ref().unwrap(),
                            results[0][1].as_ref().unwrap(),
                        ],
                        [
                            results[1][0].as_ref().unwrap(),
                            results[1][1].as_ref().unwrap(),
                        ],
                    ],
                });
            }
            assert_eq!(bits(&cuda.download(&left).unwrap()), bits(&a));
            assert_eq!(bits(&cuda.download(&right).unwrap()), bits(&b));
            assert_eq!(bits(&cuda.download(&incoming).unwrap()), bits(&dy));
        }
    }
    export.finish();
}
#[test]
fn cuda_prepared_matmul_rejects_geometry_dependencies_and_sessions() {
    let cuda = Cuda::new(0).unwrap();
    let shifted =
        |l: &Layout| Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap();
    for c in cases() {
        let left = cuda.upload(&buffer(&c.a, 0)).unwrap();
        let right = cuda.upload(&buffer(&c.b, 1)).unwrap();
        let dy = cuda.upload(&buffer(&c.dy, 2)).unwrap();
        let av = left.view(&c.a).unwrap();
        let bv = right.view(&c.b).unwrap();
        let initial = seed(&c.y, false);
        let mut out = cuda.upload(&initial).unwrap();
        let p = MatmulPlan::new(&cuda, &c.a, &c.b, &c.y).unwrap();
        assert!(
            p.run(
                &left.view(&shifted(&c.a)).unwrap(),
                &bv,
                &mut out.view_mut(&c.y).unwrap()
            )
            .is_err()
        );
        assert!(
            p.run(
                &av,
                &right.view(&shifted(&c.b)).unwrap(),
                &mut out.view_mut(&c.y).unwrap()
            )
            .is_err()
        );
        assert!(
            p.run(&av, &bv, &mut out.view_mut(&shifted(&c.y)).unwrap())
                .is_err()
        );
        assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&initial));
        for side in SIDES {
            let grad = c.gradient(side);
            let p = MatmulBackward::new(&cuda, side, &c.a, &c.b, &c.dy, grad).unwrap();
            let initial = seed(grad, true);
            let mut out = cuda.upload(&initial).unwrap();
            for j in 0..2 {
                if p.input_dependencies()[j] {
                    let mut inputs = [Some(&av), Some(&bv)];
                    inputs[j] = None;
                    assert!(
                        p.run(
                            inputs[0],
                            inputs[1],
                            &dy.view(&c.dy).unwrap(),
                            &mut out.view_mut(grad).unwrap(),
                            G::Overwrite
                        )
                        .is_err()
                    );
                }
            }
            assert!(
                p.run(
                    Some(&left.view(&shifted(&c.a)).unwrap()),
                    Some(&bv),
                    &dy.view(&c.dy).unwrap(),
                    &mut out.view_mut(grad).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&av),
                    Some(&right.view(&shifted(&c.b)).unwrap()),
                    &dy.view(&c.dy).unwrap(),
                    &mut out.view_mut(grad).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&av),
                    Some(&bv),
                    &dy.view(&shifted(&c.dy)).unwrap(),
                    &mut out.view_mut(grad).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&av),
                    Some(&bv),
                    &dy.view(&c.dy).unwrap(),
                    &mut out.view_mut(&shifted(grad)).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
            assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&initial));
        }
    }
    let other = Cuda::new(0).unwrap();
    for n in [0, 3] {
        let l = Layout::contiguous([n, n]).unwrap();
        let home = cuda.zeros([n, n]).unwrap();
        let foreign = other.zeros([n, n]).unwrap();
        let mut out = cuda.zeros([n, n]).unwrap();
        let mut foreign_out = other.zeros([n, n]).unwrap();
        let p = MatmulPlan::new(&cuda, &l, &l, &l).unwrap();
        assert!(
            p.run(
                &foreign.view(&l).unwrap(),
                &home.view(&l).unwrap(),
                &mut out.view_mut(&l).unwrap()
            )
            .is_err()
        );
        assert!(
            p.run(
                &home.view(&l).unwrap(),
                &foreign.view(&l).unwrap(),
                &mut out.view_mut(&l).unwrap()
            )
            .is_err()
        );
        assert!(
            p.run(
                &home.view(&l).unwrap(),
                &home.view(&l).unwrap(),
                &mut foreign_out.view_mut(&l).unwrap()
            )
            .is_err()
        );
        for side in SIDES {
            let p = MatmulBackward::new(&cuda, side, &l, &l, &l, &l).unwrap();
            assert!(
                p.run(
                    Some(&foreign.view(&l).unwrap()),
                    Some(&home.view(&l).unwrap()),
                    &home.view(&l).unwrap(),
                    &mut out.view_mut(&l).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&home.view(&l).unwrap()),
                    Some(&foreign.view(&l).unwrap()),
                    &home.view(&l).unwrap(),
                    &mut out.view_mut(&l).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&home.view(&l).unwrap()),
                    Some(&home.view(&l).unwrap()),
                    &foreign.view(&l).unwrap(),
                    &mut out.view_mut(&l).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&home.view(&l).unwrap()),
                    Some(&home.view(&l).unwrap()),
                    &home.view(&l).unwrap(),
                    &mut foreign_out.view_mut(&l).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
        }
        assert!(
            cuda.download(&out)
                .unwrap()
                .data()
                .iter()
                .all(|x| x.to_bits() == 0)
        );
        assert!(
            other
                .download(&foreign_out)
                .unwrap()
                .data()
                .iter()
                .all(|x| x.to_bits() == 0)
        );
    }
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([3, 4]).unwrap();
    let y = Layout::contiguous([2, 4]).unwrap();
    let flat = Layout::contiguous([6]).unwrap();
    let wrong = Layout::contiguous([4, 2]).unwrap();
    let overlap = Layout::from_strides([2, 4], [0, 1], 0).unwrap();
    assert!(MatmulPlan::new(&cuda, &flat, &b, &y).is_err());
    assert!(MatmulPlan::new(&cuda, &a, &a, &y).is_err());
    assert!(MatmulPlan::new(&cuda, &a, &b, &wrong).is_err());
    assert!(MatmulPlan::new(&cuda, &a, &b, &overlap).is_err());
    for side in SIDES {
        let grad = if side == Side::Left { &a } else { &b };
        assert!(MatmulBackward::new(&cuda, side, &a, &b, &wrong, grad).is_err());
        assert!(MatmulBackward::new(&cuda, side, &a, &b, &y, &flat).is_err());
        assert!(
            MatmulBackward::new(
                &cuda,
                side,
                &a,
                &b,
                &y,
                &Layout::from_strides(grad.shape().dims(), [0, 1], 0).unwrap()
            )
            .is_err()
        );
    }
}
#[test]
fn cuda_prepared_matmul_zero_contractions_preserve_accumulation_bits() {
    let cuda = Cuda::new(0).unwrap();
    for dims in [[0, 3, 4], [3, 0, 4], [3, 4, 0]] {
        let [m, n, k] = dims;
        let a = Layout::contiguous([m, k]).unwrap();
        let b = Layout::contiguous([k, n]).unwrap();
        let dy = Layout::contiguous([m, n]).unwrap();
        let incoming = cuda.zeros([m * n]).unwrap();
        for side in SIDES {
            let grad = if side == Side::Left {
                layout(m, k, 5)
            } else {
                layout(k, n, 4)
            };
            let p = MatmulBackward::new(&cuda, side, &a, &b, &dy, &grad).unwrap();
            assert_eq!(p.input_dependencies(), [false; 2]);
            let mut initial = seed(&grad, true);
            for (i, index) in addresses(&grad).into_iter().enumerate() {
                initial.data_mut()[index] =
                    f32::from_bits(if i % 2 == 0 { 0x80000000 } else { 0x7fc54321 });
            }
            let mut out = cuda.upload(&initial).unwrap();
            support::no_allocations(|| {
                p.run(
                    None,
                    None,
                    &incoming.view(&dy).unwrap(),
                    &mut out.view_mut(&grad).unwrap(),
                    G::Accumulate,
                )
                .unwrap()
            });
            assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&initial));
            p.run(
                None,
                None,
                &incoming.view(&dy).unwrap(),
                &mut out.view_mut(&grad).unwrap(),
                G::Overwrite,
            )
            .unwrap();
            let actual = cuda.download(&out).unwrap();
            for index in addresses(&grad) {
                assert_eq!(actual.data()[index].to_bits(), 0);
            }
            holes(&grad, &initial, &actual);
        }
        if k == 0 {
            let no_storage = cuda.zeros([0]).unwrap();
            let result = layout(m, n, 5);
            let initial = seed(&result, false);
            let mut out = cuda.upload(&initial).unwrap();
            MatmulPlan::new(&cuda, &a, &b, &result)
                .unwrap()
                .run(
                    &no_storage.view(&a).unwrap(),
                    &no_storage.view(&b).unwrap(),
                    &mut out.view_mut(&result).unwrap(),
                )
                .unwrap();
            let actual = cuda.download(&out).unwrap();
            for i in addresses(&result) {
                assert_eq!(actual.data()[i].to_bits(), 0);
            }
            holes(&result, &initial, &actual);
        }
    }
}
#[test]
fn cuda_prepared_matmul_saved_versions_and_aliased_view_composition() {
    let cuda = Cuda::new(0).unwrap();
    let a = Layout::from_strides([3, 3], [0, 1], 1).unwrap();
    let b = Layout::from_strides([3, 3], [1, 1], 0).unwrap();
    let dense = Layout::contiguous([3, 3]).unwrap();
    let host = Tensor::from_vec([7], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).unwrap();
    let base = cuda.upload(&host).unwrap();
    let dy = buffer(&dense, 1);
    let incoming = cuda.upload(&dy).unwrap();
    let mut grad = cuda.zeros([7]).unwrap();
    for side in SIDES {
        let mut local = cuda.zeros([3, 3]).unwrap();
        MatmulBackward::new(&cuda, side, &a, &b, &dense, &dense)
            .unwrap()
            .run(
                Some(&base.view(&a).unwrap()),
                Some(&base.view(&b).unwrap()),
                &incoming.view(&dense).unwrap(),
                &mut local.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        let original = if side == Side::Left { &a } else { &b };
        fast_audio::cuda::ViewBackward::new(&cuda, [7], original, &dense, 65536)
            .unwrap()
            .run(
                &local.view(&dense).unwrap(),
                &mut grad,
                if side == Side::Left {
                    G::Overwrite
                } else {
                    G::Accumulate
                },
            )
            .unwrap();
    }
    let objective = |x: &[f64]| {
        let aa: Vec<_> = addresses(&a).into_iter().map(|i| x[i]).collect();
        let bb: Vec<_> = addresses(&b).into_iter().map(|i| x[i]).collect();
        forward([3, 3, 3], &aa, &bb)
            .iter()
            .zip(dy.data())
            .map(|(y, &d)| y * d as f64)
            .sum::<f64>()
    };
    let actual = cuda.download(&grad).unwrap();
    for i in 0..7 {
        let mut x: Vec<_> = host.data().iter().map(|&x| x as f64).collect();
        x[i] += 1e-4;
        let plus = objective(&x);
        x[i] -= 2e-4;
        close(actual.data()[i], (plus - objective(&x)) / 2e-4);
    }
    let owner = Shared::new(base).unwrap();
    let saved = owner.save(&std::sync::Arc::new(a.clone())).unwrap();
    let mut out = cuda.zeros([3, 3]).unwrap();
    let p = MatmulBackward::new(&cuda, Side::Right, &a, &b, &dense, &dense).unwrap();
    {
        let read = saved.read().unwrap();
        p.run(
            Some(&read.view().unwrap()),
            None,
            &incoming.view(&dense).unwrap(),
            &mut out.view_mut(&dense).unwrap(),
            G::Overwrite,
        )
        .unwrap();
    }
    let before = cuda.download(&out).unwrap();
    drop(owner.write().unwrap());
    let result = (|| -> fast_audio::Result<()> {
        let read = saved.read()?;
        p.run(
            Some(&read.view()?),
            None,
            &incoming.view(&dense)?,
            &mut out.view_mut(&dense)?,
            G::Overwrite,
        )
    })();
    assert!(result.is_err());
    assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&before));
}
#[test]
fn cuda_prepared_matmul_large_preparation_tile_and_zero_grid_reuse() {
    let cuda = Cuda::new(0).unwrap();
    let a = Layout::from_strides([1, 5_000_000_001], [0, 0], 0).unwrap();
    let b = Layout::from_strides([5_000_000_001, 1], [0, 0], 0).unwrap();
    let y = Layout::contiguous([1, 1]).unwrap();
    let p = MatmulPlan::new(&cuda, &a, &b, &y).unwrap();
    assert_eq!(p.strategy(), Route::StridedCuda);
    assert_eq!(p.workspace_bytes(), 0);
    assert_eq!(p.shared_memory_bytes_per_block(), 2048);
    assert!(p.host_metadata_bytes() < 1024);
    let empty = Layout::contiguous([usize::MAX, 0]).unwrap();
    let b = Layout::contiguous([0, 0]).unwrap();
    let source = cuda.zeros([0]).unwrap();
    let mut out = cuda.zeros([0]).unwrap();
    MatmulPlan::new(&cuda, &empty, &b, &empty)
        .unwrap()
        .run(
            &source.view(&empty).unwrap(),
            &source.view(&b).unwrap(),
            &mut out.view_mut(&empty).unwrap(),
        )
        .unwrap();
    let count = 65535 * 16 + 17;
    for transposed in [false, true] {
        let (m, n) = if transposed { (1, count) } else { (count, 1) };
        let a = Layout::from_strides([m, 1], [0, 0], 0).unwrap();
        let b = Layout::from_strides([1, n], [0, 0], 0).unwrap();
        let result = if transposed {
            Layout::from_strides([m, n], [isize::MIN, 2], 0).unwrap()
        } else {
            Layout::from_strides([m, n], [2, isize::MIN], 0).unwrap()
        };
        let left = cuda
            .upload(&Tensor::from_vec([], vec![3.]).unwrap())
            .unwrap();
        let right = cuda
            .upload(&Tensor::from_vec([], vec![2.]).unwrap())
            .unwrap();
        let initial = seed(&result, false);
        let mut out = cuda.upload(&initial).unwrap();
        let p = MatmulPlan::new(&cuda, &a, &b, &result).unwrap();
        assert_eq!(p.strategy(), Route::StridedCuda);
        support::no_allocations(|| {
            p.run(
                &left.view(&a).unwrap(),
                &right.view(&b).unwrap(),
                &mut out.view_mut(&result).unwrap(),
            )
            .unwrap()
        });
        let actual = cuda.download(&out).unwrap();
        for i in addresses(&result) {
            assert_eq!(actual.data()[i], 6.);
        }
        holes(&result, &initial, &actual);
    }
    let n = 65535 * 256 + 17;
    let a = Layout::contiguous([1, 0]).unwrap();
    let b = Layout::contiguous([0, n]).unwrap();
    let result = Layout::contiguous([1, n]).unwrap();
    let mut out = cuda
        .upload(&Tensor::from_vec([n], vec![-0.; n]).unwrap())
        .unwrap();
    let source = cuda.zeros([0]).unwrap();
    let p = MatmulPlan::new(&cuda, &a, &b, &result).unwrap();
    assert_eq!(p.strategy(), Route::Zero);
    support::no_allocations(|| {
        p.run(
            &source.view(&a).unwrap(),
            &source.view(&b).unwrap(),
            &mut out.view_mut(&result).unwrap(),
        )
        .unwrap()
    });
    assert!(
        cuda.download(&out)
            .unwrap()
            .data()
            .iter()
            .all(|x| x.to_bits() == 0)
    );
}

#[test]
fn cuda_prepared_matmul_manual_training_matches_fp64_updates() {
    use fast_audio::{BinaryOp as B, ReduceOp, ReduceSpec, UnaryOp as U, cuda};
    macro_rules! v {
        ($t:ident,$l:ident) => {
            $t.view(&$l).unwrap()
        };
    }
    macro_rules! vm {
        ($t:ident,$l:ident) => {
            $t.view_mut(&$l).unwrap()
        };
    }
    let cuda = Cuda::new(0).unwrap();
    let [hx, ht, hw, hrate] = training_fixture();
    let x = cuda.upload(&hx).unwrap();
    let target = cuda.upload(&ht).unwrap();
    let mut weights = cuda.upload(&hw).unwrap();
    let rate = cuda.upload(&hrate).unwrap();
    let lx = Layout::contiguous([3, 2]).unwrap();
    let ly = lx.clone();
    let lw = Layout::contiguous([2, 2]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let spec = ReduceSpec::new(ReduceOp::Mean, ly.shape(), &[0, 1], false).unwrap();
    let product = MatmulPlan::new(&cuda, &lx, &lw, &ly).unwrap();
    let difference = cuda::BinaryPlan::new(&cuda, B::Sub, &ly, &ly, &ly).unwrap();
    let square = cuda::UnaryPlan::new(&cuda, U::Square, &ly, &ly).unwrap();
    let mean = cuda::ReducePlan::new(&cuda, &spec, &ly, &scalar).unwrap();
    let dmean = cuda::ReduceBackward::new(&cuda, &spec, &scalar, &ly).unwrap();
    let dsquare = cuda::UnaryBackward::new(&cuda, U::Square, &ly, &ly, &ly).unwrap();
    let ddifference =
        cuda::BinaryBackward::new(&cuda, B::Sub, Side::Left, &ly, &ly, &ly, &ly).unwrap();
    let dweights = MatmulBackward::new(&cuda, Side::Right, &lx, &lw, &ly, &lw).unwrap();
    assert_eq!(dweights.input_dependencies(), [true, false]);
    let scale = cuda::BinaryPlan::new(&cuda, B::Mul, &lw, &scalar, &lw).unwrap();
    let update = cuda::BinaryPlan::new(&cuda, B::Sub, &lw, &lw, &lw).unwrap();
    let copy = cuda::CopyPlan::new(&cuda, &lw, &lw).unwrap();
    let mut y = cuda.zeros([3, 2]).unwrap();
    let mut residual = cuda.zeros([3, 2]).unwrap();
    let mut squared = cuda.zeros([3, 2]).unwrap();
    let mut gsquare = cuda.zeros([3, 2]).unwrap();
    let mut gresidual = cuda.zeros([3, 2]).unwrap();
    let mut gy = cuda.zeros([3, 2]).unwrap();
    let mut loss = cuda.zeros([]).unwrap();
    let one = cuda
        .upload(&Tensor::from_vec([], vec![1.]).unwrap())
        .unwrap();
    let mut gw = cuda.zeros([2, 2]).unwrap();
    let mut scaled = cuda.zeros([2, 2]).unwrap();
    let mut next = cuda.zeros([2, 2]).unwrap();
    let mut control: Vec<_> = hw.data().iter().map(|&v| v as f64).collect();
    let mut first = 0.;
    let mut export = Exporter::new(Some("training-cuda"));
    for step in 0..8 {
        let before = cuda.download(&weights).unwrap();
        for (&a, &b) in before.data().iter().zip(&control) {
            close(a, b);
        }
        let (expected_loss, expected_grad) =
            training_control(&mut control, &hx, &ht, hrate.data()[0] as f64);
        support::no_allocations(|| {
            product
                .run(&v!(x, lx), &v!(weights, lw), &mut vm!(y, ly))
                .unwrap();
            difference
                .run(&v!(y, ly), &v!(target, ly), &mut vm!(residual, ly))
                .unwrap();
            square
                .run(&v!(residual, ly), &mut vm!(squared, ly))
                .unwrap();
            mean.run(&v!(squared, ly), &mut vm!(loss, scalar)).unwrap();
            dmean
                .run(&v!(one, scalar), &mut vm!(gsquare, ly), G::Overwrite)
                .unwrap();
            dsquare
                .run(
                    Some(&v!(residual, ly)),
                    &v!(gsquare, ly),
                    &mut vm!(gresidual, ly),
                    G::Overwrite,
                )
                .unwrap();
            ddifference
                .run(
                    None,
                    None,
                    &v!(gresidual, ly),
                    &mut vm!(gy, ly),
                    G::Overwrite,
                )
                .unwrap();
            dweights
                .run(
                    Some(&v!(x, lx)),
                    None,
                    &v!(gy, ly),
                    &mut vm!(gw, lw),
                    G::Overwrite,
                )
                .unwrap();
            scale
                .run(&v!(gw, lw), &v!(rate, scalar), &mut vm!(scaled, lw))
                .unwrap();
            update
                .run(&v!(weights, lw), &v!(scaled, lw), &mut vm!(next, lw))
                .unwrap();
            copy.run(&v!(next, lw), &mut vm!(weights, lw)).unwrap();
        });
        let actual_loss = cuda.download(&loss).unwrap();
        let actual_grad = cuda.download(&gw).unwrap();
        let actual_weights = cuda.download(&weights).unwrap();
        close(actual_loss.data()[0], expected_loss);
        support::close(actual_grad.data(), &expected_grad, 2e-5, 2e-5);
        support::close(actual_weights.data(), &control, 2e-5, 2e-5);
        if step == 0 {
            first = actual_loss.data()[0];
        }
        if step == 7 {
            assert!(actual_loss.data()[0] < first);
        }
        export_training(
            &mut export,
            "cuda",
            step,
            [&hx, &ht, &before, &hrate],
            &actual_loss,
            &actual_grad,
            &actual_weights,
        );
    }
    assert_eq!(bits(&cuda.download(&x).unwrap()), bits(&hx));
    assert_eq!(bits(&cuda.download(&target).unwrap()), bits(&ht));
    assert_eq!(bits(&cuda.download(&rate).unwrap()), bits(&hrate));
    export.finish();
}
