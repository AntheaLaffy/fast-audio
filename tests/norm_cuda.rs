#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, GradientMode, Layout, NormKind, NormSpec, ReduceOp, ReduceSpec, Shape, Shared,
    SoftmaxKind, SoftmaxSpec, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
    cuda::{self, Cuda},
};

fn deterministic_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u32 << 24) as f32;
            (u - 0.5) * 3.0
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        let err = (x - y).abs();
        assert!(
            err <= tol,
            "{label}[{i}] mismatch: cuda={x} cpu={y} err={err} > {tol}"
        );
    }
}

#[test]
fn cuda_norm_softmax_and_nonlinear_reductions_match_cpu_all_tiers() {
    let cuda = Cuda::new(0).unwrap();

    // 1. LayerNorm & RmsNorm across warp (<=128), vec4/block (128..=1024), general (>1024), and strided
    for kind in [NormKind::LayerNorm, NormKind::RmsNorm] {
        for &(dims, norm_dims) in &[
            (&[4, 1][..], 1),
            (&[7, 31][..], 1),
            (&[16, 64][..], 1),
            (&[9, 128][..], 1),
            (&[4, 256][..], 1), // vec4 aligned
            (&[3, 300][..], 1), // vec4 aligned
            (&[3, 301][..], 1), // reg non-vec4
            (&[2, 1024][..], 1),
            (&[2, 1500][..], 1), // general >1024
            (&[2, 3, 16][..], 2),
        ] {
            let shape = Shape::new(dims).unwrap();
            let layout = Layout::contiguous(dims).unwrap();
            let spec = NormSpec::new(kind, &shape, norm_dims, 1e-5).unwrap();
            let x_h = Tensor::from_vec(dims, deterministic_data(shape.len(), 17)).unwrap();
            let dy_h = Tensor::from_vec(dims, deterministic_data(shape.len(), 41)).unwrap();

            let mut y_cpu = Tensor::zeros(dims).unwrap();
            let mut dx_cpu = Tensor::zeros(dims).unwrap();
            cpu::NormPlan::new(&spec, &layout, &layout)
                .unwrap()
                .run(
                    &x_h.view(&layout).unwrap(),
                    &mut y_cpu.view_mut(&layout).unwrap(),
                )
                .unwrap();
            cpu::NormBackward::new(&spec, &layout, &layout, &layout)
                .unwrap()
                .run(
                    &x_h.view(&layout).unwrap(),
                    &dy_h.view(&layout).unwrap(),
                    &mut dx_cpu.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();

            let x_d = cuda.upload(&x_h).unwrap();
            let dy_d = cuda.upload(&dy_h).unwrap();
            let mut y_d = cuda.zeros(dims).unwrap();
            let mut dx_d = cuda.zeros(dims).unwrap();
            let fwd_d = cuda::NormPlan::new(&cuda, &spec, &layout, &layout).unwrap();
            let bwd_d = cuda::NormBackward::new(&cuda, &spec, &layout, &layout, &layout).unwrap();
            if spec.inner_len() <= 128 {
                assert_eq!(fwd_d.shared_memory_bytes_per_block(), 0);
            }
            support::no_allocations(|| {
                fwd_d
                    .run(
                        &x_d.view(&layout).unwrap(),
                        &mut y_d.view_mut(&layout).unwrap(),
                    )
                    .unwrap();
                bwd_d
                    .run(
                        &x_d.view(&layout).unwrap(),
                        &dy_d.view(&layout).unwrap(),
                        &mut dx_d.view_mut(&layout).unwrap(),
                        GradientMode::Overwrite,
                    )
                    .unwrap();
            });
            assert_close(
                cuda.download(&y_d).unwrap().data(),
                y_cpu.data(),
                2e-5,
                &format!("CUDA {kind:?} fwd {dims:?}"),
            );
            assert_close(
                cuda.download(&dx_d).unwrap().data(),
                dx_cpu.data(),
                5e-5,
                &format!("CUDA {kind:?} bwd {dims:?}"),
            );

            // Accumulate mode check
            let mut dx_cpu_acc = Tensor::from_vec(dims, vec![0.5; shape.len()]).unwrap();
            let mut dx_d_acc = cuda.upload(&dx_cpu_acc).unwrap();
            cpu::NormBackward::new(&spec, &layout, &layout, &layout)
                .unwrap()
                .run(
                    &x_h.view(&layout).unwrap(),
                    &dy_h.view(&layout).unwrap(),
                    &mut dx_cpu_acc.view_mut(&layout).unwrap(),
                    GradientMode::Accumulate,
                )
                .unwrap();
            bwd_d
                .run(
                    &x_d.view(&layout).unwrap(),
                    &dy_d.view(&layout).unwrap(),
                    &mut dx_d_acc.view_mut(&layout).unwrap(),
                    GradientMode::Accumulate,
                )
                .unwrap();
            assert_close(
                cuda.download(&dx_d_acc).unwrap().data(),
                dx_cpu_acc.data(),
                5e-5,
                &format!("CUDA {kind:?} bwd acc {dims:?}"),
            );
        }

        // Signed-stride reversed/permuted Norm
        let raw_x_h = Tensor::from_vec([5, 4, 3], deterministic_data(60, 77)).unwrap();
        let raw_dy_h = Tensor::from_vec([3, 3, 4], deterministic_data(36, 91)).unwrap();
        let base = Layout::contiguous([5, 4, 3]).unwrap();
        let view_l = base
            .slice(0, 4, 3, -1)
            .unwrap()
            .permute(&[2, 0, 1])
            .unwrap();
        let out_l = Layout::contiguous([3, 3, 4]).unwrap();
        let spec = NormSpec::new(kind, view_l.shape(), 2, 1e-5).unwrap();
        let mut y_cpu = Tensor::zeros([3, 3, 4]).unwrap();
        let mut dx_cpu = Tensor::zeros([3, 3, 4]).unwrap();
        cpu::NormPlan::new(&spec, &view_l, &out_l)
            .unwrap()
            .run(
                &raw_x_h.view(&view_l).unwrap(),
                &mut y_cpu.view_mut(&out_l).unwrap(),
            )
            .unwrap();
        cpu::NormBackward::new(&spec, &view_l, &out_l, &out_l)
            .unwrap()
            .run(
                &raw_x_h.view(&view_l).unwrap(),
                &raw_dy_h.view(&out_l).unwrap(),
                &mut dx_cpu.view_mut(&out_l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();

        let raw_x_d = cuda.upload(&raw_x_h).unwrap();
        let raw_dy_d = cuda.upload(&raw_dy_h).unwrap();
        let mut y_d = cuda.zeros([3, 3, 4]).unwrap();
        let mut dx_d = cuda.zeros([3, 3, 4]).unwrap();
        cuda::NormPlan::new(&cuda, &spec, &view_l, &out_l)
            .unwrap()
            .run(
                &raw_x_d.view(&view_l).unwrap(),
                &mut y_d.view_mut(&out_l).unwrap(),
            )
            .unwrap();
        cuda::NormBackward::new(&cuda, &spec, &view_l, &out_l, &out_l)
            .unwrap()
            .run(
                &raw_x_d.view(&view_l).unwrap(),
                &raw_dy_d.view(&out_l).unwrap(),
                &mut dx_d.view_mut(&out_l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert_close(
            cuda.download(&y_d).unwrap().data(),
            y_cpu.data(),
            2e-5,
            &format!("CUDA {kind:?} strided fwd"),
        );
        assert_close(
            cuda.download(&dx_d).unwrap().data(),
            dx_cpu.data(),
            5e-5,
            &format!("CUDA {kind:?} strided bwd"),
        );
    }

    // 2. Softmax & LogSoftmax across all tiers + non-last axis + run_from_output + NaN
    for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
        for &(dims, axis) in &[
            (&[4, 1][..], 1),
            (&[11, 31][..], 1),
            (&[16, 128][..], 1),
            (&[4, 256][..], 1),
            (&[4, 512][..], 1),
            (&[2, 1500][..], 1),
            (&[3, 7, 5][..], 1),
        ] {
            let shape = Shape::new(dims).unwrap();
            let layout = Layout::contiguous(dims).unwrap();
            let spec = SoftmaxSpec::new(kind, &shape, axis).unwrap();
            let x_h = Tensor::from_vec(dims, deterministic_data(shape.len(), 61)).unwrap();
            let dy_h = Tensor::from_vec(dims, deterministic_data(shape.len(), 73)).unwrap();
            let mut y_cpu = Tensor::zeros(dims).unwrap();
            let mut dx_cpu = Tensor::zeros(dims).unwrap();
            let mut dx_cpu_from_y = Tensor::zeros(dims).unwrap();
            cpu::SoftmaxPlan::new(&spec, &layout, &layout)
                .unwrap()
                .run(
                    &x_h.view(&layout).unwrap(),
                    &mut y_cpu.view_mut(&layout).unwrap(),
                )
                .unwrap();
            let bwd_cpu = cpu::SoftmaxBackward::new(&spec, &layout, &layout, &layout).unwrap();
            bwd_cpu
                .run(
                    &x_h.view(&layout).unwrap(),
                    &dy_h.view(&layout).unwrap(),
                    &mut dx_cpu.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            bwd_cpu
                .run_from_output(
                    &y_cpu.view(&layout).unwrap(),
                    &dy_h.view(&layout).unwrap(),
                    &mut dx_cpu_from_y.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            assert_close(
                dx_cpu_from_y.data(),
                dx_cpu.data(),
                1e-5,
                &format!("CPU {kind:?} bwd run_from_output {dims:?} axis={axis}"),
            );

            let x_d = cuda.upload(&x_h).unwrap();
            let dy_d = cuda.upload(&dy_h).unwrap();
            let mut y_d = cuda.zeros(dims).unwrap();
            let mut dx_d = cuda.zeros(dims).unwrap();
            let mut dx_d_from_y = cuda.zeros(dims).unwrap();
            let fwd_d = cuda::SoftmaxPlan::new(&cuda, &spec, &layout, &layout).unwrap();
            let bwd_d =
                cuda::SoftmaxBackward::new(&cuda, &spec, &layout, &layout, &layout).unwrap();
            support::no_allocations(|| {
                fwd_d
                    .run(
                        &x_d.view(&layout).unwrap(),
                        &mut y_d.view_mut(&layout).unwrap(),
                    )
                    .unwrap();
                bwd_d
                    .run(
                        &x_d.view(&layout).unwrap(),
                        &dy_d.view(&layout).unwrap(),
                        &mut dx_d.view_mut(&layout).unwrap(),
                        GradientMode::Overwrite,
                    )
                    .unwrap();
                bwd_d
                    .run_from_output(
                        &y_d.view(&layout).unwrap(),
                        &dy_d.view(&layout).unwrap(),
                        &mut dx_d_from_y.view_mut(&layout).unwrap(),
                        GradientMode::Overwrite,
                    )
                    .unwrap();
            });
            assert_close(
                cuda.download(&y_d).unwrap().data(),
                y_cpu.data(),
                2e-5,
                &format!("CUDA {kind:?} fwd {dims:?} axis={axis}"),
            );
            assert_close(
                cuda.download(&dx_d).unwrap().data(),
                dx_cpu.data(),
                2e-5,
                &format!("CUDA {kind:?} bwd {dims:?} axis={axis}"),
            );
            assert_close(
                cuda.download(&dx_d_from_y).unwrap().data(),
                dx_cpu.data(),
                2e-5,
                &format!("CUDA {kind:?} bwd run_from_output {dims:?} axis={axis}"),
            );

            // Verify Accumulate mode on run_from_output
            let mut dx_d_acc = cuda
                .upload(&Tensor::from_vec(dims, vec![0.25; shape.len()]).unwrap())
                .unwrap();
            bwd_d
                .run_from_output(
                    &y_d.view(&layout).unwrap(),
                    &dy_d.view(&layout).unwrap(),
                    &mut dx_d_acc.view_mut(&layout).unwrap(),
                    GradientMode::Accumulate,
                )
                .unwrap();
            let expected_acc: Vec<f32> = dx_cpu.data().iter().map(|&v| v + 0.25).collect();
            assert_close(
                cuda.download(&dx_d_acc).unwrap().data(),
                &expected_acc,
                2e-5,
                &format!("CUDA {kind:?} bwd run_from_output acc {dims:?} axis={axis}"),
            );
        }
    }

    // 3. Nonlinear reductions (Max, Min, LogSumExp) across warp (<=128), block (>128), ties, NaN/inf
    for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::LogSumExp] {
        for &(dims, axes) in &[
            (&[4, 8][..], &[1][..]),
            (&[3, 128][..], &[1][..]),
            (&[2, 300][..], &[1][..]),
            (&[2, 5, 4][..], &[1][..]),
        ] {
            let lx = Layout::contiguous(dims).unwrap();
            let spec = ReduceSpec::new(op, lx.shape(), axes, false).unwrap();
            let ly = Layout::contiguous(spec.output_shape().dims()).unwrap();
            let mut vals = deterministic_data(lx.shape().len(), 97);
            // Inject deliberate ties in first row
            if vals.len() >= 4 {
                vals[1] = 2.5;
                vals[2] = 2.5;
            }
            let x_h = Tensor::from_vec(dims, vals).unwrap();
            let dy_h = Tensor::from_vec(
                spec.output_shape().dims(),
                deterministic_data(spec.output_shape().len(), 109),
            )
            .unwrap();

            let mut y_cpu = Tensor::zeros(spec.output_shape().dims()).unwrap();
            let mut dx_cpu = Tensor::zeros(dims).unwrap();
            cpu::ReducePlan::new(&spec, &lx, &ly)
                .unwrap()
                .run(&x_h.view(&lx).unwrap(), &mut y_cpu.view_mut(&ly).unwrap())
                .unwrap();
            cpu::ReduceBackward::with_input(&spec, &lx, &ly, &lx)
                .unwrap()
                .run_with_input(
                    Some(&x_h.view(&lx).unwrap()),
                    &dy_h.view(&ly).unwrap(),
                    &mut dx_cpu.view_mut(&lx).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();

            let x_d = cuda.upload(&x_h).unwrap();
            let dy_d = cuda.upload(&dy_h).unwrap();
            let mut y_d = cuda.zeros(spec.output_shape().dims()).unwrap();
            let mut dx_d = cuda.zeros(dims).unwrap();
            cuda::ReducePlan::new(&cuda, &spec, &lx, &ly)
                .unwrap()
                .run(&x_d.view(&lx).unwrap(), &mut y_d.view_mut(&ly).unwrap())
                .unwrap();
            cuda::ReduceBackward::with_input(&cuda, &spec, &lx, &ly, &lx)
                .unwrap()
                .run_with_input(
                    Some(&x_d.view(&lx).unwrap()),
                    &dy_d.view(&ly).unwrap(),
                    &mut dx_d.view_mut(&lx).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            assert_close(
                cuda.download(&y_d).unwrap().data(),
                y_cpu.data(),
                2e-5,
                &format!("CUDA {op:?} fwd {dims:?}"),
            );
            assert_close(
                cuda.download(&dx_d).unwrap().data(),
                dx_cpu.data(),
                2e-5,
                &format!("CUDA {op:?} bwd {dims:?}"),
            );
        }
    }
}

#[test]
fn cuda_tape_norm_softmax_nl_reduce_fused_and_captured_matches_cpu() {
    let cuda = Cuda::new(0).unwrap();
    let cpu = Cpu::Reference;
    let x_init = Tensor::from_vec([2, 4, 64], deterministic_data(512, 5)).unwrap();
    let g_init = Tensor::from_vec([1, 4, 64], deterministic_data(256, 13)).unwrap();
    let b_init = Tensor::from_vec([1, 4, 64], deterministic_data(256, 19)).unwrap();
    let target = Tensor::from_vec([2, 4, 64], deterministic_data(512, 23)).unwrap();

    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 4, 64], true).unwrap();
    let gamma = b.leaf([1, 4, 64], true).unwrap();
    let beta = b.leaf([1, 4, 64], true).unwrap();
    let tgt = b.leaf([2, 4, 64], false).unwrap();

    let gn = b.group_norm(x, 2, 1e-5).unwrap();
    let scaled = b.binary(BinaryOp::Mul, gn, gamma).unwrap();
    let shifted = b.binary(BinaryOp::Add, scaled, beta).unwrap();
    let rn = b.rms_norm(shifted, 1, 1e-5).unwrap();
    let sm = b.softmax(rn, 2).unwrap();
    let lsm = b.log_softmax(shifted, 2).unwrap();
    let combo = b.binary(BinaryOp::Add, sm, lsm).unwrap();
    let lse = b.reduce(ReduceOp::LogSumExp, combo, &[2], true).unwrap();
    let diff = b.binary(BinaryOp::Sub, combo, tgt).unwrap();
    let diff2 = b.binary(BinaryOp::Add, diff, lse).unwrap();
    let sq = b.unary(UnaryOp::Square, diff2).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let cx = Shared::new(x_init.clone()).unwrap();
    let cg = Shared::new(g_init.clone()).unwrap();
    let cb = Shared::new(b_init.clone()).unwrap();
    let ct = Shared::new(target.clone()).unwrap();
    let mut cpu_tape = cpu::Tape::new(
        cpu,
        &program,
        &[
            Binding::new(x, cx.clone()),
            Binding::new(gamma, cg.clone()),
            Binding::new(beta, cb.clone()),
            Binding::new(tgt, ct.clone()),
        ],
    )
    .unwrap();

    let make_gpu = || {
        let gx = Shared::new(cuda.upload(&x_init).unwrap()).unwrap();
        let gg = Shared::new(cuda.upload(&g_init).unwrap()).unwrap();
        let gb = Shared::new(cuda.upload(&b_init).unwrap()).unwrap();
        let gt = Shared::new(cuda.upload(&target).unwrap()).unwrap();
        let tape = cuda::Tape::new(
            &cuda,
            &program,
            &[
                Binding::new(x, gx.clone()),
                Binding::new(gamma, gg.clone()),
                Binding::new(beta, gb.clone()),
                Binding::new(tgt, gt.clone()),
            ],
        )
        .unwrap();
        (tape, gx, gg, gb)
    };

    let (mut gpu_uncap, ux, ug, ub) = make_gpu();
    let (mut gpu_cap, kx, kg, kb) = make_gpu();

    for _ in 0..5 {
        cpu_tape.step_scalar(&[x, gamma, beta], 0.05).unwrap();
        gpu_uncap.step_scalar(&[x, gamma, beta], 0.05).unwrap();
        gpu_cap
            .step_scalar_captured(&[x, gamma, beta], 0.05)
            .unwrap();
    }
    assert!(gpu_cap.has_captured_graph());
    cuda.synchronize().unwrap();

    for (name, c, u, k) in [
        (
            "x",
            cx.read().unwrap(),
            cuda.download(&ux.read().unwrap()).unwrap(),
            cuda.download(&kx.read().unwrap()).unwrap(),
        ),
        (
            "gamma",
            cg.read().unwrap(),
            cuda.download(&ug.read().unwrap()).unwrap(),
            cuda.download(&kg.read().unwrap()).unwrap(),
        ),
        (
            "beta",
            cb.read().unwrap(),
            cuda.download(&ub.read().unwrap()).unwrap(),
            cuda.download(&kb.read().unwrap()).unwrap(),
        ),
    ] {
        let ubits: Vec<u32> = u.data().iter().map(|v| v.to_bits()).collect();
        let kbits: Vec<u32> = k.data().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            ubits, kbits,
            "CUDA captured vs uncaptured bit drift on {name}"
        );
        assert_close(k.data(), c.data(), 5e-5, &format!("CUDA vs CPU {name}"));
    }

    // Verify NVRTC fused post-Softmax tail + CUDA Graph capture bit-exactness.
    let g2_init = Tensor::from_vec([1, 4, 1], deterministic_data(4, 13)).unwrap();
    let mut fb2 = Builder::new().unwrap();
    let fx2 = fb2.leaf([2, 4, 64], true).unwrap();
    let fg2 = fb2.leaf([1, 4, 1], true).unwrap();
    let ft2 = fb2.leaf([2, 4, 64], false).unwrap();
    let ln2 = fb2.layer_norm(fx2, 1, 1e-5).unwrap();
    let sm2 = fb2.softmax(ln2, 2).unwrap();
    let sc2 = fb2.binary(BinaryOp::Mul, sm2, fg2).unwrap();
    let df2 = fb2.binary(BinaryOp::Sub, sc2, ft2).unwrap();
    let sq2 = fb2.unary(UnaryOp::Square, df2).unwrap();
    let loss2 = fb2.reduce(ReduceOp::Mean, sq2, &[0, 1, 2], false).unwrap();
    let prog2 = fb2.finish(&[loss2]).unwrap();

    let gx2 = Shared::new(cuda.upload(&x_init).unwrap()).unwrap();
    let gg2 = Shared::new(cuda.upload(&g2_init).unwrap()).unwrap();
    let gt2 = Shared::new(cuda.upload(&target).unwrap()).unwrap();
    let mut gpu_fused_cap = cuda::Tape::new(
        &cuda,
        &prog2,
        &[
            Binding::new(fx2, gx2.clone()),
            Binding::new(fg2, gg2.clone()),
            Binding::new(ft2, gt2.clone()),
        ],
    )
    .unwrap();
    let cx2 = Shared::new(x_init).unwrap();
    let cg2 = Shared::new(g2_init).unwrap();
    let ct2 = Shared::new(target).unwrap();
    let mut cpu_fused = cpu::Tape::new(
        cpu,
        &prog2,
        &[
            Binding::new(fx2, cx2.clone()),
            Binding::new(fg2, cg2.clone()),
            Binding::new(ft2, ct2.clone()),
        ],
    )
    .unwrap();
    for _ in 0..4 {
        cpu_fused.step_scalar(&[fx2, fg2], 0.05).unwrap();
        gpu_fused_cap
            .step_scalar_captured(&[fx2, fg2], 0.05)
            .unwrap();
    }
    assert!(gpu_fused_cap.has_fused_plan());
    assert!(gpu_fused_cap.has_captured_graph());
    cuda.synchronize().unwrap();
    assert_close(
        cuda.download(&gx2.read().unwrap()).unwrap().data(),
        cx2.read().unwrap().data(),
        5e-5,
        "CUDA fused+captured post-softmax x",
    );
    assert_close(
        cuda.download(&gg2.read().unwrap()).unwrap().data(),
        cg2.read().unwrap().data(),
        5e-5,
        "CUDA fused+captured post-softmax gamma",
    );
}
