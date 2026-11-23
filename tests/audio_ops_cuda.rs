#![cfg(feature = "cuda")]

use fast_audio::{
    CrossEntropySpec, DropoutSpec, EmbeddingSpec, GradientMode, IndexTensor, Layout, ReduceOp,
    RopeSpec, Shape, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu,
    cuda::{
        CrossEntropyBackward, CrossEntropyPlan, Cuda, DropoutBackward, DropoutPlan,
        EmbeddingBackward, EmbeddingPlan, RopeBackward, RopePlan, Tape,
    },
};

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_err = 0.0f32;
    let mut idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let err = (x - y).abs();
        if err > max_err || err.is_nan() {
            max_err = err;
            idx = i;
        }
    }
    assert!(
        max_err <= tol,
        "{label}: max_err={max_err} > tol={tol} at idx={idx} ({} vs {})",
        a[idx],
        b[idx]
    );
}

#[test]
fn cuda_embedding_fp32_parity_and_determinism() {
    let cuda = Cuda::new(0).unwrap();
    let indices =
        IndexTensor::from_vec([4, 17], (0..68).map(|i| (i * 7 + 3) % 19).collect()).unwrap();
    let spec = EmbeddingSpec::new(19, 36, indices.shape(), Some(5)).unwrap();
    let table = Tensor::from_vec(
        [19, 36],
        (0..19 * 36)
            .map(|i| ((i as f32 * 0.13).sin()) * 0.8)
            .collect(),
    )
    .unwrap();
    let dy = Tensor::from_vec(
        [4, 17, 36],
        (0..68 * 36)
            .map(|i| ((i as f32 * 0.17).cos()) * 0.5)
            .collect(),
    )
    .unwrap();

    // CPU reference
    let mut out_cpu = Tensor::zeros([4, 17, 36]).unwrap();
    cpu::EmbeddingPlan::new(spec.clone())
        .run(&table, &indices, &mut out_cpu)
        .unwrap();
    let mut dw_cpu = Tensor::zeros([19, 36]).unwrap();
    cpu::EmbeddingBackward::new(spec.clone())
        .run(&indices, &dy, &mut dw_cpu, GradientMode::Overwrite)
        .unwrap();

    // CUDA
    let d_table = cuda.upload(&table).unwrap();
    let d_idx = cuda.upload_indices(&indices).unwrap();
    let mut d_out = cuda.zeros([4, 17, 36]).unwrap();
    EmbeddingPlan::new(&cuda, spec.clone())
        .run(&d_table, &d_idx, &mut d_out)
        .unwrap();
    let out_gpu = cuda.download(&d_out).unwrap();
    assert_eq!(out_gpu.data(), out_cpu.data());

    let d_dy = cuda.upload(&dy).unwrap();
    let mut d_dw = cuda.zeros([19, 36]).unwrap();
    let bwd = EmbeddingBackward::new(&cuda, spec);
    bwd.run(&d_idx, &d_dy, &mut d_dw, GradientMode::Overwrite)
        .unwrap();
    let dw_gpu1 = cuda.download(&d_dw).unwrap();
    assert_close(dw_gpu1.data(), dw_cpu.data(), 1e-5, "cuda embedding bwd");
    assert_eq!(&dw_gpu1.data()[5 * 36..6 * 36], &[0.0; 36]);

    // Bitwise determinism check
    bwd.run(&d_idx, &d_dy, &mut d_dw, GradientMode::Overwrite)
        .unwrap();
    let dw_gpu2 = cuda.download(&d_dw).unwrap();
    assert_eq!(dw_gpu1.data(), dw_gpu2.data());

    // Accumulate check
    bwd.run(&d_idx, &d_dy, &mut d_dw, GradientMode::Accumulate)
        .unwrap();
    let dw_gpu_acc = cuda.download(&d_dw).unwrap();
    let expected2: Vec<f32> = dw_gpu1.data().iter().map(|v| v * 2.0).collect();
    assert_close(
        dw_gpu_acc.data(),
        &expected2,
        1e-5,
        "cuda embedding bwd acc",
    );

    // Out-of-bounds index rejected before kernel launch
    let bad = IndexTensor::from_vec([4, 17], vec![19; 68]).unwrap();
    let d_bad = cuda.upload_indices(&bad).unwrap();
    assert!(
        EmbeddingPlan::new(
            &cuda,
            EmbeddingSpec::new(19, 36, indices.shape(), None).unwrap()
        )
        .run(&d_table, &d_bad, &mut d_out)
        .is_err()
    );
}

#[test]
fn cuda_rope_fp32_parity_and_tape() {
    let cuda = Cuda::new(0).unwrap();
    for &seq_axis in &[1usize, 2usize] {
        let dims = if seq_axis == 2 {
            [2, 4, 13, 20]
        } else {
            [2, 13, 4, 20]
        };
        let shape = Shape::new(dims).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let spec = RopeSpec::new(&shape, seq_axis, Some(12), 10_000.0, 5).unwrap();
        let x = Tensor::from_vec(
            dims,
            (0..shape.len())
                .map(|i| ((i as f32 * 0.29).sin()) * 1.1)
                .collect(),
        )
        .unwrap();
        let dy = Tensor::from_vec(
            dims,
            (0..shape.len())
                .map(|i| ((i as f32 * 0.19).cos()) * 0.7)
                .collect(),
        )
        .unwrap();

        let mut y_cpu = Tensor::zeros(dims).unwrap();
        cpu::RopePlan::new(&spec, &layout, &layout)
            .unwrap()
            .run(
                &x.view(&layout).unwrap(),
                &mut y_cpu.view_mut(&layout).unwrap(),
            )
            .unwrap();
        let mut dx_cpu = Tensor::zeros(dims).unwrap();
        cpu::RopeBackward::new(&spec, &layout, &layout)
            .unwrap()
            .run(
                &dy.view(&layout).unwrap(),
                &mut dx_cpu.view_mut(&layout).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();

        let d_x = cuda.upload(&x).unwrap();
        let d_dy = cuda.upload(&dy).unwrap();
        let mut d_y = cuda.zeros(dims).unwrap();
        let mut d_dx = cuda.zeros(dims).unwrap();
        RopePlan::new(&cuda, &spec, &layout, &layout)
            .unwrap()
            .run(
                &d_x.view(&layout).unwrap(),
                &mut d_y.view_mut(&layout).unwrap(),
            )
            .unwrap();
        RopeBackward::new(&cuda, &spec, &layout, &layout)
            .unwrap()
            .run(
                &d_dy.view(&layout).unwrap(),
                &mut d_dx.view_mut(&layout).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert_close(
            cuda.download(&d_y).unwrap().data(),
            y_cpu.data(),
            1e-5,
            "cuda rope fwd",
        );
        assert_close(
            cuda.download(&d_dx).unwrap().data(),
            dx_cpu.data(),
            1e-5,
            "cuda rope bwd",
        );

        // Autograd cuda::Tape check
        let mut b = Builder::new().unwrap();
        let lx = b.leaf(dims, true).unwrap();
        let r = b.rope(lx, seq_axis, Some(12), 10_000.0, 5).unwrap();
        let sq = b.unary(UnaryOp::Square, r).unwrap();
        let loss = b.reduce(ReduceOp::Sum, sq, &[0, 1, 2, 3], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();
        let owner = Shared::new(cuda.upload(&x).unwrap()).unwrap();
        let mut tape = Tape::new(&cuda, &prog, &[Binding::new(lx, owner)]).unwrap();
        tape.forward().unwrap();
        tape.backward_scalar(GradientMode::Overwrite).unwrap();
        let grad = cuda.download(tape.gradient(lx).unwrap().unwrap()).unwrap();
        let expected: Vec<f32> = x.data().iter().map(|v| 2.0 * v).collect();
        assert_close(grad.data(), &expected, 1e-4, "cuda rope tape grad");
    }
}

#[test]
fn cuda_cross_entropy_fp32_parity() {
    let cuda = Cuda::new(0).unwrap();
    for &ls in &[0.0f32, 0.1f32] {
        let n = 16;
        let v = 131;
        let spec = CrossEntropySpec::new(n, v, Some(-100), ls).unwrap();
        let logits = Tensor::from_vec(
            [n, v],
            (0..n * v)
                .map(|i| {
                    if i / v == 1 {
                        1e20
                    } else {
                        ((i as f32 * 0.31).sin()) * 2.3
                    }
                })
                .collect(),
        )
        .unwrap();
        let targets = IndexTensor::from_vec(
            [n],
            (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        -100
                    } else {
                        ((i * 11) % v) as i64
                    }
                })
                .collect(),
        )
        .unwrap();

        let mut loss_cpu = Tensor::zeros([]).unwrap();
        cpu::CrossEntropyPlan::new(spec)
            .run(&logits, &targets, &mut loss_cpu)
            .unwrap();
        let mut grad_cpu = Tensor::zeros([n, v]).unwrap();
        cpu::CrossEntropyBackward::new(spec)
            .run(
                &logits,
                &targets,
                1.5,
                &mut grad_cpu,
                GradientMode::Overwrite,
            )
            .unwrap();

        let d_logits = cuda.upload(&logits).unwrap();
        let d_targets = cuda.upload_indices(&targets).unwrap();
        let mut d_loss = cuda.zeros([]).unwrap();
        let mut d_grad = cuda.zeros([n, v]).unwrap();
        CrossEntropyPlan::new(&cuda, spec)
            .unwrap()
            .run(&d_logits, &d_targets, &mut d_loss)
            .unwrap();
        CrossEntropyBackward::new(&cuda, spec)
            .unwrap()
            .run(
                &d_logits,
                &d_targets,
                1.5,
                &mut d_grad,
                GradientMode::Overwrite,
            )
            .unwrap();

        let loss_gpu = cuda.download(&d_loss).unwrap().data()[0];
        assert!(
            (loss_gpu - loss_cpu.data()[0]).abs() < 1e-5,
            "ls={ls} {loss_gpu} vs {}",
            loss_cpu.data()[0]
        );
        assert_close(
            cuda.download(&d_grad).unwrap().data(),
            grad_cpu.data(),
            1e-5,
            "cuda ce bwd",
        );
    }
}

#[test]
fn cuda_dropout_philox_bit_identical_to_cpu() {
    let cuda = Cuda::new(0).unwrap();
    let dims = [33, 65];
    let shape = Shape::new(dims).unwrap();
    let layout = Layout::contiguous(dims).unwrap();
    let spec = DropoutSpec::new(&shape, 0.3, 0x1234_5678_9abc_def0).unwrap();
    let x = Tensor::from_vec(
        dims,
        (0..shape.len()).map(|i| (i as f32) * 0.01 - 1.0).collect(),
    )
    .unwrap();
    let dy = Tensor::from_vec(
        dims,
        (0..shape.len())
            .map(|i| ((i as f32 * 0.17).sin()) + 0.25)
            .collect(),
    )
    .unwrap();

    let mut y_cpu = Tensor::zeros(dims).unwrap();
    cpu::DropoutPlan::new(&spec, &layout, &layout)
        .unwrap()
        .run(
            &x.view(&layout).unwrap(),
            &mut y_cpu.view_mut(&layout).unwrap(),
            42,
        )
        .unwrap();
    let mut dx_cpu = Tensor::zeros(dims).unwrap();
    cpu::DropoutBackward::new(&spec, &layout, &layout)
        .unwrap()
        .run(
            &dy.view(&layout).unwrap(),
            &mut dx_cpu.view_mut(&layout).unwrap(),
            42,
            GradientMode::Overwrite,
        )
        .unwrap();

    let d_x = cuda.upload(&x).unwrap();
    let d_dy = cuda.upload(&dy).unwrap();
    let mut d_y = cuda.zeros(dims).unwrap();
    let mut d_dx = cuda.zeros(dims).unwrap();
    DropoutPlan::new(&cuda, spec.clone())
        .run(&d_x, &mut d_y, 42)
        .unwrap();
    DropoutBackward::new(&cuda, spec)
        .run(&d_dy, &mut d_dx, 42, GradientMode::Overwrite)
        .unwrap();

    let y_gpu = cuda.download(&d_y).unwrap();
    let dx_gpu = cuda.download(&d_dx).unwrap();
    // Exact zero/nonzero mask match bit-for-bit!
    for i in 0..shape.len() {
        assert_eq!(
            y_gpu.data()[i] == 0.0,
            y_cpu.data()[i] == 0.0,
            "mask mismatch at {i}"
        );
        assert_eq!(
            dx_gpu.data()[i] == 0.0,
            dx_cpu.data()[i] == 0.0,
            "bwd mask mismatch at {i}"
        );
    }
    assert_close(y_gpu.data(), y_cpu.data(), 1e-6, "cuda dropout fwd");
    assert_close(dx_gpu.data(), dx_cpu.data(), 1e-6, "cuda dropout bwd");
}
