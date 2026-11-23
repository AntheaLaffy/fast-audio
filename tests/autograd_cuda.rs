#![cfg(feature = "cuda")]
use fast_audio::{
    BinaryOp as B, Error, GradientMode as G, Layout, ReduceOp as R, Shared, Tensor, UnaryOp as U,
    autograd::{Binding, Builder, GradientStatus as S, PrepareOptions, Seed},
    cuda::{self, Cuda},
};
#[path = "support/autograd_audio.rs"]
mod audio;
#[allow(
    dead_code,
    reason = "Reuse the independent FP64 training control, not the matrix-only fixtures."
)]
#[path = "support/matmul.rs"]
mod dense;
#[allow(
    dead_code,
    reason = "The CPU backend naming helper is not used by CUDA tests."
)]
#[path = "support/autograd_export.rs"]
mod export;
#[path = "support/autograd.rs"]
mod fixtures;
#[allow(
    dead_code,
    reason = "Tape checks use independently specified inputs and references."
)]
mod support;

#[test]
fn cuda_tape_procedural_waveform_and_spectral_training() {
    let mut export = export::Exporter::new("audio-cuda");
    let cuda = Cuda::new(0).unwrap();
    for fixture in audio::fixtures() {
        let parameters = fixture
            .parameters
            .each_ref()
            .map(|t| Shared::new(cuda.upload(t).unwrap()).unwrap());
        let input_owner = Shared::new(cuda.upload(&fixture.input).unwrap()).unwrap();
        let target_owner = Shared::new(cuda.upload(&fixture.target).unwrap()).unwrap();
        let bindings = [
            Binding::new(fixture.leaves[0], input_owner.clone()),
            Binding::new(fixture.leaves[1], target_owner.clone()),
            Binding::new(fixture.leaves[2], parameters[0].clone()),
            Binding::new(fixture.leaves[3], parameters[1].clone()),
        ];
        let mut tape = cuda::Tape::new(&cuda, &fixture.program, &bindings).unwrap();
        let roots = fixture.program.roots();
        let scalar = cuda
            .upload(&Tensor::from_vec([], vec![1.]).unwrap())
            .unwrap();
        let zeros = cuda.zeros(fixture.target.shape().dims()).unwrap();
        let layouts = [
            Layout::contiguous([]).unwrap(),
            Layout::contiguous(fixture.target.shape().dims()).unwrap(),
        ];
        let seeds = [
            Seed::new(roots[0], &scalar, &layouts[0]),
            Seed::new(roots[1], &zeros, &layouts[1]),
        ];
        let mut control = fixture.initial();
        let initial_loss = fixture.reference(&control).loss;
        assert!(tape.resources().edge_scratch_bytes > 0);
        assert_eq!(tape.resources().view_index_bytes, 0);
        for step in 0..8 {
            let before = parameters
                .each_ref()
                .map(|p| cuda.download(&p.read().unwrap()).unwrap());
            let expected = fixture.reference(&control);
            support::no_allocations(|| {
                tape.forward().unwrap();
                tape.backward(&seeds, G::Overwrite).unwrap();
            });
            let loss = cuda
                .download(tape.output(roots[0]).unwrap().storage())
                .unwrap();
            let prediction = {
                let output = tape.output(roots[1]).unwrap();
                cuda.download(output.storage())
                    .unwrap()
                    .view(output.layout())
                    .unwrap()
                    .to_tensor()
                    .unwrap()
            };
            support::close(loss.data(), &[expected.loss], 2e-5, 2e-5);
            support::close(prediction.data(), &expected.prediction, 2e-5, 2e-5);
            let gradients = [fixture.leaves[2], fixture.leaves[3]]
                .map(|v| cuda.download(tape.gradient(v).unwrap().unwrap()).unwrap());
            for (actual, expected) in gradients.iter().zip(&expected.gradients) {
                support::close(actual.data(), expected, 2e-5, 2e-5);
            }
            assert_eq!(tape.gradient_status(fixture.leaves[0]).unwrap(), S::Unused);
            assert_eq!(tape.gradient_status(fixture.leaves[1]).unwrap(), S::Unused);
            let versions = parameters.each_ref().map(|p| p.version().unwrap());
            support::no_allocations(|| {
                tape.sgd(
                    &[fixture.leaves[2], fixture.leaves[3], fixture.leaves[2]],
                    0.05,
                )
                .unwrap()
            });
            for side in 0..2 {
                for (p, g) in control[side].iter_mut().zip(&expected.gradients[side]) {
                    *p -= 0.05f32 as f64 * g;
                }
                support::close(
                    cuda.download(&parameters[side].read().unwrap())
                        .unwrap()
                        .data(),
                    &control[side],
                    2e-5,
                    2e-5,
                );
                assert_eq!(
                    parameters[side].version().unwrap().generation(),
                    versions[side].generation() + 1
                );
            }
            assert_eq!(
                export::bits(&cuda.download(&input_owner.read().unwrap()).unwrap()),
                export::bits(&fixture.input)
            );
            assert_eq!(
                export::bits(&cuda.download(&target_owner.read().unwrap()).unwrap()),
                export::bits(&fixture.target)
            );
            export.record(serde_json::json!({"schema_version":1,"kind":"audio_sgd","backend":"cuda",
                "fixture":fixture.name,"step":step,"steps":8,"input":export::tensor(&fixture.input),"target":export::tensor(&fixture.target),
                "parameters":before.each_ref().map(export::tensor),"rate":0.05f32.to_bits(),"loss":export::tensor(&loss),
                "prediction":export::tensor(&prediction),"gradients":gradients.each_ref().map(export::tensor),
                "updated":parameters.each_ref().map(|p|export::tensor(&cuda.download(&p.read().unwrap()).unwrap()))}));
        }
        assert!(fixture.reference(&control).loss < initial_loss);
        tape.synchronize().unwrap();
    }
    export.finish();
}

#[test]
fn cuda_tape_all_local_operations_compose_with_views_and_broadcasts() {
    let mut export = export::Exporter::new("composition-cuda");
    let cuda = Cuda::new(0).unwrap();
    for (index, fixture) in fixtures::fixtures().into_iter().enumerate() {
        let owners = fixture
            .inputs
            .each_ref()
            .map(|t| Shared::new(cuda.upload(t).unwrap()).unwrap());
        let bindings = [
            Binding::new(fixture.leaves[0], owners[0].clone()),
            Binding::new(fixture.leaves[1], owners[1].clone()),
        ];
        let mut tape = cuda::Tape::new(&cuda, &fixture.program, &bindings).unwrap();
        let root = fixture.program.roots()[0];
        let seed = cuda.upload(&fixture.seed).unwrap();
        let layout = Layout::contiguous(fixture.seed.shape().dims()).unwrap();
        for binding in 0..2 {
            let inputs = fixture.changed_inputs(binding);
            let expected = fixture.reference(&inputs);
            for (owner, input) in owners.iter().zip(&inputs) {
                cuda.copy_from_host(input, &mut owner.write().unwrap())
                    .unwrap();
            }
            let seeds = [Seed::new(root, &seed, &layout)];
            support::no_allocations(|| {
                tape.forward().unwrap();
                tape.backward(&seeds, G::Overwrite).unwrap();
            });
            {
                let output = tape.output(root).unwrap();
                support::close(
                    cuda.download(output.storage()).unwrap().data(),
                    &expected.output,
                    2e-5,
                    2e-5,
                );
            }
            let first = fixture.leaves.map(|leaf| {
                cuda.download(tape.gradient(leaf).unwrap().unwrap())
                    .unwrap()
            });
            for (actual, expected) in first.iter().zip(&expected.gradients) {
                support::close(actual.data(), expected, 2e-5, 2e-5);
            }
            tape.backward(&seeds, G::Overwrite).unwrap();
            for (leaf, old) in fixture.leaves.iter().zip(&first) {
                assert_eq!(
                    dense::bits(
                        &cuda
                            .download(tape.gradient(*leaf).unwrap().unwrap())
                            .unwrap()
                    ),
                    dense::bits(old)
                );
            }
            tape.backward(&seeds, G::Accumulate).unwrap();
            for (leaf, expected) in fixture.leaves.iter().zip(&expected.gradients) {
                support::close(
                    cuda.download(tape.gradient(*leaf).unwrap().unwrap())
                        .unwrap()
                        .data(),
                    &expected.iter().map(|v| v * 2.).collect::<Vec<_>>(),
                    2e-5,
                    2e-5,
                );
            }
            for (owner, input) in owners.iter().zip(&inputs) {
                assert_eq!(
                    dense::bits(&cuda.download(&owner.read().unwrap()).unwrap()),
                    dense::bits(input)
                );
            }
            let output = cuda.download(tape.output(root).unwrap().storage()).unwrap();
            let accumulated = fixture.leaves.map(|leaf| {
                cuda.download(tape.gradient(leaf).unwrap().unwrap())
                    .unwrap()
            });
            export.composition(export::Composition {
                backend: "cuda",
                index,
                binding,
                inputs: &inputs,
                seed: &fixture.seed,
                output: &output,
                gradients: [&first, &accumulated],
            });
        }
    }
    export.finish();
}

#[test]
fn cuda_tape_dense_training_reuses_native_gradients_and_updates() {
    let mut export = export::Exporter::new("training-cuda");
    let cuda = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([3, 2], false).unwrap();
    let target = b.leaf([3, 2], false).unwrap();
    let w = b.leaf([2, 2], true).unwrap();
    let y = b.matmul(x, w).unwrap();
    let residual = b.binary(B::Sub, y, target).unwrap();
    let square = b.unary(U::Square, residual).unwrap();
    let loss = b.reduce(R::Mean, square, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let [xx, tt, ww, rate] = dense::training_fixture();
    let owner = Shared::new(cuda.upload(&ww).unwrap()).unwrap();
    let bindings = [
        Binding::new(x, Shared::new(cuda.upload(&xx).unwrap()).unwrap()),
        Binding::new(target, Shared::new(cuda.upload(&tt).unwrap()).unwrap()),
        Binding::new(w, owner.clone()),
    ];
    let mut tape = cuda::Tape::new(&cuda, &program, &bindings).unwrap();
    let mut control = ww.data().iter().map(|&v| v as f64).collect::<Vec<_>>();
    assert_eq!(tape.gradient_status(w).unwrap(), S::Uncomputed);
    assert!(tape.sgd(&[w], 0.05).is_err());
    for step in 0..8 {
        let before = cuda.download(&owner.read().unwrap()).unwrap();
        let (expected_loss, gradient) =
            dense::training_control(&mut control, &xx, &tt, rate.data()[0] as f64);
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward_scalar(G::Overwrite).unwrap();
        });
        {
            let output = tape.output(loss).unwrap();
            support::close(
                cuda.download(output.storage()).unwrap().data(),
                &[expected_loss],
                2e-5,
                2e-5,
            );
        }
        support::close(
            cuda.download(tape.gradient(w).unwrap().unwrap())
                .unwrap()
                .data(),
            &gradient,
            2e-5,
            2e-5,
        );
        assert_eq!(tape.gradient_status(x).unwrap(), S::Unused);
        let actual_loss = cuda.download(tape.output(loss).unwrap().storage()).unwrap();
        let actual_gradient = cuda.download(tape.gradient(w).unwrap().unwrap()).unwrap();
        let version = owner.version().unwrap();
        support::no_allocations(|| tape.sgd(&[w, w], rate.data()[0]).unwrap());
        assert_eq!(
            owner.version().unwrap().generation(),
            version.generation() + 1
        );
        support::close(
            cuda.download(&owner.read().unwrap()).unwrap().data(),
            &control,
            2e-5,
            2e-5,
        );
        assert!(tape.output(loss).is_err());
        assert!(tape.sgd(&[w], 0.05).is_err());
        export.record(serde_json::json!({"schema_version":1,"kind":"dense_sgd","backend":"cuda",
            "step":step,"steps":8,"input":export::tensor(&xx),"target":export::tensor(&tt),
            "parameters":[export::tensor(&before)],"rate":rate.data()[0].to_bits(),"loss":export::tensor(&actual_loss),
            "gradients":[export::tensor(&actual_gradient)],"updated":[export::tensor(&cuda.download(&owner.read().unwrap()).unwrap())]}));
    }
    tape.synchronize().unwrap();
    export.finish();
}

#[test]
fn cuda_tape_tied_overlap_detach_ancestor_and_duplicate_roots() {
    let mut export = export::Exporter::new("branches-cuda");
    let cuda = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([3], true).unwrap();
    let tied = b.leaf([3], true).unwrap();
    let c = b.leaf([3], false).unwrap();
    let v = b
        .view(x, &Layout::from_strides([2, 2], [1, 1], 0).unwrap())
        .unwrap();
    let q = b.unary(U::Square, v).unwrap();
    let sum = b.reduce(R::Sum, q, &[0, 1], false).unwrap();
    let prod = b.binary(B::Mul, x, tied).unwrap();
    let prod = b.reduce(R::Sum, prod, &[0], false).unwrap();
    let coeff = b.binary(B::Mul, x, c).unwrap();
    let coeff = b.reduce(R::Sum, coeff, &[0], false).unwrap();
    let detached = b.detach(v).unwrap();
    let z = b.binary(B::Mul, v, detached).unwrap();
    let z = b.reduce(R::Sum, z, &[0, 1], false).unwrap();
    let roots = [sum, sum, prod, coeff, z, q];
    let program = b.finish(&roots).unwrap();
    let owner = Shared::new(
        cuda.upload(&Tensor::from_vec([3], vec![1., 2., 3.]).unwrap())
            .unwrap(),
    )
    .unwrap();
    let bindings = [
        Binding::new(x, owner.clone()),
        Binding::new(tied, owner.clone()),
        Binding::new(c, owner.clone()),
    ];
    let scalar = Layout::contiguous([]).unwrap();
    let layouts = [
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar,
        Layout::from_strides([2, 2], [-3, 1], 3).unwrap(),
    ];
    let seeds_host = [
        Tensor::from_vec([], vec![0.5]).unwrap(),
        Tensor::from_vec([], vec![1.5]).unwrap(),
        Tensor::from_vec([], vec![1.]).unwrap(),
        Tensor::from_vec([], vec![1.]).unwrap(),
        Tensor::from_vec([], vec![1.]).unwrap(),
        Tensor::from_vec([5], vec![0.3, 0.4, f32::from_bits(0x7fc12345), 0.1, 0.2]).unwrap(),
    ];
    let seed_tensors = seeds_host.each_ref().map(|t| cuda.upload(t).unwrap());
    let seeds = roots
        .iter()
        .zip(&seed_tensors)
        .zip(&layouts)
        .map(|((&v, t), l)| Seed::new(v, t, l))
        .collect::<Vec<_>>();
    assert!(
        cuda::Tape::with_options(
            &cuda,
            &program,
            &bindings,
            PrepareOptions {
                seed_layouts: Some(&layouts),
                view_index_limit_bytes: 0
            }
        )
        .is_err()
    );
    let mut tape = cuda::Tape::with_options(
        &cuda,
        &program,
        &bindings,
        PrepareOptions {
            seed_layouts: Some(&layouts),
            ..PrepareOptions::default()
        },
    )
    .unwrap();
    support::no_allocations(|| {
        tape.forward().unwrap();
        tape.backward(&seeds, G::Overwrite).unwrap();
    });
    support::close(
        cuda.download(tape.gradient(x).unwrap().unwrap())
            .unwrap()
            .data(),
        &[8.2, 28., 26.4],
        2e-5,
        2e-5,
    );
    assert!(std::ptr::eq(
        tape.gradient(x).unwrap().unwrap(),
        tape.gradient(tied).unwrap().unwrap()
    ));
    assert_eq!(tape.gradient_status(c).unwrap(), S::Unused);
    let first = cuda.download(tape.gradient(x).unwrap().unwrap()).unwrap();
    let outputs = roots
        .map(|root| export::tensor(&cuda.download(tape.output(root).unwrap().storage()).unwrap()));
    tape.backward(&seeds, G::Accumulate).unwrap();
    support::close(
        cuda.download(tape.gradient(x).unwrap().unwrap())
            .unwrap()
            .data(),
        &[16.4, 56., 52.8],
        2e-5,
        2e-5,
    );
    let version = owner.version().unwrap();
    tape.sgd(&[x, tied, x], 0.01).unwrap();
    assert_eq!(
        owner.version().unwrap().generation(),
        version.generation() + 1
    );
    support::close(
        cuda.download(&owner.read().unwrap()).unwrap().data(),
        &[
            1. - 0.01f32 as f64 * 16.4,
            2. - 0.01f32 as f64 * 56.,
            3. - 0.01f32 as f64 * 52.8,
        ],
        2e-5,
        2e-5,
    );
    assert_eq!(
        dense::bits(&cuda.download(&seed_tensors[5]).unwrap()),
        dense::bits(&seeds_host[5])
    );
    export.record(serde_json::json!({"schema_version":1,"kind":"branches","backend":"cuda",
        "input":export::tensor(&Tensor::from_vec([3],vec![1.,2.,3.]).unwrap()),"outputs":outputs,
        "gradient":export::tensor(&first),"accumulated":export::tensor(&cuda.download(tape.gradient(x).unwrap().unwrap()).unwrap()),
        "updated":export::tensor(&cuda.download(&owner.read().unwrap()).unwrap()),"rate":0.01f32.to_bits()}));
    export.finish();
}

#[test]
fn cuda_tape_rejects_foreign_sessions_and_stale_or_borrowed_parameters() {
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let unused = b.leaf([0], false).unwrap();
    let y = b.unary(U::Square, x).unwrap();
    let program = b.finish(&[y]).unwrap();
    let host = Tensor::from_vec([2], vec![1., 2.]).unwrap();
    let owner = Shared::new(cuda.upload(&host).unwrap()).unwrap();
    assert!(
        cuda::Tape::new(
            &cuda,
            &program,
            &[
                Binding::new(x, owner.clone()),
                Binding::new(unused, Shared::new(foreign.zeros([0]).unwrap()).unwrap())
            ]
        )
        .is_err()
    );
    let bindings = [
        Binding::new(x, owner.clone()),
        Binding::new(unused, Shared::new(cuda.zeros([0]).unwrap()).unwrap()),
    ];
    let mut tape = cuda::Tape::new(&cuda, &program, &bindings).unwrap();
    let layout = Layout::contiguous([2]).unwrap();
    let seed = cuda
        .upload(&Tensor::from_vec([2], vec![1., 1.]).unwrap())
        .unwrap();
    let bad_seed = foreign.zeros([2]).unwrap();
    tape.forward().unwrap();
    assert!(
        tape.backward(&[Seed::new(y, &bad_seed, &layout)], G::Overwrite)
            .is_err()
    );
    assert_eq!(tape.gradient_status(x).unwrap(), S::Uncomputed);
    tape.backward(&[Seed::new(y, &seed, &layout)], G::Overwrite)
        .unwrap();
    let before = owner.version().unwrap();
    let read = owner.read().unwrap();
    assert!(matches!(
        tape.sgd(&[x], 0.1),
        Err(Error::StorageBorrowConflict)
    ));
    assert_eq!(owner.version().unwrap(), before);
    drop(read);
    cuda.copy_from_host(
        &Tensor::from_vec([2], vec![3., 4.]).unwrap(),
        &mut owner.write().unwrap(),
    )
    .unwrap();
    assert!(matches!(
        tape.backward(&[Seed::new(y, &seed, &layout)], G::Overwrite),
        Err(Error::StorageModified { .. })
    ));
    assert!(matches!(
        tape.sgd(&[x], 0.1),
        Err(Error::StorageModified { .. })
    ));
    assert_eq!(
        cuda.download(tape.gradient(x).unwrap().unwrap())
            .unwrap()
            .data(),
        &[2., 4.]
    );
    tape.forward().unwrap();
    assert!(tape.sgd(&[x], 0.1).is_err());
    tape.backward(&[Seed::new(y, &seed, &layout)], G::Overwrite)
        .unwrap();
    assert_eq!(
        cuda.download(tape.gradient(x).unwrap().unwrap())
            .unwrap()
            .data(),
        &[6., 8.]
    );
    *owner.write().unwrap() = foreign.zeros([2]).unwrap();
    assert!(tape.forward().is_err());
    assert!(tape.output(y).is_err());
}

#[test]
fn cuda_tape_sgd_zeroing_empty_storage_and_grid_reuse() {
    let cuda = Cuda::new(0).unwrap();
    for n in [0, 65535 * 256 + 17] {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([n], true).unwrap();
        let program = b.finish(&[x]).unwrap();
        let owner = Shared::new(cuda.zeros([n]).unwrap()).unwrap();
        let mut tape = cuda::Tape::new(&cuda, &program, &[Binding::new(x, owner.clone())]).unwrap();
        let seed_host = Tensor::from_vec([n], vec![1.; n]).unwrap();
        let seed = cuda.upload(&seed_host).unwrap();
        let layout = Layout::contiguous([n]).unwrap();
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward(&[Seed::new(x, &seed, &layout)], G::Overwrite)
                .unwrap();
            tape.sgd(&[x, x], 0.25).unwrap();
        });
        assert!(
            cuda.download(&owner.read().unwrap())
                .unwrap()
                .data()
                .iter()
                .all(|&v| v.to_bits() == (-0.25f32).to_bits())
        );
        support::no_allocations(|| tape.zero_grad().unwrap());
        assert!(
            cuda.download(tape.gradient(x).unwrap().unwrap())
                .unwrap()
                .data()
                .iter()
                .all(|&v| v.to_bits() == 0)
        );
        assert!(tape.sgd(&[x], 0.25).is_err());
        assert_eq!(tape.resources().activation_bytes, 0);
        assert_eq!(tape.resources().edge_scratch_bytes, 0);
    }
}

#[test]
fn cuda_tape_zero_contribution_edges_preserve_root_bits_without_view_indices() {
    let cuda = Cuda::new(0).unwrap();
    for matrix in [false, true] {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([4], true).unwrap();
        let v = b
            .view(x, &Layout::from_strides([2, 3], [1, 1], 0).unwrap())
            .unwrap();
        let empty = b
            .leaf(if matrix { vec![3, 0] } else { vec![0, 2, 3] }, false)
            .unwrap();
        let y = if matrix {
            b.matmul(v, empty).unwrap()
        } else {
            b.binary(B::Add, v, empty).unwrap()
        };
        let program = b.finish(&[x, y]).unwrap();
        let owner = Shared::new(cuda.zeros([4]).unwrap()).unwrap();
        let zero = Shared::new(cuda.zeros(program.shape(empty).unwrap().dims()).unwrap()).unwrap();
        let mut tape = cuda::Tape::with_options(
            &cuda,
            &program,
            &[Binding::new(x, owner), Binding::new(empty, zero)],
            PrepareOptions {
                view_index_limit_bytes: 0,
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        assert_eq!(tape.resources().edge_scratch_bytes, 0);
        assert_eq!(tape.resources().view_index_bytes, 0);
        let bits = [0x80000000, 0x7fc54321, 0x80000000, 0x7fc12345];
        let seed_host =
            Tensor::from_vec([4], bits.iter().map(|&v| f32::from_bits(v)).collect()).unwrap();
        let seed = cuda.upload(&seed_host).unwrap();
        let empty_seed = cuda.zeros(program.shape(y).unwrap().dims()).unwrap();
        let first = Layout::contiguous([4]).unwrap();
        let second = Layout::contiguous(program.shape(y).unwrap().dims()).unwrap();
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward(
                &[
                    Seed::new(x, &seed, &first),
                    Seed::new(y, &empty_seed, &second),
                ],
                G::Overwrite,
            )
            .unwrap();
        });
        assert_eq!(
            dense::bits(&cuda.download(tape.gradient(x).unwrap().unwrap()).unwrap()),
            bits
        );
    }
}

#[test]
fn cuda_tape_strided_matrix_both_operand_gradients_and_independent_instances() {
    let cuda = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 3], true).unwrap();
    let reversed = b.slice(x, 1, 2, 3, -1).unwrap();
    let a = b.permute(reversed, &[1, 0]).unwrap();
    let gram = b.matmul(a, reversed).unwrap();
    let program = b.finish(&[gram]).unwrap();
    let initial = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let owner = Shared::new(cuda.upload(&initial).unwrap()).unwrap();
    let bindings = [Binding::new(x, owner.clone())];
    let mut first = cuda::Tape::new(&cuda, &program, &bindings).unwrap();
    let mut second = cuda::Tape::new(&cuda, &program, &bindings).unwrap();
    assert_eq!(first.resources().max_static_shared_bytes_per_block, 2048);
    assert_eq!(first.resources().edge_scratch_bytes, 48);
    let seed = cuda
        .upload(&Tensor::from_vec([3, 3], (1..=9).map(|v| v as f32).collect()).unwrap())
        .unwrap();
    let layout = Layout::contiguous([3, 3]).unwrap();
    let seeds = [Seed::new(gram, &seed, &layout)];
    for tape in [&mut first, &mut second] {
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward(&seeds, G::Overwrite).unwrap();
        });
        assert_eq!(
            cuda.download(tape.output(gram).unwrap().storage())
                .unwrap()
                .data(),
            &[45., 36., 27., 36., 29., 22., 27., 22., 17.]
        );
        assert_eq!(
            cuda.download(tape.gradient(x).unwrap().unwrap())
                .unwrap()
                .data(),
            &[76., 52., 28., 202., 142., 82.]
        );
    }
    let expected = initial
        .data()
        .iter()
        .zip([76., 52., 28., 202., 142., 82.])
        .map(|(&p, g)| p - 0.01 * g)
        .collect::<Vec<f32>>();
    first.sgd(&[x], 0.01).unwrap();
    assert!(matches!(
        second.sgd(&[x], 0.01),
        Err(Error::StorageModified { .. })
    ));
    support::close(
        cuda.download(&owner.read().unwrap()).unwrap().data(),
        &expected.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        2e-5,
        2e-5,
    );
    second.forward().unwrap();
    second.backward(&seeds, G::Overwrite).unwrap();
    let weights = [[18., 14., 10.], [14., 10., 6.], [10., 6., 2.]];
    let gradient = (0..2)
        .flat_map(|r| {
            let values = &expected;
            let weights = &weights;
            (0..3).map(move |c| {
                (0..3)
                    .map(|k| values[r * 3 + k] as f64 * weights[k][c])
                    .sum::<f64>()
            })
        })
        .collect::<Vec<_>>();
    support::close(
        cuda.download(second.gradient(x).unwrap().unwrap())
            .unwrap()
            .data(),
        &gradient,
        2e-5,
        2e-5,
    );
}

#[test]
fn cuda_tape_captured_step_matches_uncaptured_bits_and_enforces_safety() {
    let cuda = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([3, 4], false).unwrap();
    let w = b.leaf([4, 2], true).unwrap();
    let bias = b.leaf([1, 2], true).unwrap();
    let target = b.leaf([3, 2], false).unwrap();
    let mm = b.matmul(x, w).unwrap();
    let z = b.binary(B::Add, mm, bias).unwrap();
    let pred = b.unary(U::Tanh, z).unwrap();
    let diff = b.binary(B::Sub, pred, target).unwrap();
    let sq = b.unary(U::Square, diff).unwrap();
    let loss = b.reduce(R::Mean, sq, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let x_host =
        Tensor::from_vec([3, 4], (0..12).map(|i| (i as f32 - 5.5) / 8.0).collect()).unwrap();
    let w_host =
        Tensor::from_vec([4, 2], (0..8).map(|i| (i as f32 - 3.5) / 16.0).collect()).unwrap();
    let bias_host = Tensor::from_vec([1, 2], vec![0.125, -0.25]).unwrap();
    let target_host =
        Tensor::from_vec([3, 2], (0..6).map(|i| (i as f32 - 2.0) / 4.0).collect()).unwrap();

    let mk_owners = || {
        (
            Shared::new(cuda.upload(&x_host).unwrap()).unwrap(),
            Shared::new(cuda.upload(&w_host).unwrap()).unwrap(),
            Shared::new(cuda.upload(&bias_host).unwrap()).unwrap(),
            Shared::new(cuda.upload(&target_host).unwrap()).unwrap(),
        )
    };
    let (rx, rw, rb, rt) = mk_owners();
    let (ux, uw, ub, ut) = mk_owners();
    let (cx, cw, cb, ct) = mk_owners();

    let ref_bindings = [
        Binding::new(x, rx.clone()),
        Binding::new(w, rw.clone()),
        Binding::new(bias, rb.clone()),
        Binding::new(target, rt.clone()),
    ];
    let uncap_bindings = [
        Binding::new(x, ux.clone()),
        Binding::new(w, uw.clone()),
        Binding::new(bias, ub.clone()),
        Binding::new(target, ut.clone()),
    ];
    let cap_bindings = [
        Binding::new(x, cx.clone()),
        Binding::new(w, cw.clone()),
        Binding::new(bias, cb.clone()),
        Binding::new(target, ct.clone()),
    ];
    let mut ref_tape = cuda::Tape::new(&cuda, &program, &ref_bindings).unwrap();
    let mut uncap_tape = cuda::Tape::new(&cuda, &program, &uncap_bindings).unwrap();
    let mut cap_tape = cuda::Tape::new(&cuda, &program, &cap_bindings).unwrap();
    let params = [w, bias];
    let rate = 0.0625_f32;

    assert!(!cap_tape.has_captured_graph());
    for step in 0..8 {
        ref_tape.forward().unwrap();
        ref_tape.backward_scalar(G::Overwrite).unwrap();
        ref_tape.sgd(&params, rate).unwrap();

        if step >= 2 {
            support::no_allocations(|| {
                uncap_tape.step_scalar(&params, rate).unwrap();
                cap_tape.step_scalar_captured(&params, rate).unwrap();
            });
        } else {
            uncap_tape.step_scalar(&params, rate).unwrap();
            cap_tape.step_scalar_captured(&params, rate).unwrap();
        }
        assert!(cap_tape.has_fused_plan());
        assert!(uncap_tape.has_fused_plan());
        assert_eq!(cap_tape.has_captured_graph(), step >= 1);

        let ref_w = cuda.download(&rw.read().unwrap()).unwrap();
        let uncap_w = cuda.download(&uw.read().unwrap()).unwrap();
        let cap_w = cuda.download(&cw.read().unwrap()).unwrap();
        assert_eq!(
            dense::bits(&uncap_w),
            dense::bits(&cap_w),
            "captured vs uncaptured fused w mismatch at step {step}"
        );
        support::close(
            cap_w.data(),
            &ref_w.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
            2e-5,
            2e-5,
        );

        let ref_b = cuda.download(&rb.read().unwrap()).unwrap();
        let uncap_b = cuda.download(&ub.read().unwrap()).unwrap();
        let cap_b = cuda.download(&cb.read().unwrap()).unwrap();
        assert_eq!(
            dense::bits(&uncap_b),
            dense::bits(&cap_b),
            "captured vs uncaptured fused b mismatch at step {step}"
        );
        support::close(
            cap_b.data(),
            &ref_b.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
            2e-5,
            2e-5,
        );

        let uncap_gw = cuda
            .download(uncap_tape.gradient(w).unwrap().unwrap())
            .unwrap();
        let cap_gw = cuda
            .download(cap_tape.gradient(w).unwrap().unwrap())
            .unwrap();
        assert_eq!(
            dense::bits(&uncap_gw),
            dense::bits(&cap_gw),
            "gw mismatch at step {step}"
        );

        let uncap_gb = cuda
            .download(uncap_tape.gradient(bias).unwrap().unwrap())
            .unwrap();
        let cap_gb = cuda
            .download(cap_tape.gradient(bias).unwrap().unwrap())
            .unwrap();
        assert_eq!(
            dense::bits(&uncap_gb),
            dense::bits(&cap_gb),
            "gb mismatch at step {step}"
        );
    }

    // External SavedValue version must be invalidated by captured replay.
    let w_layout = std::sync::Arc::new(Layout::contiguous([4, 2]).unwrap());
    let saved_w = cw.save(&w_layout).unwrap();
    let v_before = cw.version().unwrap();
    support::no_allocations(|| {
        cap_tape.step_scalar_captured(&params, rate).unwrap();
    });
    assert_eq!(
        cw.version().unwrap().generation(),
        v_before.generation() + 1
    );
    assert!(matches!(saved_w.read(), Err(Error::StorageModified { .. })));

    // Concurrent read lock on an updated parameter must reject replay before launch.
    let held_read = cw.read().unwrap();
    assert!(matches!(
        cap_tape.step_scalar_captured(&params, rate),
        Err(Error::StorageBorrowConflict)
    ));
    drop(held_read);

    // Rate change invalidates old capture and re-captures cleanly with bit-exact result.
    let new_rate = 0.03125_f32;
    uncap_tape.step_scalar(&params, rate).unwrap();
    uncap_tape.step_scalar(&params, new_rate).unwrap();
    cap_tape.step_scalar_captured(&params, new_rate).unwrap();
    assert!(cap_tape.has_captured_graph());
    assert_eq!(
        dense::bits(&cuda.download(&uw.read().unwrap()).unwrap()),
        dense::bits(&cuda.download(&cw.read().unwrap()).unwrap())
    );

    // Explicit invalidate_capture drops the graph cleanly; next call re-captures.
    cap_tape.invalidate_capture();
    assert!(!cap_tape.has_captured_graph());
    uncap_tape.step_scalar(&params, new_rate).unwrap();
    cap_tape.step_scalar_captured(&params, new_rate).unwrap();
    assert!(cap_tape.has_captured_graph());
    assert_eq!(
        dense::bits(&cuda.download(&uw.read().unwrap()).unwrap()),
        dense::bits(&cuda.download(&cw.read().unwrap()).unwrap())
    );
}

#[test]
fn cuda_tape_fused_audio_waveform_and_spectrum_topologies_match_unfused() {
    let cuda = Cuda::new(0).unwrap();
    // Covers SingleBlockAll (small), SplitBlocksPerGroup (wave large), and WarpPerGroup (spectrum large)!
    for &(family, reversed, large) in &[
        ("waveform", false, false),
        ("waveform", true, false),
        ("waveform", true, true),
        ("spectrum", false, false),
        ("spectrum", true, false),
        ("spectrum", true, true),
    ] {
        let mut b = Builder::new().unwrap();
        let (x_dims, tgt_dims, g_dims, b_dims, g_view_dims, b_view_dims) = if family == "waveform" {
            let (bs, c, t) = if large { (2, 4, 1025) } else { (2, 2, 19) };
            (
                vec![bs, c, t],
                if reversed {
                    vec![bs, t, c]
                } else {
                    vec![bs, c, t]
                },
                vec![c],
                vec![c],
                vec![1, c, 1],
                vec![1, c, 1],
            )
        } else {
            let (bs, c, f, t) = if large { (2, 4, 33, 21) } else { (2, 2, 5, 7) };
            (
                vec![bs, c, f, t, 2],
                if reversed {
                    vec![bs, t, c, f]
                } else {
                    vec![bs, c, f, t]
                },
                vec![c, f],
                vec![2],
                vec![1, c, f, 1, 1],
                vec![2],
            )
        };
        let x = b.leaf(&x_dims, false).unwrap();
        let tgt = b.leaf(&tgt_dims, false).unwrap();
        let gain = b.leaf(&g_dims, true).unwrap();
        let bias = b.leaf(&b_dims, true).unwrap();
        let gv = b.reshape(gain, g_view_dims).unwrap();
        let bv = if family == "waveform" {
            b.reshape(bias, b_view_dims).unwrap()
        } else {
            bias
        };
        let scaled = b.binary(B::Mul, x, gv).unwrap();
        let shifted = b.binary(B::Add, scaled, bv).unwrap();
        let mut pred = if family == "waveform" {
            b.unary(U::Tanh, shifted).unwrap()
        } else {
            let sq = b.unary(U::Square, shifted).unwrap();
            b.reduce(R::Sum, sq, &[4], false).unwrap()
        };
        if reversed {
            let axis = if family == "waveform" { 2 } else { 3 };
            let t = x_dims[axis];
            pred = b.slice(pred, axis, t - 1, t, -1).unwrap();
            pred = b
                .permute(pred, if axis == 2 { &[0, 2, 1] } else { &[0, 3, 1, 2] })
                .unwrap();
        }
        let res = b.binary(B::Sub, pred, tgt).unwrap();
        let sq = b.unary(U::Square, res).unwrap();
        let axes: Vec<usize> = (0..tgt_dims.len()).collect();
        let loss = b.reduce(R::Mean, sq, &axes, false).unwrap();
        let detached = b.detach(pred).unwrap();
        let program = b.finish(&[loss, detached]).unwrap();

        let mk_tensor = |dims: &[usize], salt: usize| {
            let len: usize = dims.iter().product();
            Tensor::from_vec(
                dims,
                (0..len)
                    .map(|i| (((i * (7 + salt) + 3) % 29) as f32 - 14.0) / 32.0)
                    .collect(),
            )
            .unwrap()
        };
        let x_t = mk_tensor(&x_dims, 0);
        let tgt_t = mk_tensor(&tgt_dims, 1);
        let g_t = mk_tensor(&g_dims, 2);
        let b_t = mk_tensor(&b_dims, 3);
        let mk = || {
            (
                Shared::new(cuda.upload(&x_t).unwrap()).unwrap(),
                Shared::new(cuda.upload(&tgt_t).unwrap()).unwrap(),
                Shared::new(cuda.upload(&g_t).unwrap()).unwrap(),
                Shared::new(cuda.upload(&b_t).unwrap()).unwrap(),
            )
        };
        let (rx, rt, rg, rb) = mk();
        let (cx, ct, cg, cb) = mk();
        let seed_layouts = [
            Layout::contiguous([]).unwrap(),
            Layout::from_strides(&tgt_dims, vec![0; tgt_dims.len()], 0).unwrap(),
        ];
        let mut ref_tape = cuda::Tape::with_options(
            &cuda,
            &program,
            &[
                Binding::new(x, rx),
                Binding::new(tgt, rt),
                Binding::new(gain, rg.clone()),
                Binding::new(bias, rb.clone()),
            ],
            PrepareOptions {
                seed_layouts: Some(&seed_layouts),
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        let mut cap_tape = cuda::Tape::with_options(
            &cuda,
            &program,
            &[
                Binding::new(x, cx),
                Binding::new(tgt, ct),
                Binding::new(gain, cg.clone()),
                Binding::new(bias, cb.clone()),
            ],
            PrepareOptions {
                seed_layouts: Some(&seed_layouts),
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        let one = cuda
            .upload(&Tensor::from_vec([], vec![1.0]).unwrap())
            .unwrap();
        let zero = cuda.zeros([]).unwrap();
        let seeds = [
            Seed::new(loss, &one, &seed_layouts[0]),
            Seed::new(detached, &zero, &seed_layouts[1]),
        ];
        let params = [gain, bias];
        for step in 0..6 {
            ref_tape.forward().unwrap();
            ref_tape.backward(&seeds, G::Overwrite).unwrap();
            ref_tape.sgd(&params, 0.05).unwrap();
            if step >= 2 {
                support::no_allocations(|| {
                    cap_tape.step_captured(&seeds, &params, 0.05).unwrap();
                });
            } else {
                cap_tape.step_captured(&seeds, &params, 0.05).unwrap();
            }
            assert!(
                cap_tape.has_fused_plan(),
                "expected NVRTC fusion for {family} reversed={reversed} large={large}"
            );
            assert_eq!(cap_tape.has_captured_graph(), step >= 1);
            for (r_owner, c_owner, p_val) in [(&rg, &cg, gain), (&rb, &cb, bias)] {
                let rp = cuda.download(&r_owner.read().unwrap()).unwrap();
                let cp = cuda.download(&c_owner.read().unwrap()).unwrap();
                support::close(
                    cp.data(),
                    &rp.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    2e-5,
                    2e-5,
                );
                let rgrad = cuda
                    .download(ref_tape.gradient(p_val).unwrap().unwrap())
                    .unwrap();
                let cgrad = cuda
                    .download(cap_tape.gradient(p_val).unwrap().unwrap())
                    .unwrap();
                support::close(
                    cgrad.data(),
                    &rgrad.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    2e-5,
                    2e-5,
                );
            }
        }
    }
}
