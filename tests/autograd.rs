use fast_audio::{
    BinaryOp as B, Error, GradientMode as G, Layout, ReduceOp as R, Shared, Tensor, UnaryOp as U,
    autograd::{Binding, Builder, GradientStatus as S, PrepareOptions, Seed},
    cpu::{self, Cpu},
};
#[path = "support/autograd_audio.rs"]
mod audio;
#[path = "support/autograd_export.rs"]
mod export;
#[path = "support/autograd.rs"]
mod fixtures;
#[allow(
    dead_code,
    reason = "Tape fixtures specify their own independent numerical inputs."
)]
mod support;

fn values(t: &Tensor) -> Vec<f64> {
    t.data().iter().map(|&v| v as f64).collect()
}

#[test]
fn cpu_tape_all_local_operations_compose_with_views_and_broadcasts() {
    let mut export = export::Exporter::new("composition-cpu");
    for cpu in support::backends() {
        for (index, fixture) in fixtures::fixtures().into_iter().enumerate() {
            let owners = fixture
                .inputs
                .each_ref()
                .map(|t| Shared::new(t.clone()).unwrap());
            let bindings = [
                Binding::new(fixture.leaves[0], owners[0].clone()),
                Binding::new(fixture.leaves[1], owners[1].clone()),
            ];
            let mut tape = cpu::Tape::new(cpu, &fixture.program, &bindings).unwrap();
            let root = fixture.program.roots()[0];
            let layout = Layout::contiguous(fixture.seed.shape().dims()).unwrap();
            for binding in 0..2 {
                let inputs = fixture.changed_inputs(binding);
                let expected = fixture.reference(&inputs);
                for (owner, input) in owners.iter().zip(&inputs) {
                    owner
                        .write()
                        .unwrap()
                        .data_mut()
                        .copy_from_slice(input.data());
                }
                let seeds = [Seed::new(root, &fixture.seed, &layout)];
                support::no_allocations(|| {
                    tape.forward().unwrap();
                    tape.backward(&seeds, G::Overwrite).unwrap();
                });
                close(
                    tape.output(root).unwrap().storage().data(),
                    &expected.output,
                );
                let output = tape.output(root).unwrap().storage().clone();
                let first = fixture
                    .leaves
                    .map(|leaf| tape.gradient(leaf).unwrap().unwrap().clone());
                for (leaf, gradient) in fixture.leaves.iter().zip(&expected.gradients) {
                    close(tape.gradient(*leaf).unwrap().unwrap().data(), gradient);
                }
                tape.backward(&seeds, G::Overwrite).unwrap();
                for (leaf, gradient) in fixture.leaves.iter().zip(&first) {
                    assert_eq!(
                        tape.gradient(*leaf).unwrap().unwrap().data(),
                        gradient.data()
                    );
                }
                tape.backward(&seeds, G::Accumulate).unwrap();
                for (leaf, gradient) in fixture.leaves.iter().zip(&expected.gradients) {
                    close(
                        tape.gradient(*leaf).unwrap().unwrap().data(),
                        &gradient.iter().map(|v| v * 2.).collect::<Vec<_>>(),
                    );
                }
                for (owner, input) in owners.iter().zip(&inputs) {
                    assert_eq!(owner.read().unwrap().data(), input.data());
                }
                let accumulated = fixture
                    .leaves
                    .map(|leaf| tape.gradient(leaf).unwrap().unwrap().clone());
                export.composition(export::Composition {
                    backend: export::backend(cpu),
                    index,
                    binding,
                    inputs: &inputs,
                    seed: &fixture.seed,
                    output: &output,
                    gradients: [&first, &accumulated],
                });
            }
        }
    }
    export.finish();
}
fn close(actual: &[f32], expected: &[f64]) {
    support::close(actual, expected, 2e-5, 2e-5);
}

#[test]
fn cpu_tape_procedural_waveform_and_spectral_training() {
    let mut export = export::Exporter::new("audio-cpu");
    for cpu in support::backends() {
        for fixture in audio::fixtures() {
            let parameters = fixture
                .parameters
                .each_ref()
                .map(|t| Shared::new(t.clone()).unwrap());
            let input_owner = Shared::new(fixture.input.clone()).unwrap();
            let target_owner = Shared::new(fixture.target.clone()).unwrap();
            let bindings = [
                Binding::new(fixture.leaves[0], input_owner.clone()),
                Binding::new(fixture.leaves[1], target_owner.clone()),
                Binding::new(fixture.leaves[2], parameters[0].clone()),
                Binding::new(fixture.leaves[3], parameters[1].clone()),
            ];
            let mut tape = cpu::Tape::new(cpu, &fixture.program, &bindings).unwrap();
            let roots = fixture.program.roots();
            let scalar = Tensor::from_vec([], vec![1.]).unwrap();
            let zeros = Tensor::zeros(fixture.target.shape().dims()).unwrap();
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
                let before = parameters.each_ref().map(|p| p.read().unwrap().clone());
                let expected = fixture.reference(&control);
                support::no_allocations(|| {
                    tape.forward().unwrap();
                    tape.backward(&seeds, G::Overwrite).unwrap();
                });
                let loss = tape.output(roots[0]).unwrap().storage().clone();
                // The prediction is a signed/permuted alias, so read its logical order.
                let prediction = tape
                    .output(roots[1])
                    .unwrap()
                    .view()
                    .unwrap()
                    .to_tensor()
                    .unwrap();
                close(loss.data(), &[expected.loss]);
                close(prediction.data(), &expected.prediction);
                let gradients = [fixture.leaves[2], fixture.leaves[3]]
                    .map(|v| tape.gradient(v).unwrap().unwrap().clone());
                for (actual, expected) in gradients.iter().zip(&expected.gradients) {
                    close(actual.data(), expected);
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
                    close(parameters[side].read().unwrap().data(), &control[side]);
                    assert_eq!(
                        parameters[side].version().unwrap().generation(),
                        versions[side].generation() + 1
                    );
                }
                assert_eq!(
                    export::bits(&input_owner.read().unwrap()),
                    export::bits(&fixture.input)
                );
                assert_eq!(
                    export::bits(&target_owner.read().unwrap()),
                    export::bits(&fixture.target)
                );
                export.record(serde_json::json!({"schema_version":1,"kind":"audio_sgd","backend":export::backend(cpu),
                    "fixture":fixture.name,"step":step,"steps":8,"input":export::tensor(&fixture.input),"target":export::tensor(&fixture.target),
                    "parameters":before.each_ref().map(export::tensor),"rate":0.05f32.to_bits(),"loss":export::tensor(&loss),
                    "prediction":export::tensor(&prediction),"gradients":gradients.each_ref().map(export::tensor),
                    "updated":parameters.each_ref().map(|p|export::tensor(&p.read().unwrap()))}));
            }
            assert!(fixture.reference(&control).loss < initial_loss);
        }
    }
    export.finish();
}

#[test]
fn cpu_tape_dense_training_reuses_native_gradients_and_updates() {
    let mut export = export::Exporter::new("training-cpu");
    for cpu in support::backends() {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([3, 2], false).unwrap();
        let target = b.leaf([3, 2], false).unwrap();
        let weights = b.leaf([2, 2], true).unwrap();
        let product = b.matmul(x, weights).unwrap();
        let residual = b.binary(B::Sub, product, target).unwrap();
        let square = b.unary(U::Square, residual).unwrap();
        let loss = b.reduce(R::Mean, square, &[0, 1], false).unwrap();
        let program = b.finish(&[loss]).unwrap();
        let xx = Tensor::from_vec([3, 2], vec![0.2, -0.3, 0.4, 0.1, -0.2, 0.5]).unwrap();
        let tt = Tensor::from_vec([3, 2], vec![0.05, -0.1, 0.2, 0.15, -0.05, 0.25]).unwrap();
        let owner =
            Shared::new(Tensor::from_vec([2, 2], vec![0.1, -0.2, 0.3, 0.4]).unwrap()).unwrap();
        let mut control = values(&owner.read().unwrap());
        let bindings = [
            Binding::new(x, Shared::new(xx.clone()).unwrap()),
            Binding::new(target, Shared::new(tt.clone()).unwrap()),
            Binding::new(weights, owner.clone()),
        ];
        let mut tape = cpu::Tape::new(cpu, &program, &bindings).unwrap();
        assert_eq!(tape.gradient_status(weights).unwrap(), S::Uncomputed);
        assert!(tape.gradient(weights).is_err());
        assert!(tape.backward_scalar(G::Overwrite).is_err());
        assert!(tape.sgd(&[weights], 0.05).is_err());
        assert_eq!(tape.resources().edge_scratch_bytes, 0);
        for step in 0..8 {
            let before = owner.read().unwrap().clone();
            let residual: Vec<_> = (0..3)
                .flat_map(|i| {
                    let w = &control;
                    let x = &xx;
                    let t = &tt;
                    (0..2).map(move |j| {
                        (0..2)
                            .map(|k| x.data()[i * 2 + k] as f64 * w[k * 2 + j])
                            .sum::<f64>()
                            - t.data()[i * 2 + j] as f64
                    })
                })
                .collect();
            let expected_loss = residual.iter().map(|v| v * v).sum::<f64>() / 6.;
            let gradient: Vec<_> = (0..2)
                .flat_map(|k| {
                    let x = &xx;
                    let r = &residual;
                    (0..2).map(move |j| {
                        (0..3)
                            .map(|i| x.data()[i * 2 + k] as f64 * 2. * r[i * 2 + j] / 6.)
                            .sum::<f64>()
                    })
                })
                .collect();
            support::no_allocations(|| {
                tape.forward().unwrap();
                tape.backward_scalar(G::Overwrite).unwrap();
            });
            close(
                tape.output(loss).unwrap().storage().data(),
                &[expected_loss],
            );
            close(tape.gradient(weights).unwrap().unwrap().data(), &gradient);
            let actual_loss = tape.output(loss).unwrap().storage().clone();
            let actual_gradient = tape.gradient(weights).unwrap().unwrap().clone();
            assert!(tape.gradient(x).unwrap().is_none());
            support::no_allocations(|| tape.sgd(&[weights, weights], 0.05).unwrap());
            for (w, g) in control.iter_mut().zip(gradient) {
                *w -= 0.05f32 as f64 * g;
            }
            close(owner.read().unwrap().data(), &control);
            assert!(tape.output(loss).is_err());
            assert!(tape.sgd(&[weights], 0.05).is_err());
            export.record(serde_json::json!({"schema_version":1,"kind":"dense_sgd","backend":export::backend(cpu),
                "step":step,"steps":8,"input":export::tensor(&xx),"target":export::tensor(&tt),
                "parameters":[export::tensor(&before)],"rate":0.05f32.to_bits(),"loss":export::tensor(&actual_loss),
                "gradients":[export::tensor(&actual_gradient)],"updated":[export::tensor(&owner.read().unwrap())]}));
        }
    }
    export.finish();
}

#[test]
fn cpu_tape_tied_overlapping_detached_branches_and_ordered_roots() {
    let mut export = export::Exporter::new("branches-cpu");
    for cpu in support::backends() {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([3], true).unwrap();
        let tied = b.leaf([3], true).unwrap();
        let coefficient = b.leaf([3], false).unwrap();
        let unused = b.leaf([1], true).unwrap();
        let view = b
            .view(x, &Layout::from_strides([2, 2], [1, 1], 0).unwrap())
            .unwrap();
        let q = b.unary(U::Square, view).unwrap();
        let sum = b.reduce(R::Sum, q, &[0, 1], false).unwrap();
        let product = b.binary(B::Mul, x, tied).unwrap();
        let product = b.reduce(R::Sum, product, &[0], false).unwrap();
        let coefficient_product = b.binary(B::Mul, x, coefficient).unwrap();
        let coefficient_sum = b.reduce(R::Sum, coefficient_product, &[0], false).unwrap();
        let detached = b.detach(view).unwrap();
        let detached_product = b.binary(B::Mul, view, detached).unwrap();
        let detached_sum = b.reduce(R::Sum, detached_product, &[0, 1], false).unwrap();
        let roots = [sum, sum, product, coefficient_sum, detached_sum, q];
        let program = b.finish(&roots).unwrap();
        let owner = Shared::new(Tensor::from_vec([3], vec![1., 2., 3.]).unwrap()).unwrap();
        let bindings = [
            Binding::new(x, owner.clone()),
            Binding::new(tied, owner.clone()),
            Binding::new(coefficient, owner.clone()),
            Binding::new(unused, Shared::new(Tensor::zeros([1]).unwrap()).unwrap()),
        ];
        let scalar = Layout::contiguous([]).unwrap();
        let qseed = Layout::from_strides([2, 2], [-3, 1], 3).unwrap();
        let layouts = [
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            qseed,
        ];
        let seed_tensors = [
            Tensor::from_vec([], vec![0.5]).unwrap(),
            Tensor::from_vec([], vec![1.5]).unwrap(),
            Tensor::from_vec([], vec![1.]).unwrap(),
            Tensor::from_vec([], vec![1.]).unwrap(),
            Tensor::from_vec([], vec![1.]).unwrap(),
            Tensor::from_vec([5], vec![0.3, 0.4, f32::from_bits(0x7fc12345), 0.1, 0.2]).unwrap(),
        ];
        let seeds: Vec<_> = roots
            .iter()
            .zip(&seed_tensors)
            .zip(&layouts)
            .map(|((&v, t), l)| Seed::new(v, t, l))
            .collect();
        assert!(
            cpu::Tape::with_options(
                cpu,
                &program,
                &bindings,
                PrepareOptions {
                    seed_layouts: Some(&layouts),
                    view_index_limit_bytes: 0
                }
            )
            .is_err()
        );
        let mut tape = cpu::Tape::with_options(
            cpu,
            &program,
            &bindings,
            PrepareOptions {
                seed_layouts: Some(&layouts),
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        assert!(tape.resources().edge_scratch_bytes > 0 && tape.resources().view_index_bytes > 0);
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward(&seeds, G::Overwrite).unwrap();
        });
        close(tape.gradient(x).unwrap().unwrap().data(), &[8.2, 28., 26.4]);
        assert!(std::ptr::eq(
            tape.gradient(x).unwrap().unwrap(),
            tape.gradient(tied).unwrap().unwrap()
        ));
        assert_eq!(tape.gradient_status(coefficient).unwrap(), S::Unused);
        assert_eq!(tape.gradient_status(unused).unwrap(), S::Unused);
        assert!(tape.gradient(view).is_err());
        let first = tape.gradient(x).unwrap().unwrap().data().to_vec();
        let outputs = roots.map(|root| export::tensor(tape.output(root).unwrap().storage()));
        tape.backward(&seeds, G::Overwrite).unwrap();
        assert_eq!(tape.gradient(x).unwrap().unwrap().data(), first);
        tape.backward(&seeds, G::Accumulate).unwrap();
        close(
            tape.gradient(x).unwrap().unwrap().data(),
            &[16.4, 56., 52.8],
        );
        let before = owner.version().unwrap();
        assert!(tape.sgd(&[x, coefficient], 0.01).is_err());
        assert_eq!(owner.version().unwrap(), before);
        tape.sgd(&[x, tied, x], 0.01).unwrap();
        assert_eq!(
            owner.version().unwrap().generation(),
            before.generation() + 1
        );
        close(
            owner.read().unwrap().data(),
            &[
                1. - 0.01f32 as f64 * 16.4,
                2. - 0.01f32 as f64 * 56.,
                3. - 0.01f32 as f64 * 52.8,
            ],
        );
        assert_eq!(seed_tensors[5].data()[2].to_bits(), 0x7fc12345);
        export.record(serde_json::json!({"schema_version":1,"kind":"branches","backend":export::backend(cpu),
            "input":export::tensor(&Tensor::from_vec([3],vec![1.,2.,3.]).unwrap()),"outputs":outputs,
            "gradient":export::tensor(&Tensor::from_vec([3],first).unwrap()),
            "accumulated":export::tensor(tape.gradient(x).unwrap().unwrap()),
            "updated":export::tensor(&owner.read().unwrap()),"rate":0.01f32.to_bits()}));
    }
    export.finish();
}

#[test]
fn cpu_tape_selective_saves_and_update_epoch_checks() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let y = b.unary(U::Neg, x).unwrap();
    let sum = b.reduce(R::Sum, y, &[0], false).unwrap();
    let program = b.finish(&[sum]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![2., 3.]).unwrap()).unwrap();
    let mut tape =
        cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(x, owner.clone())]).unwrap();
    tape.forward().unwrap();
    owner.write().unwrap().data_mut()[0] = 4.;
    // Neg/Reduce derivatives need no old input payload; updates still require the
    // original parameter version even when that payload was not saved for a VJP.
    tape.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[-1., -1.]);
    assert!(matches!(
        tape.sgd(&[x], 0.1),
        Err(Error::StorageModified { .. })
    ));
    tape.forward().unwrap();
    assert!(tape.sgd(&[x], 0.1).is_err());
    tape.backward_scalar(G::Overwrite).unwrap();
    let before = owner.version().unwrap();
    let reader = owner.read().unwrap();
    assert!(matches!(
        tape.sgd(&[x], 0.1),
        Err(Error::StorageBorrowConflict)
    ));
    assert_eq!(tape.gradient_status(x).unwrap(), S::Valid);
    assert_eq!(owner.version().unwrap(), before);
    drop(reader);
    for rate in [f32::NAN, f32::INFINITY, -0.1] {
        assert!(tape.sgd(&[x], rate).is_err());
    }
    tape.zero_grad().unwrap();
    assert!(tape.sgd(&[x], 0.1).is_err());
    tape.backward_scalar(G::Accumulate).unwrap();
    tape.sgd(&[x], 0.1).unwrap();
    close(owner.read().unwrap().data(), &[4.1, 3.1]);

    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    let square = b.unary(U::Square, x).unwrap();
    let loss = b.reduce(R::Sum, square, &[0], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let mut tape =
        cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(x, owner.clone())]).unwrap();
    tape.forward().unwrap();
    tape.backward_scalar(G::Overwrite).unwrap();
    owner.write().unwrap().data_mut()[0] = 9.;
    assert!(matches!(
        tape.backward_scalar(G::Accumulate),
        Err(Error::StorageModified { .. })
    ));
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[2., 4.]);
    tape.forward().unwrap();
    tape.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[18., 4.]);
}

#[test]
fn cpu_tape_scalar_empty_and_payload_free_view_roots() {
    for cpu in support::backends() {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([], true).unwrap();
        let expanded = b.broadcast_to(x, [2, 3]).unwrap();
        let empty = b.slice(expanded, 0, 0, 0, 1).unwrap();
        let empty_sum = b.reduce(R::Sum, empty, &[0, 1], false).unwrap();
        let full_sum = b.reduce(R::Sum, expanded, &[0, 1], false).unwrap();
        let result = b.binary(B::Add, empty_sum, full_sum).unwrap();
        let program = b.finish(&[result]).unwrap();
        let owner = Shared::new(Tensor::from_vec([], vec![2.]).unwrap()).unwrap();
        let mut tape = cpu::Tape::with_options(
            cpu,
            &program,
            &[Binding::new(x, owner)],
            PrepareOptions {
                view_index_limit_bytes: 0,
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        support::no_allocations(|| {
            tape.forward().unwrap();
            tape.backward_scalar(G::Overwrite).unwrap();
        });
        assert_eq!(tape.output(result).unwrap().storage().data(), &[12.]);
        assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[6.]);
        assert_eq!(tape.resources().view_index_bytes, 0);
    }
}

#[test]
fn cpu_tape_rejects_foreign_values_bindings_seeds_and_failed_shapes() {
    let mut other = Builder::new().unwrap();
    let foreign = other.leaf([2], true).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2], true).unwrap();
    assert!(b.binary(B::Add, x, foreign).is_err());
    assert!(b.reduce(R::Sum, x, &[0, 0], false).is_err());
    assert!(b.matmul(x, x).is_err());
    assert!(
        b.view(x, &Layout::from_strides([3], [1], 0).unwrap())
            .is_err()
    );
    let y = b.unary(U::Square, x).unwrap();
    let program = b.finish(&[y]).unwrap();
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    assert!(cpu::Tape::new(Cpu::Reference, &program, &[]).is_err());
    assert!(cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(y, owner.clone())]).is_err());
    assert!(
        cpu::Tape::new(
            Cpu::Reference,
            &program,
            &[Binding::new(foreign, owner.clone())]
        )
        .is_err()
    );
    let mut tape =
        cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(x, owner.clone())]).unwrap();
    tape.forward().unwrap();
    let layout = Layout::contiguous([2]).unwrap();
    let seed = Tensor::from_vec([2], vec![3., 4.]).unwrap();
    tape.backward(&[Seed::new(y, &seed, &layout)], G::Overwrite)
        .unwrap();
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[6., 16.]);
    assert!(
        tape.backward(&[Seed::new(foreign, &seed, &layout)], G::Overwrite)
            .is_err()
    );
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[6., 16.]);
    *owner.write().unwrap() = Tensor::zeros([1, 2]).unwrap();
    assert!(tape.forward().is_err());
    assert!(tape.output(y).is_err());
}

#[test]
fn cpu_tape_independent_instances_share_leaves_on_real_threads() {
    for cpu in support::backends() {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([2, 3], true).unwrap();
        let transpose = b.permute(x, &[1, 0]).unwrap();
        let gram = b.matmul(transpose, x).unwrap();
        let loss = b.reduce(R::Sum, gram, &[0, 1], false).unwrap();
        let program = b.finish(&[loss]).unwrap();
        let owner =
            Shared::new(Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap()).unwrap();
        let instances = (0..3)
            .map(|_| cpu::Tape::new(cpu, &program, &[Binding::new(x, owner.clone())]).unwrap())
            .collect::<Vec<_>>();
        std::thread::scope(|scope| {
            for mut tape in instances {
                scope.spawn(move || {
                    assert_eq!(tape.resources().activation_bytes, 40);
                    assert_eq!(tape.resources().gradient_bytes, 64);
                    assert_eq!(tape.resources().edge_scratch_bytes, 24);
                    for _ in 0..2 {
                        support::no_allocations(|| {
                            tape.forward().unwrap();
                            tape.backward_scalar(G::Overwrite).unwrap();
                        });
                        assert_eq!(tape.output(loss).unwrap().storage().data(), &[261.]);
                        assert_eq!(
                            tape.gradient(x).unwrap().unwrap().data(),
                            &[12., 12., 12., 30., 30., 30.]
                        );
                    }
                });
            }
        });
        assert_eq!(owner.read().unwrap().data(), &[1., 2., 3., 4., 5., 6.]);
    }
}

#[test]
fn cpu_tape_miri_signed_overlap_and_repeated_epochs() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([4], true).unwrap();
    let view = b
        .view(x, &Layout::from_strides([2, 3], [1, -1], 2).unwrap())
        .unwrap();
    let square = b.unary(U::Square, view).unwrap();
    let loss = b.reduce(R::Sum, square, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let owner = Shared::new(Tensor::from_vec([4], vec![1., 2., 3., 4.]).unwrap()).unwrap();
    let mut tape =
        cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(x, owner.clone())]).unwrap();
    assert_eq!(tape.resources().activation_bytes, 28);
    assert_eq!(tape.resources().gradient_bytes, 44);
    assert_eq!(tape.resources().edge_scratch_bytes, 24);
    tape.forward().unwrap();
    tape.backward_scalar(G::Overwrite).unwrap();
    assert_eq!(
        tape.gradient(x).unwrap().unwrap().data(),
        &[2., 8., 12., 8.]
    );
    tape.backward_scalar(G::Accumulate).unwrap();
    assert_eq!(
        tape.gradient(x).unwrap().unwrap().data(),
        &[4., 16., 24., 16.]
    );
    tape.sgd(&[x], 0.1).unwrap();
    close(owner.read().unwrap().data(), &[0.6, 0.4, 0.6, 2.4]);
    assert!(tape.backward_scalar(G::Overwrite).is_err());
    tape.forward().unwrap();
    tape.backward_scalar(G::Overwrite).unwrap();
    let p = owner.read().unwrap();
    close(
        tape.gradient(x).unwrap().unwrap().data(),
        &[
            2. * p.data()[0] as f64,
            4. * p.data()[1] as f64,
            4. * p.data()[2] as f64,
            2. * p.data()[3] as f64,
        ],
    );
}

#[test]
fn cpu_tape_zero_contribution_edges_preserve_root_bits_without_view_indices() {
    for cpu in support::backends() {
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
            let owner = Shared::new(Tensor::from_vec([4], vec![1., 2., 3., 4.]).unwrap()).unwrap();
            let zero =
                Shared::new(Tensor::zeros(program.shape(empty).unwrap().dims()).unwrap()).unwrap();
            let mut tape = cpu::Tape::with_options(
                cpu,
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
            let seed =
                Tensor::from_vec([4], bits.iter().map(|&v| f32::from_bits(v)).collect()).unwrap();
            let empty_seed = Tensor::zeros(program.shape(y).unwrap().dims()).unwrap();
            let first = Layout::contiguous([4]).unwrap();
            let second = Layout::contiguous(program.shape(y).unwrap().dims()).unwrap();
            tape.forward().unwrap();
            tape.backward(
                &[
                    Seed::new(x, &seed, &first),
                    Seed::new(y, &empty_seed, &second),
                ],
                G::Overwrite,
            )
            .unwrap();
            assert_eq!(
                tape.gradient(x)
                    .unwrap()
                    .unwrap()
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            );
        }
    }
}

#[test]
fn cpu_tape_view_outputs_and_large_virtual_roots_do_not_materialize() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([4], true).unwrap();
    let v = b
        .view(x, &Layout::from_strides([2, 2], [1, -1], 1).unwrap())
        .unwrap();
    let program = b.finish(&[v]).unwrap();
    let owner = Shared::new(Tensor::from_vec([4], vec![1., 2., 3., 4.]).unwrap()).unwrap();
    let mut tape =
        cpu::Tape::new(Cpu::Reference, &program, &[Binding::new(x, owner.clone())]).unwrap();
    tape.forward().unwrap();
    {
        let output = tape.output(v).unwrap();
        assert_eq!(output.storage().shape().dims(), &[4]);
        assert_eq!(output.layout(), program.layout(v).unwrap());
        assert_eq!(
            output.storage().data().as_ptr(),
            owner.read().unwrap().data().as_ptr()
        );
        let view = output.view().unwrap();
        assert_eq!(view.get(&[0, 0]), Some(&2.));
        assert_eq!(view.get(&[1, 1]), Some(&2.));
    }
    assert_eq!(tape.resources().activation_bytes, 0);
    let seed = Tensor::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap();
    let layout = Layout::contiguous([2, 2]).unwrap();
    tape.backward(&[Seed::new(v, &seed, &layout)], G::Overwrite)
        .unwrap();
    assert_eq!(tape.gradient(x).unwrap().unwrap().data(), &[2., 5., 3., 0.]);

    let n = 6_000_000_001usize;
    let mut b = Builder::new().unwrap();
    let x = b.leaf([], true).unwrap();
    let root = b.broadcast_to(x, [n]).unwrap();
    let program = b.finish(&[root]).unwrap();
    let seed_layout = [Layout::from_strides([n], [0], 0).unwrap()];
    let mut tape = cpu::Tape::with_options(
        Cpu::Reference,
        &program,
        &[Binding::new(
            x,
            Shared::new(Tensor::zeros([]).unwrap()).unwrap(),
        )],
        PrepareOptions {
            seed_layouts: Some(&seed_layout),
            view_index_limit_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(tape.resources().activation_bytes, 0);
    assert_eq!(tape.resources().gradient_bytes, 4);
    assert_eq!(tape.resources().edge_scratch_bytes, 0);
    assert_eq!(tape.resources().view_index_bytes, 0);
    assert!(tape.resources().operator_host_metadata_bytes < 2048);
    tape.forward().unwrap();
    assert_eq!(tape.output(root).unwrap().storage().shape().len(), 1);
}

#[test]
fn cpu_tape_fused_step_matches_unfused_across_dense_waveform_spectrum_and_conv() {
    // 1. Waveform (contiguous + reversed-permuted) & Spectrum (contiguous + reversed-permuted, K=2)
    //    matching training-characterization-v1 topology with detached prediction root.
    for &(family, reversed) in &[
        ("waveform", false),
        ("waveform", true),
        ("spectrum", false),
        ("spectrum", true),
    ] {
        let mut b = Builder::new().unwrap();
        let (x_dims, tgt_dims, g_dims, b_dims, g_view_dims, b_view_dims) = if family == "waveform" {
            (
                vec![2, 3, 19],
                if reversed {
                    vec![2, 19, 3]
                } else {
                    vec![2, 3, 19]
                },
                vec![3],
                vec![3],
                vec![1, 3, 1],
                vec![1, 3, 1],
            )
        } else {
            (
                vec![2, 2, 5, 7, 2],
                if reversed {
                    vec![2, 7, 2, 5]
                } else {
                    vec![2, 2, 5, 7]
                },
                vec![2, 5],
                vec![2],
                vec![1, 2, 5, 1, 1],
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
        let mk_owners = || {
            (
                Shared::new(x_t.clone()).unwrap(),
                Shared::new(tgt_t.clone()).unwrap(),
                Shared::new(g_t.clone()).unwrap(),
                Shared::new(b_t.clone()).unwrap(),
            )
        };
        let (rx, rt, rg, rb) = mk_owners();
        let (fx, ft, fg, fb) = mk_owners();
        let seed_layouts = [
            Layout::contiguous([]).unwrap(),
            Layout::from_strides(&tgt_dims, vec![0; tgt_dims.len()], 0).unwrap(),
        ];
        let mut ref_tape = cpu::Tape::with_options(
            Cpu::Reference,
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
        let mut fused_tape = cpu::Tape::with_options(
            Cpu::Reference,
            &program,
            &[
                Binding::new(x, fx),
                Binding::new(tgt, ft),
                Binding::new(gain, fg.clone()),
                Binding::new(bias, fb.clone()),
            ],
            PrepareOptions {
                seed_layouts: Some(&seed_layouts),
                ..PrepareOptions::default()
            },
        )
        .unwrap();
        let one = Tensor::from_vec([], vec![1.0]).unwrap();
        let zero = Tensor::zeros([]).unwrap();
        let seeds = [
            Seed::new(loss, &one, &seed_layouts[0]),
            Seed::new(detached, &zero, &seed_layouts[1]),
        ];
        let params = [gain, bias];
        for step in 0..8 {
            ref_tape.forward().unwrap();
            ref_tape.backward(&seeds, G::Overwrite).unwrap();
            ref_tape.sgd(&params, 0.05).unwrap();
            if step >= 1 {
                support::no_allocations(|| {
                    fused_tape.step(&seeds, &params, 0.05).unwrap();
                });
            } else {
                fused_tape.step(&seeds, &params, 0.05).unwrap();
            }
            assert!(
                fused_tape.has_fused_plan(),
                "expected fusion for {family} reversed={reversed}"
            );
            for (r_owner, f_owner, p_val) in [(&rg, &fg, gain), (&rb, &fb, bias)] {
                let rp = r_owner.read().unwrap();
                let fp = f_owner.read().unwrap();
                support::close(
                    fp.data(),
                    &rp.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    2e-5,
                    2e-5,
                );
                let rgrad = ref_tape.gradient(p_val).unwrap().unwrap();
                let fgrad = fused_tape.gradient(p_val).unwrap().unwrap();
                support::close(
                    fgrad.data(),
                    &rgrad.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    2e-5,
                    2e-5,
                );
            }
        }
        // SavedValue version invalidation and borrow-conflict check.
        let p0_layout = std::sync::Arc::new(Layout::contiguous(&g_dims).unwrap());
        let saved = fg.save(&p0_layout).unwrap();
        let v_before = fg.version().unwrap();
        support::no_allocations(|| {
            fused_tape.step(&seeds, &params, 0.05).unwrap();
        });
        assert_eq!(
            fg.version().unwrap().generation(),
            v_before.generation() + 1
        );
        assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
        let held = fg.read().unwrap();
        assert!(matches!(
            fused_tape.step(&seeds, &params, 0.05),
            Err(Error::StorageBorrowConflict)
        ));
        drop(held);
    }

    // 2. BarrierWithTail: Matmul + bias + Tanh + MSE (tests barrier + SmallGlobal tail param)
    let mut b = Builder::new().unwrap();
    let x = b.leaf([4, 3], false).unwrap();
    let w = b.leaf([3, 2], true).unwrap();
    let bias = b.leaf([1, 2], true).unwrap();
    let target = b.leaf([4, 2], false).unwrap();
    let mm = b.matmul(x, w).unwrap();
    let z = b.binary(B::Add, mm, bias).unwrap();
    let pred = b.unary(U::Tanh, z).unwrap();
    let diff = b.binary(B::Sub, pred, target).unwrap();
    let sq = b.unary(U::Square, diff).unwrap();
    let loss = b.reduce(R::Mean, sq, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();
    let x_t = Tensor::from_vec([4, 3], (0..12).map(|i| (i as f32 - 5.5) / 8.0).collect()).unwrap();
    let w_t = Tensor::from_vec([3, 2], (0..6).map(|i| (i as f32 - 2.5) / 16.0).collect()).unwrap();
    let b_t = Tensor::from_vec([1, 2], vec![0.125, -0.25]).unwrap();
    let tgt_t = Tensor::from_vec([4, 2], (0..8).map(|i| (i as f32 - 3.5) / 4.0).collect()).unwrap();
    let (rx, rw, rb, rt) = (
        Shared::new(x_t.clone()).unwrap(),
        Shared::new(w_t.clone()).unwrap(),
        Shared::new(b_t.clone()).unwrap(),
        Shared::new(tgt_t.clone()).unwrap(),
    );
    let (fx, fw, fb, ft) = (
        Shared::new(x_t).unwrap(),
        Shared::new(w_t).unwrap(),
        Shared::new(b_t).unwrap(),
        Shared::new(tgt_t).unwrap(),
    );
    let mut ref_tape = cpu::Tape::new(
        Cpu::Reference,
        &program,
        &[
            Binding::new(x, rx),
            Binding::new(w, rw.clone()),
            Binding::new(bias, rb.clone()),
            Binding::new(target, rt),
        ],
    )
    .unwrap();
    let mut fused_tape = cpu::Tape::new(
        Cpu::Reference,
        &program,
        &[
            Binding::new(x, fx),
            Binding::new(w, fw.clone()),
            Binding::new(bias, fb.clone()),
            Binding::new(target, ft),
        ],
    )
    .unwrap();
    for step in 0..6 {
        ref_tape.forward().unwrap();
        ref_tape.backward_scalar(G::Overwrite).unwrap();
        ref_tape.sgd(&[w, bias], 0.0625).unwrap();
        if step >= 1 {
            support::no_allocations(|| {
                fused_tape.step_scalar(&[w, bias], 0.0625).unwrap();
            });
        } else {
            fused_tape.step_scalar(&[w, bias], 0.0625).unwrap();
        }
        assert!(fused_tape.has_fused_plan());
        support::close(
            fw.read().unwrap().data(),
            &rw.read()
                .unwrap()
                .data()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            2e-5,
            2e-5,
        );
        support::close(
            fb.read().unwrap().data(),
            &rb.read()
                .unwrap()
                .data()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            2e-5,
            2e-5,
        );
    }
}

#[test]
fn cpu_tape_activation_liveness_reuses_dead_buffers_and_preserves_exact_values() {
    // 8-step additive/negation chain: none of Add/Sub/Neg require saving their inputs for backward.
    // Exact liveness reuse must keep peak activation_bytes to at most 2 intermediate buffers + scalar
    // instead of 8 full-sized buffers, while preserving exact forward & backward results.
    let mut b = Builder::new().unwrap();
    let x = b.leaf([4, 64], true).unwrap();
    let c = b.leaf([4, 64], false).unwrap();
    let mut cur = x;
    for step in 0..8 {
        cur = if step % 2 == 0 {
            b.binary(B::Add, cur, c).unwrap()
        } else {
            b.unary(U::Neg, cur).unwrap()
        };
    }
    let loss = b.reduce(R::Sum, cur, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let sx = Shared::new(
        Tensor::from_vec(
            [4, 64],
            (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let sc = Shared::new(
        Tensor::from_vec([4, 64], (0..256).map(|i| ((i % 7) as f32) * 0.05).collect()).unwrap(),
    )
    .unwrap();
    let mut tape = cpu::Tape::new(
        Cpu::Reference,
        &program,
        &[Binding::new(x, sx.clone()), Binding::new(c, sc.clone())],
    )
    .unwrap();

    let unshared_bytes = 8 * 256 * size_of::<f32>() + size_of::<f32>();
    let reused_bytes = tape.resources().activation_bytes;
    assert!(
        reused_bytes < unshared_bytes / 2,
        "expected activation liveness reuse to slash activation_bytes: reused={reused_bytes}, unshared={unshared_bytes}"
    );

    tape.forward().unwrap();
    // Trace the 8 steps mathematically:
    // s0: x+c, s1: -(x+c), s2: -x, s3: x, s4: x+c, s5: -(x+c), s6: -x, s7: x => sum(x)
    let expected_loss: f32 = sx.read().unwrap().data().iter().sum();
    let got_loss = tape.output(loss).unwrap().storage().data()[0];
    assert!((got_loss - expected_loss).abs() < 1e-4);

    tape.backward_scalar(G::Overwrite).unwrap();
    let gx = tape.gradient(x).unwrap().unwrap();
    assert!(gx.data().iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn cpu_fused_2d_cache_tiled_spectrum_and_waveform_match_unfused_zero_alloc() {
    // Exercise 2D L1 micro-transpose fast kernels:
    // 1) spectrum reversed-permuted with F=19 >= 8, C=4, |s_inner|=76 >= 16
    // 2) waveform reversed-permuted with C=8 >= 4, T=65, |s_inner|=8 >= 4
    for (family, dims) in [
        ("spectrum", vec![2usize, 4, 19, 33]),
        ("waveform", vec![2usize, 8, 65]),
    ] {
        let mut b = Builder::new().unwrap();
        let (x_dims, tgt_dims, g_dims, b_dims) = if family == "spectrum" {
            let [bs, c, f, t] = [dims[0], dims[1], dims[2], dims[3]];
            (vec![bs, c, f, t, 2], vec![bs, t, c, f], vec![c, f], vec![2])
        } else {
            let [bs, c, t] = [dims[0], dims[1], dims[2]];
            (vec![bs, c, t], vec![bs, t, c], vec![c], vec![c])
        };
        let x = b.leaf(&x_dims, false).unwrap();
        let tgt = b.leaf(&tgt_dims, false).unwrap();
        let gain = b.leaf(&g_dims, true).unwrap();
        let bias = b.leaf(&b_dims, true).unwrap();
        let pred = if family == "spectrum" {
            let [_bs, c, f, t] = [dims[0], dims[1], dims[2], dims[3]];
            let g5 = b.reshape(gain, [1, c, f, 1, 1]).unwrap();
            let s = b.binary(B::Mul, x, g5).unwrap();
            let z = b.binary(B::Add, s, bias).unwrap();
            let sq = b.unary(U::Square, z).unwrap();
            let raw = b.reduce(R::Sum, sq, &[4], false).unwrap();
            let rev = b.slice(raw, 3, t - 1, t, -1).unwrap();
            b.permute(rev, &[0, 3, 1, 2]).unwrap()
        } else {
            let [bs, c, t] = [dims[0], dims[1], dims[2]];
            let _ = bs;
            let g3 = b.reshape(gain, [1, c, 1]).unwrap();
            let b3 = b.reshape(bias, [1, c, 1]).unwrap();
            let s = b.binary(B::Mul, x, g3).unwrap();
            let z = b.binary(B::Add, s, b3).unwrap();
            let raw = b.unary(U::Tanh, z).unwrap();
            let rev = b.slice(raw, 2, t - 1, t, -1).unwrap();
            b.permute(rev, &[0, 2, 1]).unwrap()
        };
        let diff = b.binary(B::Sub, pred, tgt).unwrap();
        let sq = b.unary(U::Square, diff).unwrap();
        let axes: Vec<usize> = (0..tgt_dims.len()).collect();
        let loss = b.reduce(R::Mean, sq, &axes, false).unwrap();
        let program = b.finish(&[loss]).unwrap();

        let mk = |ds: &[usize], salt: usize| {
            let len: usize = ds.iter().product();
            Tensor::from_vec(
                ds,
                (0..len)
                    .map(|i| (((i * (7 + salt) + 3) % 29) as f32 - 14.0) / 32.0)
                    .collect(),
            )
            .unwrap()
        };
        let xt = mk(&x_dims, 0);
        let tt = mk(&tgt_dims, 1);
        let gt = mk(&g_dims, 2);
        let bt = mk(&b_dims, 3);

        let rx = Shared::new(xt.clone()).unwrap();
        let rt = Shared::new(tt.clone()).unwrap();
        let rg = Shared::new(gt.clone()).unwrap();
        let rb = Shared::new(bt.clone()).unwrap();
        let fx = Shared::new(xt).unwrap();
        let ft = Shared::new(tt).unwrap();
        let fg = Shared::new(gt).unwrap();
        let fb = Shared::new(bt).unwrap();

        let mut ref_tape = cpu::Tape::new(
            Cpu::Reference,
            &program,
            &[
                Binding::new(x, rx),
                Binding::new(tgt, rt),
                Binding::new(gain, rg.clone()),
                Binding::new(bias, rb.clone()),
            ],
        )
        .unwrap();
        let mut fused_tape = cpu::Tape::new(
            Cpu::Reference,
            &program,
            &[
                Binding::new(x, fx),
                Binding::new(tgt, ft),
                Binding::new(gain, fg.clone()),
                Binding::new(bias, fb.clone()),
            ],
        )
        .unwrap();

        for step in 0..4 {
            ref_tape.forward().unwrap();
            ref_tape.backward_scalar(G::Overwrite).unwrap();
            ref_tape.sgd(&[gain, bias], 0.05).unwrap();
            if step >= 1 {
                support::no_allocations(|| {
                    fused_tape.step_scalar(&[gain, bias], 0.05).unwrap();
                });
            } else {
                fused_tape.step_scalar(&[gain, bias], 0.05).unwrap();
            }
            assert!(fused_tape.has_fused_plan());
            for (r, f) in [(&rg, &fg), (&rb, &fb)] {
                let rv: Vec<f64> = r.read().unwrap().data().iter().map(|&v| v as f64).collect();
                support::close(f.read().unwrap().data(), &rv, 2e-5, 2e-5);
            }
        }
    }
}
