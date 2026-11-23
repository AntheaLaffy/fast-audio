use fast_audio::{
    Activation, Conv1dSpec, Error, Layout, StorageVersion, Tensor, Trackable, Tracked, cpu,
};
use std::{collections::HashSet, panic::AssertUnwindSafe, sync::Arc};

#[allow(
    dead_code,
    reason = "Storage tests use exact fixtures rather than tolerance helpers."
)]
mod support;
use support::{backends, no_allocations};

fn changed<T: Trackable>(owner: &Tracked<T>, saved: StorageVersion) {
    match owner.check_version(saved) {
        Err(Error::StorageIdentityMismatch {
            saved: actual,
            current,
        }) => {
            assert_eq!(actual, saved.storage_id());
            assert_eq!(current, owner.version().storage_id());
            assert_ne!(current, actual);
        }
        Err(Error::StorageModified {
            storage,
            saved: actual,
            current,
        }) => {
            assert_eq!(storage, saved.storage_id());
            assert_eq!(storage, owner.version().storage_id());
            assert_eq!(actual, saved.generation());
            assert_eq!(current, owner.version().generation());
            assert_ne!(current, actual);
        }
        _ => panic!("expected a structured identity/generation mismatch"),
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn untracked_inference_types_keep_their_existing_size_budgets() {
    assert!(size_of::<Tensor>() <= 48);
    assert!(
        size_of::<Error>() <= 32,
        "error size: {}",
        size_of::<Error>()
    );
    #[cfg(feature = "cuda")]
    assert!(size_of::<fast_audio::cuda::DeviceTensor>() <= 40);
}

#[test]
fn identities_follow_ownership_moves_not_addresses_or_copies() {
    let a = Tracked::new(Tensor::from_vec([2], vec![2.0, 3.0]).unwrap()).unwrap();
    let saved = a.version();
    let address = a.read().data().as_ptr();
    let mut owners = vec![a];
    let mut a = owners.pop().unwrap();
    assert_eq!(a.version(), saved);
    assert_eq!(a.check_version(saved).unwrap().data().as_ptr(), address);

    let copy = Tracked::new(a.read().clone()).unwrap();
    assert_ne!(copy.version().storage_id(), saved.storage_id());
    assert_eq!(copy.read().data(), a.read().data());
    changed(&copy, saved);
    a.write().unwrap().data_mut()[0] = 8.0;
    assert_eq!(copy.read().data(), &[2.0, 3.0]);
    changed(&a, saved);

    let before = a.version();
    let old = std::mem::replace(a.write().unwrap(), Tensor::zeros([3]).unwrap());
    assert_eq!(a.version().storage_id(), saved.storage_id());
    assert_eq!(a.version().generation(), before.generation() + 1);
    changed(&a, before);
    let old = Tracked::new(old).unwrap();
    changed(&old, before);
    let before = old.version();
    let unwrapped = old.into_inner();
    let address = unwrapped.data().as_ptr();
    let rewrapped = Tracked::new(unwrapped).unwrap();
    assert_eq!(rewrapped.read().data().as_ptr(), address);
    assert_eq!(rewrapped.version().generation(), 0);
    changed(&rewrapped, before);
}

#[test]
fn concurrent_host_owners_reserve_unique_identities_and_share_reads() {
    let shared = Arc::new(Tracked::new(Tensor::from_vec([], vec![7.0]).unwrap()).unwrap());
    let saved = shared.version();
    let owners: Vec<_> = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..8)
            .map(|_| {
                let shared = Arc::clone(&shared);
                scope.spawn(move || {
                    (0..64)
                        .map(|_| {
                            assert_eq!(shared.check_version(saved).unwrap().data(), &[7.0]);
                            Tracked::new(Tensor::zeros([0]).unwrap()).unwrap()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        jobs.into_iter().flat_map(|j| j.join().unwrap()).collect()
    });
    let ids: HashSet<_> = owners.iter().map(|t| t.version().storage_id()).collect();
    assert_eq!(ids.len(), 8 * 64);
    assert!(!ids.contains(&saved.storage_id()));
    for owner in owners {
        assert_eq!(owner.version().generation(), 0);
        changed(&owner, saved);
    }
}

#[test]
fn saved_checks_match_an_independent_access_history() {
    let mut owners = std::array::from_fn::<_, 3, _>(|_| {
        Tracked::new(Tensor::from_vec([2], vec![0.0, 1.0]).unwrap()).unwrap()
    });
    let mut epochs = [0; 3];
    let mut values = [0.0; 3];
    let mut history = Vec::new();
    let mut state = 17_u32;
    for step in 0..400 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let index = (state >> 8) as usize % owners.len();
        match state % 4 {
            0 => history.push((owners[index].version(), index, epochs[index])),
            1 => {
                let _ = owners[index].write().unwrap();
                epochs[index] += 1;
            }
            2 => {
                owners[index].write().unwrap().data_mut()[0] = step as f32;
                epochs[index] += 1;
                values[index] = step as f32;
            }
            _ => assert_eq!(owners[index].read().data()[0], values[index]),
        }
        for &(saved, source, epoch) in &history {
            for (current, owner) in owners.iter().enumerate() {
                let acceptable = current == source && epochs[current] == epoch;
                assert_eq!(owner.check_version(saved).is_ok(), acceptable);
                if !acceptable {
                    changed(owner, saved);
                }
            }
        }
    }
    assert!(history.len() > 50);
    assert!(epochs.iter().all(|&n| n > 30));
}

#[test]
fn partial_writes_unwinding_and_failed_operations_invalidate_saved_versions() {
    let mut owner = Tracked::new(Tensor::from_vec([2], vec![1.0, 2.0]).unwrap()).unwrap();
    let saved = owner.version();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        owner.write().unwrap().data_mut()[0] = 9.0;
        panic!("simulated failure after a partial update");
    }));
    assert!(result.is_err());
    changed(&owner, saved);
    assert_eq!(owner.read().data(), &[9.0, 2.0]);

    let a = Tensor::zeros([1, 2]).unwrap();
    let b = Tensor::zeros([2, 1]).unwrap();
    let saved = owner.version();
    assert!(
        cpu::Cpu::Reference
            .matmul(&a, &b, owner.write().unwrap())
            .is_err()
    );
    changed(&owner, saved);
    assert_eq!(owner.read().data(), &[9.0, 2.0]);
    assert_eq!(owner.version().generation(), 2);
    assert!(
        Error::StorageModified {
            storage: saved.storage_id(),
            saved: saved.generation(),
            current: owner.version().generation()
        }
        .to_string()
        .contains("saved generation")
    );
}

#[test]
fn layout_reads_preserve_versions_and_disjoint_writes_invalidate_the_whole_owner() {
    let mut owner =
        Tracked::new(Tensor::from_vec([6], (0..6).map(|i| i as f32).collect()).unwrap()).unwrap();
    let transpose = Layout::contiguous([2, 3])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let saved = owner.version();
    assert_eq!(
        owner
            .read()
            .view(&transpose)
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    assert!(owner.check_version(saved).is_ok());
    let reverse = Layout::from_strides([3], [-2], 5).unwrap();
    {
        let inner = owner.write().unwrap();
        let mut view = inner.view_mut(&reverse).unwrap();
        let mut elements = view.iter_mut();
        let first = elements.next().unwrap();
        let last = elements.next_back().unwrap();
        *first = 50.0;
        *last = 10.0;
    }
    assert_eq!(owner.version().generation(), 1);
    changed(&owner, saved);
    assert_eq!(owner.read().data(), &[0.0, 10.0, 2.0, 3.0, 4.0, 50.0]);
}

#[test]
fn prepared_math_and_strided_copy_keep_source_versions_and_invalidate_outputs() {
    let a = Tracked::new(Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap()).unwrap();
    let b =
        Tracked::new(Tensor::from_vec([3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap()).unwrap();
    for cpu in backends() {
        let mut out = Tracked::new(Tensor::zeros([2, 2]).unwrap()).unwrap();
        let before = out.version();
        let source = a.version();
        cpu.matmul(a.read(), b.read(), out.write().unwrap())
            .unwrap();
        changed(&out, before);
        assert!(a.check_version(source).is_ok());
        assert_eq!(out.read().data(), &[58., 64., 139., 154.]);
        let x = Layout::contiguous([2, 2])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        let y = Layout::contiguous([2, 2]).unwrap();
        let plan = cpu::CopyPlan::new(&x, &y).unwrap();
        let mut copy = Tracked::new(Tensor::zeros([2, 2]).unwrap()).unwrap();
        let before = copy.version();
        no_allocations(|| {
            for _ in 0..8 {
                cpu.matmul(a.read(), b.read(), out.write().unwrap())
                    .unwrap();
                let current = out.version();
                plan.run(
                    &out.check_version(current)
                        .unwrap()
                        .view(plan.source_layout())
                        .unwrap(),
                    &mut copy
                        .write()
                        .unwrap()
                        .view_mut(plan.destination_layout())
                        .unwrap(),
                )
                .unwrap();
            }
        });
        changed(&copy, before);
        assert_eq!(copy.version().generation(), 8);
        assert_eq!(copy.read().data(), &[58., 139., 64., 154.]);
    }
}

#[test]
fn packed_snapshots_can_be_guarded_and_explicitly_refreshed() {
    for cpu in backends() {
        for kernel in [1, 3] {
            let spec = Conv1dSpec {
                batch: 1,
                in_channels: 1,
                out_channels: 1,
                input_len: 4,
                kernel,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
            };
            let x = Tensor::from_vec([1, 1, 4], vec![1., 2., 3., 4.]).unwrap();
            let mut weights =
                Tracked::new(Tensor::from_vec([1, 1, kernel], vec![2.; kernel]).unwrap()).unwrap();
            let mut bias = Tracked::new(Tensor::from_vec([1], vec![1.]).unwrap()).unwrap();
            let versions = (weights.version(), bias.version());
            let mut old = cpu::PreparedConv1d::new(
                cpu,
                spec,
                weights.read(),
                Some(bias.read()),
                Activation::Identity,
            )
            .unwrap();
            weights.write().unwrap().data_mut().fill(3.);
            bias.write().unwrap().data_mut().fill(4.);
            changed(&weights, versions.0);
            changed(&bias, versions.1);
            let mut out = Tensor::zeros(spec.output_shape().unwrap()).unwrap();
            old.run(&x, &mut out).unwrap();
            assert_eq!(
                out.data(),
                if kernel == 1 {
                    &[3., 5., 7., 9.][..]
                } else {
                    &[13., 19.][..]
                }
            );
            let mut fresh = cpu::PreparedConv1d::new(
                cpu,
                spec,
                weights.check_version(weights.version()).unwrap(),
                Some(bias.check_version(bias.version()).unwrap()),
                Activation::Identity,
            )
            .unwrap();
            fresh.run(&x, &mut out).unwrap();
            assert_eq!(
                out.data(),
                if kernel == 1 {
                    &[7., 10., 13., 16.][..]
                } else {
                    &[22., 31.][..]
                }
            );
        }
    }
}

#[test]
fn successful_tracking_access_and_construction_do_not_allocate() {
    let tensor = Tensor::zeros([2]).unwrap();
    let mut tracked = None;
    no_allocations(|| tracked = Some(Tracked::new(tensor).unwrap()));
    let mut tracked = tracked.unwrap();
    no_allocations(|| {
        for _ in 0..100 {
            let saved = tracked.version();
            assert_eq!(tracked.check_version(saved).unwrap().data()[0], 0.0);
            let _ = tracked.write().unwrap();
            changed(&tracked, saved);
        }
    });
    assert_eq!(tracked.version().generation(), 100);
}

#[test]
fn empty_and_scalar_storage_and_zero_contractions_keep_conservative_versions() {
    let mut scalar = Tracked::new(Tensor::from_vec([], vec![4.0]).unwrap()).unwrap();
    let mut empty = Tracked::new(Tensor::zeros([2, 0]).unwrap()).unwrap();
    assert_ne!(scalar.version().storage_id(), empty.version().storage_id());
    let saved = scalar.version();
    scalar.write().unwrap().data_mut()[0] = 5.0;
    changed(&scalar, saved);
    let saved = empty.version();
    let a = Tensor::zeros([2, 0]).unwrap();
    let b = Tensor::zeros([0, 0]).unwrap();
    cpu::Cpu::Reference
        .matmul(&a, &b, empty.write().unwrap())
        .unwrap();
    changed(&empty, saved);
    for cpu in backends() {
        let b = Tensor::zeros([0, 3]).unwrap();
        let mut out = Tracked::new(Tensor::from_vec([2, 3], vec![9.; 6]).unwrap()).unwrap();
        let saved = out.version();
        cpu.matmul(&a, &b, out.write().unwrap()).unwrap();
        changed(&out, saved);
        assert!(out.read().data().iter().all(|x| x.to_bits() == 0));
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_queued_writes_and_layout_copy_invalidate_before_synchronization() {
    use fast_audio::cuda::{CopyPlan, Cuda};
    let cuda = Cuda::new(0).unwrap();
    let host =
        Tracked::new(Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap()).unwrap();
    let mut a = Tracked::new(cuda.upload(host.read()).unwrap()).unwrap();
    let b = Tracked::new(
        cuda.upload(&Tensor::from_vec([3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap())
            .unwrap(),
    )
    .unwrap();
    let mut out = Tracked::new(cuda.zeros([2, 2]).unwrap()).unwrap();
    changed(&a, host.version());
    let before = out.version();
    let source = a.version();
    cuda.matmul(a.read(), b.read(), out.write().unwrap())
        .unwrap();
    changed(&out, before); // No synchronization or download precedes this check.
    assert!(a.check_version(source).is_ok());
    let x = Layout::contiguous([2, 2])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let y = Layout::contiguous([2, 2]).unwrap();
    let plan = CopyPlan::new(&cuda, &x, &y).unwrap();
    let mut copy = Tracked::new(cuda.zeros([2, 2]).unwrap()).unwrap();
    let before = copy.version();
    plan.run(
        &out.read().view(plan.source_layout()).unwrap(),
        &mut copy
            .write()
            .unwrap()
            .view_mut(plan.destination_layout())
            .unwrap(),
    )
    .unwrap();
    changed(&copy, before);
    assert_eq!(
        cuda.download(copy.read()).unwrap().data(),
        &[58., 139., 64., 154.]
    );
    no_allocations(|| {
        for _ in 0..8 {
            cuda.matmul(a.read(), b.read(), out.write().unwrap())
                .unwrap();
            plan.run(
                &out.read().view(plan.source_layout()).unwrap(),
                &mut copy
                    .write()
                    .unwrap()
                    .view_mut(plan.destination_layout())
                    .unwrap(),
            )
            .unwrap();
        }
    });
    assert_eq!(
        cuda.download(copy.read()).unwrap().data(),
        &[58., 139., 64., 154.]
    );
    let before = a.version();
    cuda.copy_from_host(&Tensor::zeros([2, 3]).unwrap(), a.write().unwrap())
        .unwrap();
    changed(&a, before);
    cuda.matmul(a.read(), b.read(), out.write().unwrap())
        .unwrap();
    assert!(
        cuda.download(out.read())
            .unwrap()
            .data()
            .iter()
            .all(|&x| x == 0.0)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_session_errors_and_empty_scalar_writes_follow_the_same_version_contract() {
    use fast_audio::cuda::{CopyPlan, Cuda};
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let host = Tensor::from_vec([], vec![7.0]).unwrap();
    let mut scalar = Tracked::new(cuda.upload(&host).unwrap()).unwrap();
    let saved = scalar.version();
    assert!(
        foreign
            .copy_from_host(&host, scalar.write().unwrap())
            .is_err()
    );
    changed(&scalar, saved);
    assert_eq!(cuda.download(scalar.read()).unwrap().data(), &[7.0]);
    let layout = Layout::contiguous([]).unwrap();
    let plan = CopyPlan::new(&cuda, &layout, &layout).unwrap();
    let input = Tracked::new(foreign.upload(&host).unwrap()).unwrap();
    let saved = scalar.version();
    assert!(
        plan.run(
            &input.read().view(plan.source_layout()).unwrap(),
            &mut scalar
                .write()
                .unwrap()
                .view_mut(plan.destination_layout())
                .unwrap()
        )
        .is_err()
    );
    changed(&scalar, saved);
    assert_eq!(cuda.download(scalar.read()).unwrap().data(), &[7.0]);
    let input = cuda
        .upload(&Tensor::from_vec([], vec![11.0]).unwrap())
        .unwrap();
    let saved = scalar.version();
    plan.run(
        &input.view(plan.source_layout()).unwrap(),
        &mut scalar
            .write()
            .unwrap()
            .view_mut(plan.destination_layout())
            .unwrap(),
    )
    .unwrap();
    changed(&scalar, saved);
    assert_eq!(cuda.download(scalar.read()).unwrap().data(), &[11.0]);
    let mut empty = Tracked::new(cuda.zeros([2, 0]).unwrap()).unwrap();
    let saved = empty.version();
    cuda.copy_from_host(&Tensor::zeros([2, 0]).unwrap(), empty.write().unwrap())
        .unwrap();
    changed(&empty, saved);
    let a = cuda.zeros([2, 0]).unwrap();
    let b = cuda.zeros([0, 3]).unwrap();
    let mut zero = Tracked::new(
        cuda.upload(&Tensor::from_vec([2, 3], vec![9.; 6]).unwrap())
            .unwrap(),
    )
    .unwrap();
    let saved = zero.version();
    cuda.matmul(&a, &b, zero.write().unwrap()).unwrap();
    changed(&zero, saved);
    assert!(
        cuda.download(zero.read())
            .unwrap()
            .data()
            .iter()
            .all(|x| x.to_bits() == 0)
    );
    let host_empty = Tracked::new(Tensor::zeros([2, 0]).unwrap()).unwrap();
    changed(&empty, host_empty.version());
    assert!(cuda.download(empty.read()).unwrap().data().is_empty());
}
