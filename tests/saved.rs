use fast_audio::{Error, Layout, Shared, Tensor, Tracked, cpu};
use std::{
    panic::AssertUnwindSafe,
    sync::{Arc, Barrier},
};

#[allow(
    dead_code,
    reason = "Retained-storage tests use exact fixtures rather than tolerance helpers."
)]
mod support;
use support::{backends, no_allocations};

#[test]
fn saved_values_keep_payload_and_immutable_layout_after_other_owners_are_dropped() {
    let cases = [
        (
            Layout::from_strides([2, 3], [3, 1], 1).unwrap(),
            vec![1., 2., 3., 4., 5., 6.],
        ),
        (
            Layout::from_strides([2, 3], [-3, -1], 6).unwrap(),
            vec![6., 5., 4., 3., 2., 1.],
        ),
        (
            Layout::from_strides([2, 3], [0, 1], 2).unwrap(),
            vec![2., 3., 4., 2., 3., 4.],
        ),
        (
            Layout::from_strides([2, 2], [1, 1], 1).unwrap(),
            vec![1., 2., 2., 3.],
        ),
        (Layout::from_strides([], [], 7).unwrap(), vec![7.]),
        (
            Layout::from_strides([3, 0, 5], [5, 1, 2], 8).unwrap(),
            vec![],
        ),
    ];
    for (layout, expected) in cases {
        let owner = Shared::new(Tensor::from_vec([8], (0..8).map(|i| i as f32).collect()).unwrap())
            .unwrap();
        let address = owner.read().unwrap().data().as_ptr();
        let mut layout = Arc::new(layout);
        let saved = owner.save(&layout).unwrap();
        let version = saved.version();
        assert!(std::ptr::eq(saved.layout(), layout.as_ref()));
        let captured = saved.layout().clone();
        *Arc::make_mut(&mut layout) = Layout::contiguous([99]).unwrap();
        drop(owner);
        drop(layout);
        let saved = saved.clone();
        let read = saved.read().unwrap();
        assert_eq!(saved.version(), version);
        assert_eq!(read.storage().data().as_ptr(), address);
        assert_eq!(read.layout(), &captured);
        assert_eq!(
            read.view().unwrap().iter().copied().collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn shared_aliases_and_unique_extraction_preserve_tracking_identity() {
    let mut tracked = Tracked::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    tracked.write().unwrap().data_mut()[0] = 3.;
    let version = tracked.version();
    let address = tracked.read().data().as_ptr();
    let owner = Shared::from_tracked(tracked);
    let alias = owner.clone();
    assert_eq!(owner.version().unwrap(), version);
    let copy = Shared::new(owner.read().unwrap().clone()).unwrap();
    assert_ne!(copy.version().unwrap().storage_id(), version.storage_id());
    let saved = owner
        .save(&Arc::new(Layout::contiguous([2]).unwrap()))
        .unwrap();
    let owner = owner.try_unwrap().err().unwrap();
    drop(alias);
    let owner = owner.try_unwrap().err().unwrap(); // Saved payload is still a strong owner.
    drop(saved);
    let mut tracked = owner.try_unwrap().ok().unwrap();
    assert_eq!(tracked.version(), version);
    assert_eq!(tracked.read().data().as_ptr(), address);
    tracked.write().unwrap().data_mut()[0] = 9.;
    assert_eq!(copy.read().unwrap().data(), &[3., 2.]);
}

#[test]
fn conflicting_access_never_advances_versions_and_one_write_guard_advances_once() {
    let owner = Shared::new(Tensor::zeros([2]).unwrap()).unwrap();
    let alias = owner.clone();
    let layout = Arc::new(Layout::contiguous([2]).unwrap());
    let saved = owner.save(&layout).unwrap();
    let before = saved.version();
    {
        let first = owner.read().unwrap();
        let second = alias.read().unwrap();
        let saved_read = saved.read().unwrap();
        assert_eq!(first.version(), second.version());
        assert_eq!(saved_read.view().unwrap().as_slice(), Some(&[0., 0.][..]));
        assert!(matches!(owner.write(), Err(Error::StorageBorrowConflict)));
        assert_eq!(owner.version().unwrap(), before);
    }
    {
        let mut writer = owner.write().unwrap();
        assert!(matches!(owner.read(), Err(Error::StorageBorrowConflict)));
        assert!(matches!(owner.version(), Err(Error::StorageBorrowConflict)));
        assert!(matches!(alias.write(), Err(Error::StorageBorrowConflict)));
        assert!(matches!(
            owner.save(&layout),
            Err(Error::StorageBorrowConflict)
        ));
        assert!(matches!(saved.read(), Err(Error::StorageBorrowConflict)));
        writer.data_mut()[0] = 7.;
        writer.data_mut()[1] = 9.;
        writer.data_mut()[1] += 2.;
    }
    assert_eq!(
        owner.version().unwrap().generation(),
        before.generation() + 1
    );
    assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
    assert_eq!(owner.read().unwrap().data(), &[7., 11.]);
    assert!(
        Error::StorageBorrowConflict
            .to_string()
            .contains("conflicting active borrows")
    );
}

#[test]
fn unwinding_recovers_valid_partial_storage_but_rejects_old_saved_payload() {
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([2]).unwrap());
    let saved = owner.save(&layout).unwrap();
    let before = saved.version();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut writer = owner.write().unwrap();
        writer.data_mut()[0] = 9.;
        panic!("partial update under the write guard");
    }));
    assert!(result.is_err());
    assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
    assert_eq!(owner.read().unwrap().data(), &[9., 2.]);
    assert_eq!(
        owner.version().unwrap().generation(),
        before.generation() + 1
    );
    owner.write().unwrap().data_mut()[1] = 7.;
    let fresh = owner.save(&layout).unwrap();
    assert_eq!(fresh.read().unwrap().storage().data(), &[9., 7.]);
    drop(saved);
    drop(fresh);
    let tracked = owner.try_unwrap().ok().unwrap();
    assert_eq!(tracked.version().generation(), before.generation() + 2);
    assert_eq!(tracked.read().data(), &[9., 7.]);
}

#[test]
fn concurrent_host_readers_and_a_writer_follow_explicit_borrow_boundaries() {
    let owner = Shared::new(Tensor::from_vec([2], vec![1., 2.]).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([2]).unwrap());
    let saved = owner.save(&layout).unwrap();
    std::thread::scope(|scope| {
        let entered = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        let jobs: Vec<_> = (0..2)
            .map(|_| {
                let saved = saved.clone();
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                scope.spawn(move || {
                    let reader = saved.read().unwrap();
                    entered.wait();
                    release.wait();
                    assert_eq!(reader.storage().data(), &[1., 2.]);
                })
            })
            .collect();
        entered.wait();
        let denied_write = matches!(owner.write(), Err(Error::StorageBorrowConflict));
        let accepted_read = saved.read().is_ok();
        release.wait();
        for job in jobs {
            job.join().unwrap();
        }
        assert!(denied_write);
        assert!(accepted_read);
    });
    std::thread::scope(|scope| {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let writer_owner = owner.clone();
        let writer_entered = Arc::clone(&entered);
        let writer_release = Arc::clone(&release);
        let job = scope.spawn(move || {
            let mut writer = writer_owner.write().unwrap();
            writer.data_mut().fill(5.);
            writer_entered.wait();
            writer_release.wait();
        });
        entered.wait();
        let denied_read = matches!(owner.read(), Err(Error::StorageBorrowConflict));
        let denied_save = matches!(owner.save(&layout), Err(Error::StorageBorrowConflict));
        release.wait();
        job.join().unwrap();
        assert!(denied_read);
        assert!(denied_save);
    });
    assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
    assert_eq!(owner.read().unwrap().data(), &[5., 5.]);
}

#[test]
fn retained_values_feed_prepared_math_and_layout_copies_without_allocations() {
    for cpu in backends() {
        let a =
            Shared::new(Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap()).unwrap();
        let b = Shared::new(Tensor::from_vec([3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap())
            .unwrap();
        let dense_a = a
            .save(&Arc::new(Layout::contiguous([2, 3]).unwrap()))
            .unwrap();
        let dense_b = b
            .save(&Arc::new(Layout::contiguous([3, 2]).unwrap()))
            .unwrap();
        let transpose = Arc::new(
            Layout::contiguous([2, 3])
                .unwrap()
                .permute(&[1, 0])
                .unwrap(),
        );
        let strided_a = a.save(&transpose).unwrap();
        let destination = Layout::contiguous([3, 2]).unwrap();
        let plan = cpu::CopyPlan::new(&transpose, &destination).unwrap();
        drop(a);
        drop(b);
        let output = Shared::new(Tensor::zeros([2, 2]).unwrap()).unwrap();
        let copied = Shared::new(Tensor::zeros([3, 2]).unwrap()).unwrap();
        no_allocations(|| {
            for _ in 0..8 {
                let a = dense_a.read().unwrap();
                let b = dense_b.read().unwrap();
                cpu.matmul(a.storage(), b.storage(), &mut output.write().unwrap())
                    .unwrap();
                let source = strided_a.read().unwrap();
                let mut writer = copied.write().unwrap();
                plan.run(
                    &source.view().unwrap(),
                    &mut writer.view_mut(plan.destination_layout()).unwrap(),
                )
                .unwrap();
            }
        });
        assert_eq!(output.read().unwrap().data(), &[58., 64., 139., 154.]);
        assert_eq!(copied.read().unwrap().data(), &[1., 4., 2., 5., 3., 6.]);
        assert_eq!(output.version().unwrap().generation(), 8);
        assert_eq!(copied.version().unwrap().generation(), 8);
    }
}

#[test]
fn prepared_capture_clone_borrow_and_mutation_checks_do_not_allocate() {
    let owner = Shared::new(Tensor::zeros([4]).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([2, 2]).unwrap());
    no_allocations(|| {
        for _ in 0..100 {
            let alias = owner.clone();
            let saved = alias.save(&layout).unwrap();
            let copy = saved.clone();
            {
                let read = copy.read().unwrap();
                assert_eq!(read.view().unwrap().get(&[0, 0]), Some(&0.0));
                assert!(matches!(owner.write(), Err(Error::StorageBorrowConflict)));
            }
            alias.write().unwrap().data_mut()[0] = 0.;
            assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
        }
    });
    assert_eq!(owner.version().unwrap().generation(), 100);
}

#[test]
fn invalid_capture_and_owner_replacement_never_expose_stale_layouts() {
    let mut values = Vec::with_capacity(16);
    values.resize(4, 1.);
    let owner = Shared::new(Tensor::from_vec([4], values).unwrap()).unwrap();
    let bad = Arc::new(Layout::from_strides([1], [1], 4).unwrap());
    let before = owner.version().unwrap();
    assert!(matches!(owner.save(&bad), Err(Error::Invalid(_))));
    let bad_empty = Arc::new(Layout::from_strides([0], [0], 5).unwrap());
    assert!(owner.save(&bad_empty).is_err());
    assert_eq!(owner.version().unwrap(), before);
    let layout = Arc::new(Layout::contiguous([4]).unwrap());
    let saved = owner.save(&layout).unwrap();
    *owner.write().unwrap() = Tensor::zeros([1]).unwrap();
    assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
    assert!(owner.save(&layout).is_err());
    assert_eq!(owner.read().unwrap().data(), &[0.]);
    let empty = Shared::new(Tensor::zeros([0]).unwrap()).unwrap();
    assert!(
        empty
            .save(&Arc::new(Layout::contiguous([]).unwrap()))
            .is_err()
    );
    let layout = Arc::new(Layout::contiguous([7, 0]).unwrap());
    let saved_empty = empty.save(&layout).unwrap();
    drop(empty);
    assert_eq!(
        saved_empty.read().unwrap().view().unwrap().iter().count(),
        0
    );
}

#[cfg(feature = "cuda")]
#[test]
fn retained_cuda_layouts_survive_external_owner_and_facade_drop() {
    use fast_audio::cuda::{CopyPlan, Cuda};
    let cuda = Cuda::new(0).unwrap();
    let context = cuda.clone();
    let host = Tensor::from_vec([8], (0..8).map(|i| i as f32).collect()).unwrap();
    let owner = Shared::new(cuda.upload(&host).unwrap()).unwrap();
    let layout = Arc::new(Layout::from_strides([2, 3], [-3, -1], 6).unwrap());
    let saved = owner.save(&layout).unwrap();
    let saved = saved.clone(); // DeviceTensor has no Clone implementation.
    let destination = Layout::contiguous([2, 3]).unwrap();
    let plan = CopyPlan::new(&cuda, &layout, &destination).unwrap();
    let output = Shared::new(cuda.zeros([2, 3]).unwrap()).unwrap();
    drop(owner);
    drop(layout);
    drop(cuda);
    no_allocations(|| {
        for _ in 0..8 {
            let source = saved.read().unwrap();
            let mut destination = output.write().unwrap();
            plan.run(
                &source.view().unwrap(),
                &mut destination.view_mut(plan.destination_layout()).unwrap(),
            )
            .unwrap();
        }
    });
    assert_eq!(
        context.download(&output.read().unwrap()).unwrap().data(),
        &[6., 5., 4., 3., 2., 1.]
    );
    let output_layout = Arc::new(Layout::contiguous([2, 3]).unwrap());
    let retained = output.save(&output_layout).unwrap();
    drop(output);
    assert_eq!(
        context
            .download(retained.read().unwrap().storage())
            .unwrap()
            .data(),
        &[6., 5., 4., 3., 2., 1.]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn queued_and_rejected_cuda_writes_invalidate_retained_values_with_scalar_empty_cases() {
    use fast_audio::cuda::{CopyPlan, Cuda};
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let scalar = Tensor::from_vec([], vec![7.]).unwrap();
    let owner = Shared::new(cuda.upload(&scalar).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([]).unwrap());
    let saved = owner.save(&layout).unwrap();
    assert!(
        foreign
            .copy_from_host(&scalar, &mut owner.write().unwrap())
            .is_err()
    );
    assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
    assert_eq!(cuda.download(&owner.read().unwrap()).unwrap().data(), &[7.]);
    let a = cuda
        .upload(&Tensor::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap())
        .unwrap();
    let b = cuda
        .upload(&Tensor::from_vec([2, 2], vec![5., 6., 7., 8.]).unwrap())
        .unwrap();
    let out = Shared::new(cuda.zeros([2, 2]).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([2, 2]).unwrap());
    let before = out.save(&layout).unwrap();
    cuda.matmul(&a, &b, &mut out.write().unwrap()).unwrap();
    assert!(matches!(before.read(), Err(Error::StorageModified { .. })));
    let pending = out.save(&layout).unwrap();
    drop(before);
    drop(out);
    let plan = CopyPlan::new(&cuda, &layout, &layout).unwrap();
    let mut output = cuda.zeros([2, 2]).unwrap();
    {
        let source = pending.read().unwrap();
        plan.run(
            &source.view().unwrap(),
            &mut output.view_mut(plan.destination_layout()).unwrap(),
        )
        .unwrap();
    }
    drop(pending); // Last source owner is released with device work potentially queued.
    assert_eq!(
        cuda.download(&output).unwrap().data(),
        &[19., 22., 43., 50.]
    );
    let empty = Shared::new(cuda.zeros([0]).unwrap()).unwrap();
    let layout = Arc::new(Layout::contiguous([7, 0]).unwrap());
    let saved_empty = empty.save(&layout).unwrap();
    drop(empty);
    assert_eq!(
        saved_empty
            .read()
            .unwrap()
            .view()
            .unwrap()
            .layout()
            .shape()
            .len(),
        0
    );
}
