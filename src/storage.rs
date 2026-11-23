use crate::{Error, Layout, Result, Shape, Tensor};
use std::{
    ops::{Deref, DerefMut},
    sync::{
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError, TryLockResult,
        atomic::{AtomicU64, Ordering},
    },
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::Tensor {}
    #[cfg(feature = "cuda")]
    impl Sealed for crate::cuda::DeviceTensor {}
}

/// Audited exclusive storage supported by [`Tracked`]. This trait is sealed.
///
/// Shared access must not permit payload mutation, and moving the value must keep
/// exclusive ownership. External interior-mutable storage cannot satisfy this contract.
///
/// ```compile_fail
/// use fast_audio::{Shape, Trackable};
/// struct ExternalStorage;
/// impl Trackable for ExternalStorage {
///     fn shape(&self) -> &Shape { unimplemented!() }
/// }
/// ```
pub trait Trackable: sealed::Sealed {
    fn shape(&self) -> &Shape;
}
impl Trackable for Tensor {
    fn shape(&self) -> &Shape {
        Tensor::shape(self)
    }
}
#[cfg(feature = "cuda")]
impl Trackable for crate::cuda::DeviceTensor {
    fn shape(&self) -> &Shape {
        crate::cuda::DeviceTensor::shape(self)
    }
}

/// Process-local identity of one tracked ownership lifetime; never a payload address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StorageId(u64);

/// Saved identity and potential-write generation. Does not retain storage or layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StorageVersion {
    storage: StorageId,
    generation: u64,
}
impl StorageVersion {
    pub const fn storage_id(self) -> StorageId {
        self.storage
    }
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn reserve_id(counter: &AtomicU64) -> Result<StorageId> {
    let mut id = counter.load(Ordering::Relaxed);
    loop {
        let next = id.checked_add(1).ok_or(Error::StorageIdentityExhausted)?;
        // Only uniqueness is required: publishing the owner follows normal Rust
        // ownership/synchronization, rather than this construction-only counter.
        match counter.compare_exchange_weak(id, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(StorageId(id)),
            Err(actual) => id = actual,
        }
    }
}

/// Opt-in versioned ownership. Ordinary inference storage carries no tracking state.
///
/// ```
/// use fast_audio::{Error, Tensor, Tracked};
/// let mut weights = Tracked::new(Tensor::zeros([2, 2])?)?;
/// let saved = weights.version();
/// weights.write()?.data_mut().fill(1.0);
/// assert!(matches!(weights.check_version(saved),
///     Err(Error::StorageModified { .. })));
/// # Ok::<(), fast_audio::Error>(())
/// ```
/// Shared reads cannot bypass tracking:
/// ```compile_fail
/// use fast_audio::{Tensor, Tracked};
/// let weights = Tracked::new(Tensor::zeros([4]).unwrap()).unwrap();
/// weights.read().data_mut().fill(1.0);
/// ```
/// A retained view also prevents another exclusive access:
/// ```compile_fail
/// use fast_audio::{Layout, Tensor, Tracked};
/// let mut weights = Tracked::new(Tensor::zeros([4]).unwrap()).unwrap();
/// let layout = Layout::contiguous([4]).unwrap();
/// let view = weights.read().view(&layout).unwrap();
/// weights.write().unwrap().data_mut().fill(1.0);
/// assert_eq!(view.get(&[0]), Some(&0.0));
/// ```
#[derive(Debug)]
pub struct Tracked<T: Trackable> {
    value: T,
    version: StorageVersion,
}
impl<T: Trackable> Tracked<T> {
    /// Move existing storage into a fresh tracking lifetime, without copying payloads.
    pub fn new(value: T) -> Result<Self> {
        Ok(Self {
            value,
            version: StorageVersion {
                storage: reserve_id(&NEXT_ID)?,
                generation: 0,
            },
        })
    }
    #[inline]
    pub fn read(&self) -> &T {
        &self.value
    }
    /// Invalidate saved versions before exposing possible writes, including no-ops,
    /// replacement, errors or unwinding. CUDA completion is a separate concern.
    #[inline]
    pub fn write(&mut self) -> Result<&mut T> {
        self.version.generation = self
            .version
            .generation
            .checked_add(1)
            .ok_or(Error::StorageGenerationExhausted)?;
        Ok(&mut self.value)
    }
    #[inline]
    pub fn version(&self) -> StorageVersion {
        self.version
    }
    /// Validate identity and generation together before borrowing a saved value.
    #[inline]
    pub fn check_version(&self, saved: StorageVersion) -> Result<&T> {
        if saved.storage != self.version.storage {
            Err(Error::StorageIdentityMismatch {
                saved: saved.storage,
                current: self.version.storage,
            })
        } else if saved.generation != self.version.generation {
            Err(Error::StorageModified {
                storage: saved.storage,
                saved: saved.generation,
                current: self.version.generation,
            })
        } else {
            Ok(&self.value)
        }
    }
    /// Retire this tracking lifetime. Rewrapping the value assigns a fresh identity.
    pub fn into_inner(self) -> T {
        self.value
    }
}

fn acquire<G>(attempt: TryLockResult<G>) -> Result<G> {
    match attempt {
        Ok(guard) => Ok(guard),
        Err(TryLockError::WouldBlock) => Err(Error::StorageBorrowConflict),
        // Sealed storage remains a valid owned Rust value after unwinding. A writer
        // already advanced its generation before payload access, so old saves fail
        // validation. Permit inspection of partial updates without claiming rollback.
        Err(TryLockError::Poisoned(error)) => Ok(error.into_inner()),
    }
}

/// Shared, versioned ownership with nonblocking read/write acquisition.
///
/// ```
/// use fast_audio::{Error, Layout, Shared, Tensor};
/// use std::sync::Arc;
/// let owner = Shared::new(Tensor::from_vec([2], vec![2., 3.])?)?;
/// let alias = owner.clone();
/// let layout = Arc::new(Layout::contiguous([2])?);
/// let saved = owner.save(&layout)?;
/// drop(owner);
/// {
///     let read = saved.read()?;
///     assert_eq!(read.view()?.as_slice(), Some(&[2., 3.][..]));
/// }
/// alias.write()?.data_mut()[0] = 9.;
/// assert!(matches!(saved.read(), Err(Error::StorageModified { .. })));
/// # Ok::<(), fast_audio::Error>(())
/// ```
#[derive(Debug)]
pub struct Shared<T: Trackable> {
    inner: Arc<RwLock<Tracked<T>>>,
}
impl<T: Trackable> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
impl<T: Trackable> Shared<T> {
    pub fn new(value: T) -> Result<Self> {
        Ok(Self::from_tracked(Tracked::new(value)?))
    }
    pub fn from_tracked(value: Tracked<T>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
        }
    }
    pub fn read(&self) -> Result<ReadGuard<'_, T>> {
        Ok(ReadGuard {
            inner: acquire(self.inner.try_read())?,
        })
    }
    pub fn write(&self) -> Result<WriteGuard<'_, T>> {
        let mut inner = acquire(self.inner.try_write())?;
        inner.write()?;
        Ok(WriteGuard { inner })
    }
    pub(crate) fn check_writable(&self) -> Result<()> {
        let inner = acquire(self.inner.try_write())?;
        if inner.version().generation() == u64::MAX {
            return Err(Error::StorageGenerationExhausted);
        }
        Ok(())
    }
    pub(crate) fn write_checked(&self, saved: StorageVersion) -> Result<WriteGuard<'_, T>> {
        let mut inner = acquire(self.inner.try_write())?;
        inner.check_version(saved)?;
        inner.write()?;
        Ok(WriteGuard { inner })
    }
    pub fn version(&self) -> Result<StorageVersion> {
        Ok(self.read()?.version())
    }
    pub fn save(&self, layout: &Arc<Layout>) -> Result<SavedValue<T>> {
        let read = self.read()?;
        layout.validate_storage(read.shape().len())?;
        Ok(SavedValue {
            owner: self.clone(),
            layout: Arc::clone(layout),
            version: read.version(),
        })
    }
    /// Recover unique tracked ownership without resetting identity or generation.
    pub fn try_unwrap(self) -> std::result::Result<Tracked<T>, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(lock) => Ok(lock
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)),
            Err(inner) => Err(Self { inner }),
        }
    }
}

/// Shared payload access; references cannot escape this guard.
///
/// ```compile_fail
/// use fast_audio::{Shared, Tensor};
/// let owner = Shared::new(Tensor::zeros([2]).unwrap()).unwrap();
/// let data = { let read = owner.read().unwrap(); read.data() };
/// assert_eq!(data[0], 0.0);
/// ```
pub struct ReadGuard<'a, T: Trackable> {
    inner: RwLockReadGuard<'a, Tracked<T>>,
}
impl<T: Trackable> ReadGuard<'_, T> {
    pub fn version(&self) -> StorageVersion {
        self.inner.version()
    }
}
impl<T: Trackable> Deref for ReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.read()
    }
}

/// Exclusive access with one generation advance before construction.
///
/// ```compile_fail
/// use fast_audio::{Shared, Tensor};
/// let owner = Shared::new(Tensor::zeros([2]).unwrap()).unwrap();
/// let element = { let mut write = owner.write().unwrap(); &mut write.data_mut()[0] };
/// *element = 1.0;
/// ```
pub struct WriteGuard<'a, T: Trackable> {
    inner: RwLockWriteGuard<'a, Tracked<T>>,
}
impl<T: Trackable> Deref for WriteGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.read()
    }
}
impl<T: Trackable> DerefMut for WriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // The private constructor advanced exactly once. Do not expose its version
        // while this guard may still write: saves/versions require a separate reader.
        &mut self.inner.value
    }
}

/// Retained payload owner, immutable prepared layout and captured modification version.
#[derive(Debug)]
pub struct SavedValue<T: Trackable> {
    owner: Shared<T>,
    layout: Arc<Layout>,
    version: StorageVersion,
}
impl<T: Trackable> Clone for SavedValue<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            layout: Arc::clone(&self.layout),
            version: self.version,
        }
    }
}
impl<T: Trackable> SavedValue<T> {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn version(&self) -> StorageVersion {
        self.version
    }
    pub fn read(&self) -> Result<SavedRead<'_, T>> {
        let storage = self.owner.read()?;
        storage.inner.check_version(self.version)?;
        Ok(SavedRead {
            storage,
            layout: &self.layout,
        })
    }
}

/// Checked saved-value access. A derived view cannot outlive this guard.
///
/// ```compile_fail
/// use fast_audio::{Layout, Shared, Tensor};
/// use std::sync::Arc;
/// let owner = Shared::new(Tensor::zeros([2]).unwrap()).unwrap();
/// let saved = owner.save(&Arc::new(Layout::contiguous([2]).unwrap())).unwrap();
/// let view = saved.read().unwrap().view().unwrap();
/// assert_eq!(view.get(&[0]), Some(&0.0));
/// ```
pub struct SavedRead<'a, T: Trackable> {
    storage: ReadGuard<'a, T>,
    layout: &'a Layout,
}
impl<T: Trackable> SavedRead<'_, T> {
    pub fn storage(&self) -> &T {
        &self.storage
    }
    pub fn layout(&self) -> &Layout {
        self.layout
    }
}
impl SavedRead<'_, Tensor> {
    pub fn view(&self) -> Result<crate::TensorView<'_>> {
        self.storage.view(self.layout)
    }
}
#[cfg(feature = "cuda")]
/// Retained CUDA storage preserves its original host-thread/session restriction.
///
/// ```compile_fail
/// use fast_audio::{Shared, cuda::Cuda};
/// let cuda = Cuda::new(0).unwrap();
/// let owner = Shared::new(cuda.zeros([2]).unwrap()).unwrap();
/// std::thread::spawn(move || drop(owner));
/// ```
impl SavedRead<'_, crate::cuda::DeviceTensor> {
    pub fn view(&self) -> Result<crate::cuda::DeviceView<'_>> {
        self.storage.view(self.layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_exhaustion_never_wraps_or_reissues_an_id() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(reserve_id(&counter).unwrap(), StorageId(u64::MAX - 1));
        for _ in 0..3 {
            assert!(matches!(
                reserve_id(&counter),
                Err(Error::StorageIdentityExhausted)
            ));
            assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        }
    }

    #[test]
    fn exhausted_generation_keeps_current_reads_and_rejects_write_access() {
        let mut owner = Tracked::new(Tensor::from_vec([1], vec![7.0]).unwrap()).unwrap();
        owner.version.generation = u64::MAX - 1;
        let before = owner.version();
        owner.write().unwrap().data_mut()[0] = 9.0;
        let last = owner.version();
        assert_eq!(last.generation(), u64::MAX);
        assert!(owner.check_version(before).is_err());
        for _ in 0..3 {
            assert!(matches!(
                owner.write(),
                Err(Error::StorageGenerationExhausted)
            ));
            assert_eq!(owner.version(), last);
            assert_eq!(owner.check_version(last).unwrap().data(), &[9.0]);
        }
    }

    #[test]
    fn shared_write_guard_exposes_last_generation_once_then_rejects_exhaustion() {
        let mut tracked = Tracked::new(Tensor::zeros([1]).unwrap()).unwrap();
        tracked.version.generation = u64::MAX - 1;
        let owner = Shared::from_tracked(tracked);
        {
            let mut write = owner.write().unwrap();
            write.data_mut()[0] = 7.0;
            write.data_mut()[0] += 2.0;
        }
        let saved = owner
            .save(&Arc::new(Layout::contiguous([1]).unwrap()))
            .unwrap();
        assert_eq!(owner.version().unwrap().generation(), u64::MAX);
        assert!(matches!(
            owner.write(),
            Err(Error::StorageGenerationExhausted)
        ));
        assert!(matches!(
            owner.check_writable(),
            Err(Error::StorageGenerationExhausted)
        ));
        assert!(matches!(
            owner.write_checked(saved.version()),
            Err(Error::StorageGenerationExhausted)
        ));
        assert_eq!(saved.read().unwrap().storage().data(), &[9.0]);
    }
}
