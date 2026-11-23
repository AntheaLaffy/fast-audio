use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Allocation(String),
    Unsupported(String),
    StorageIdentityMismatch {
        saved: crate::StorageId,
        current: crate::StorageId,
    },
    StorageModified {
        storage: crate::StorageId,
        saved: u64,
        current: u64,
    },
    StorageIdentityExhausted,
    StorageGenerationExhausted,
    StorageBorrowConflict,
    Native {
        backend: &'static str,
        code: i64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid argument: {message}"),
            Self::Allocation(message) => write!(f, "allocation failed: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported capability: {message}"),
            Self::StorageIdentityMismatch { saved, current } => {
                write!(
                    f,
                    "storage identity mismatch: saved {saved:?}, current {current:?}"
                )
            }
            Self::StorageModified {
                storage,
                saved,
                current,
            } => {
                write!(
                    f,
                    "storage {storage:?} modified: saved generation {saved}, current {current}"
                )
            }
            Self::StorageIdentityExhausted => {
                f.write_str("tracked storage identity space exhausted")
            }
            Self::StorageGenerationExhausted => f.write_str("tracked storage generation exhausted"),
            Self::StorageBorrowConflict => f.write_str("storage has conflicting active borrows"),
            Self::Native { backend, code } => write!(f, "{backend} failed (status {code})"),
        }
    }
}
impl std::error::Error for Error {}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[cfg(any(feature = "mkl", feature = "cuda", feature = "onednn"))]
pub(crate) fn native(backend: &'static str, code: i64) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Native { backend, code })
    }
}

pub(crate) fn int(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid("dimension exceeds the backend's i32 limit"))
}
