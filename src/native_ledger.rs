//! Local durable, fail-closed CPU batch reservations for the native GGML path.
//! This is not the shared R53 hardware namespace, a lease, or a distributed lock.
use crate::{invalid, moe, sha256, Error, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct NativeBatchLedger {
    directory: PathBuf,
    handle: File,
}

impl NativeBatchLedger {
    /// The caller supplies one stable, shared, pre-existing directory on a local
    /// filesystem supporting exclusive create and directory fsync. No paths are
    /// created implicitly, and changing directories can bypass this ledger.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute() {
            return Err(invalid("native ledger path must be absolute"));
        }
        let meta =
            fs::symlink_metadata(path).map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(invalid("native ledger must be a real directory"));
        }
        let directory =
            fs::canonicalize(path).map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        let handle = File::open(&directory).map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        let this = Self { directory, handle };
        this.check_identity()?;
        Ok(this)
    }

    #[cfg(unix)]
    fn check_identity(&self) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        let stored = self
            .handle
            .metadata()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        let current = fs::symlink_metadata(&self.directory)
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        if !current.is_dir()
            || current.file_type().is_symlink()
            || (stored.dev(), stored.ino()) != (current.dev(), current.ino())
        {
            return Err(invalid("native ledger directory identity changed"));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    fn check_identity(&self) -> Result<()> {
        Err(invalid(
            "native durable ledger requires Unix directory fsync",
        ))
    }

    /// Reserve the full stamp before any request is transmitted. Once the file
    /// exists, even an incomplete reservation stays consumed and is never reused.
    pub fn reserve(&self, stamp: &moe::Stamp, attempt: u64) -> Result<()> {
        if attempt == 0
            || [&stamp.run_id, &stamp.model_revision, &stamp.plan_id]
                .iter()
                .any(|s| s.trim().is_empty() || s.len() > 256)
        {
            return Err(invalid("invalid native ledger identity"));
        }
        self.check_identity()?;
        // Pin key encoding independently of JSON and serde versions.
        let mut stamp_bytes = b"VF_NATIVE_BATCH_V1\0".to_vec();
        for value in [&stamp.run_id, &stamp.model_revision, &stamp.plan_id] {
            stamp_bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            stamp_bytes.extend_from_slice(value.as_bytes());
        }
        stamp_bytes.extend_from_slice(&stamp.layer.to_le_bytes());
        stamp_bytes.extend_from_slice(&stamp.batch.to_le_bytes());
        stamp_bytes.extend_from_slice(&stamp.epoch.to_le_bytes());
        let name = format!("native-{}.claim.json", sha256::hex(&stamp_bytes));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(name))
            .map_err(|e| Error::new("LEDGER_RESERVATION", e.to_string()))?;
        // Persist the directory entry before writing; any error leaves it consumed.
        self.handle
            .sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.check_identity()?;
        let record = serde_json::json!({"status":"RESERVED_UNKNOWN_NO_REPLAY", "stamp":stamp, "attempt":attempt});
        serde_json::to_writer(&mut file, &record)
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        file.write_all(b"\n")
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        file.sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.handle
            .sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.check_identity()?;
        Ok(())
    }
}
