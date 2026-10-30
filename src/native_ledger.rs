//! Local durable, fail-closed CPU batch reservations for the native GGML path.
//! This is not the shared R53 hardware namespace, a lease, or a distributed lock.
use crate::{invalid, moe, sha256, Error, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct NativeBatchLedger {
    directory: PathBuf,
    handle: File,
}

/// One local reservation, intentionally non-cloneable. This is not an R53 lease.
pub struct Reservation {
    stamp: moe::Stamp,
    attempt: u64,
    directory_id: (u64, u64),
}
impl Reservation {
    pub fn stamp(&self) -> &moe::Stamp {
        &self.stamp
    }
    pub fn attempt(&self) -> u64 {
        self.attempt
    }
    pub(crate) fn directory_id(&self) -> (u64, u64) {
        self.directory_id
    }
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

    // Resolve only against the pinned directory descriptor. Re-resolving the
    // path after an identity check permits a transient directory substitution.
    #[cfg(target_os = "linux")]
    fn create_claim(&self, name: &str) -> Result<File> {
        let fd = rustix::fs::openat(
            &self.handle,
            name,
            rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::from_bits_truncate(0o666),
        )
        .map_err(|e| Error::new("LEDGER_RESERVATION", e.to_string()))?;
        Ok(File::from(fd))
    }

    #[cfg(not(target_os = "linux"))]
    fn create_claim(&self, _name: &str) -> Result<File> {
        Err(invalid(
            "native durable ledger requires Linux pinned directory creation",
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
        let mut file = self.create_claim(&name)?;
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

    /// Reserve a local eviction before transmission. The key excludes `attempt`
    /// and payload digest so a changed attempt or binding cannot replay the same
    /// expert after an uncertain outcome. This is not an R53 lease.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn reserve_eviction(
        &self,
        stamp: &moe::Stamp,
        attempt: u64,
        binding: &crate::native_ggml_cpu::Binding,
    ) -> Result<String> {
        if attempt == 0
            || [&stamp.run_id, &stamp.model_revision, &stamp.plan_id]
                .iter()
                .any(|s| s.trim().is_empty() || s.len() > 256)
            || stamp.model_revision != binding.model_revision
            || stamp.layer != binding.layer
            || stamp.epoch != binding.epoch
            || !sha256::is_hex_digest(&binding.index_sha256)
            || !sha256::is_hex_digest(&binding.blob_sha256)
        {
            return Err(invalid("invalid native eviction reservation"));
        }
        self.check_identity()?;
        let mut key = b"VF_NATIVE_EVICTION_V1\0".to_vec();
        for value in [&stamp.run_id, &stamp.model_revision, &stamp.plan_id] {
            key.extend_from_slice(&(value.len() as u64).to_le_bytes());
            key.extend_from_slice(value.as_bytes());
        }
        for v in [stamp.layer, binding.expert] {
            key.extend_from_slice(&v.to_le_bytes());
        }
        key.extend_from_slice(&stamp.batch.to_le_bytes());
        key.extend_from_slice(&stamp.epoch.to_le_bytes());
        let operation_id = sha256::hex(&key);
        let mut file = self.create_claim(&format!("native-evict-{operation_id}.claim.json"))?;
        self.handle
            .sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.check_identity()?;
        serde_json::to_writer(&mut file, &serde_json::json!({"status":"RESERVED_UNKNOWN_NO_REPLAY", "operation_id":operation_id,"stamp":stamp,"attempt":attempt,"binding":binding}))
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        file.write_all(b"\n")
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        file.sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.handle
            .sync_all()
            .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
        self.check_identity()?;
        Ok(operation_id)
    }

    /// Create the claim first; a failed or abandoned reservation is never reusable.
    pub fn reserve_ticket(&self, stamp: &moe::Stamp, attempt: u64) -> Result<Reservation> {
        self.reserve(stamp, attempt)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = self
                .handle
                .metadata()
                .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
            Ok(Reservation {
                stamp: stamp.clone(),
                attempt,
                directory_id: (meta.dev(), meta.ino()),
            })
        }
        #[cfg(not(unix))]
        {
            Err(invalid("native durable ledger requires Unix"))
        }
    }

    /// Verify that a ticket belongs to this still-pinned directory.
    pub fn validate_ticket(&self, ticket: &Reservation, stamp: &moe::Stamp) -> Result<()> {
        self.check_identity()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = self
                .handle
                .metadata()
                .map_err(|e| Error::new("LEDGER_IO", e.to_string()))?;
            if ticket.stamp != *stamp
                || ticket.attempt == 0
                || ticket.directory_id != (meta.dev(), meta.ino())
            {
                return Err(invalid("native reservation identity mismatch"));
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = (ticket, stamp);
            Err(invalid("native durable ledger requires Unix"))
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod pinned_directory_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn claim_is_created_in_pinned_directory_if_path_is_replaced() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".runs")
            .join(format!(
                "r545-ledger-pinned-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        let moved = root.with_extension("pinned");
        fs::create_dir(&root).unwrap();
        let ledger = NativeBatchLedger::open(&root).unwrap();
        ledger.check_identity().unwrap();
        fs::rename(&root, &moved).unwrap();
        fs::create_dir(&root).unwrap();
        drop(ledger.create_claim("synthetic.claim.json").unwrap());
        let replacement = root.with_extension("replacement");
        fs::rename(&root, &replacement).unwrap();
        fs::rename(&moved, &root).unwrap();
        ledger.check_identity().unwrap();
        assert!(
            root.join("synthetic.claim.json").is_file(),
            "claim escaped pinned directory"
        );
        assert!(!replacement.join("synthetic.claim.json").exists());
    }
}
