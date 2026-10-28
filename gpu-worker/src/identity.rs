//! Linux loader identity for this native, statically declared C ABI only.
//! The qualified library and its directory must remain trusted and unmodified for
//! the worker lifetime. This is not an arbitrary-plugin or hostile-same-UID sandbox.
use std::{
    ffi::CStr,
    fs::{self, File, Metadata},
    io::Read,
    os::unix::fs::MetadataExt,
};
use voltafabric::{sha256, Error, Result};

fn signature(m: &Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn invalid(message: &str) -> Error {
    Error::new("GPU_LIBRARY_IDENTITY", message)
}

pub struct Identity {
    // Keep the verified mapped inode open, including across a path rename/unlink.
    file: File,
    signature: (u64, u64, u64, i64, i64, i64, i64),
    pub sha256: String,
}
impl Identity {
    pub fn resolve() -> Result<Self> {
        let address = crate::resident::library_symbol() as usize;
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        // SAFETY: address is a linked exported function; info points to writable Dl_info.
        if unsafe { libc::dladdr(address as *const libc::c_void, info.as_mut_ptr()) } == 0 {
            return Err(invalid("cannot resolve native library symbol"));
        }
        // SAFETY: successful dladdr initialized info; filename is a loader-owned C string.
        let info = unsafe { info.assume_init() };
        if info.dli_fname.is_null() {
            return Err(invalid("missing loader filename"));
        }
        let name = unsafe { CStr::from_ptr(info.dli_fname) }
            .to_str()
            .map_err(|_| invalid("non-UTF8 native library path"))?;
        let mut file = File::open(name).map_err(|_| invalid("cannot pin native library inode"))?;
        let before = file
            .metadata()
            .map_err(|_| invalid("native library metadata"))?;
        if !before.is_file() || before.len() > 16 * 1024 * 1024 {
            return Err(invalid(
                "native library exceeds bounded regular-file contract",
            ));
        }
        // Prove the file being hashed is the object covering the resolved address,
        // not merely a replacement at the pathname returned by the loader.
        let maps = fs::read_to_string("/proc/self/maps")
            .map_err(|_| invalid("native mappings unavailable"))?;
        let mapped = maps.lines().any(|line| {
            let fields: Vec<_> = line.split_whitespace().take(5).collect();
            if fields.len() != 5 {
                return false;
            }
            let Some((start, end)) = fields[0].split_once('-') else {
                return false;
            };
            let Some((major, minor)) = fields[3].split_once(':') else {
                return false;
            };
            let Ok(start) = usize::from_str_radix(start, 16) else {
                return false;
            };
            let Ok(end) = usize::from_str_radix(end, 16) else {
                return false;
            };
            let Ok(major) = u32::from_str_radix(major, 16) else {
                return false;
            };
            let Ok(minor) = u32::from_str_radix(minor, 16) else {
                return false;
            };
            start <= address
                && address < end
                && fields[4].parse::<u64>() == Ok(before.ino())
                && major == libc::major(before.dev())
                && minor == libc::minor(before.dev())
        });
        if !mapped {
            return Err(invalid("hashed inode differs from mapped native function"));
        }
        let mut bytes = Vec::new();
        file.by_ref()
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| invalid("native library read"))?;
        let after = file
            .metadata()
            .map_err(|_| invalid("native library metadata"))?;
        let hash = sha256::hex(&bytes);
        if signature(&before) != signature(&after)
            || bytes.len() as u64 != before.len()
            || hash != env!("VF_MXFP4_LIBRARY_SHA256")
        {
            return Err(invalid(
                "native library changed or differs from qualified build hash",
            ));
        }
        Ok(Self {
            file,
            signature: signature(&after),
            sha256: hash,
        })
    }
    pub fn unchanged(&self) -> Result<()> {
        if self
            .file
            .metadata()
            .is_ok_and(|m| signature(&m) == self.signature)
        {
            Ok(())
        } else {
            Err(invalid(
                "pinned native library was modified; worker is terminal",
            ))
        }
    }
}
