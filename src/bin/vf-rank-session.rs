const REMOTE_MIN_AVAILABLE: u64 = (2 * 1024 + 256) * 1024 * 1024;
const REMOTE_MIN_SWAP: u64 = 1024 * 1024 * 1024;
const MAX_WORKER_EXE_BYTES: u64 = 128 * 1024 * 1024;

#[cfg(unix)]
fn executable_matches(file: &mut std::fs::File, expected: &str) -> std::io::Result<bool> {
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;
    let before = file.metadata()?;
    if !voltafabric::sha256::is_hex_digest(expected)
        || !before.is_file()
        || before.len() == 0
        || before.len() > MAX_WORKER_EXE_BYTES
    {
        return Ok(false);
    }
    let mut bytes = Vec::new();
    (&mut *file)
        .take(MAX_WORKER_EXE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let identity = |m: &std::fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    Ok(identity(&before) == identity(&after)
        && bytes.len() as u64 == before.len()
        && voltafabric::sha256::hex(&bytes) == expected)
}

fn remote_memory_admitted(meminfo: &str) -> bool {
    let mut available = None;
    let mut swap = None;
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        let destination = match fields.next() {
            Some("MemAvailable:") => &mut available,
            Some("SwapFree:") => &mut swap,
            _ => continue,
        };
        if destination.is_some() {
            return false;
        }
        let Some(value) = fields.next() else {
            return false;
        };
        if value.is_empty()
            || !value.bytes().all(|c| c.is_ascii_digit())
            || fields.next() != Some("kB")
            || fields.next().is_some()
        {
            return false;
        }
        let Ok(kib) = value.parse::<u64>() else {
            return false;
        };
        let Some(bytes) = kib.checked_mul(1024) else {
            return false;
        };
        *destination = Some(bytes);
    }
    available.is_some_and(|v| v >= REMOTE_MIN_AVAILABLE)
        && swap.is_some_and(|v| v >= REMOTE_MIN_SWAP)
}

fn main() {
    // This executable is the CPU session worker. Unknown flags must not silently
    // masquerade as an explicitly requested GPU backend before the HELLO frame.
    let args: Vec<_> = std::env::args_os().collect();
    let guarded_remote = args.len() == 4
        && args[1] == "--guarded-remote"
        && args[2] == "--expected-exe-sha256"
        && args[3]
            .to_str()
            .is_some_and(voltafabric::sha256::is_hex_digest);
    if args.len() != 1 && !guarded_remote {
        eprintln!(
            "USAGE: vf-rank-session [--guarded-remote --expected-exe-sha256 SHA256] (CPU worker)"
        );
        std::process::exit(2);
    }
    if guarded_remote
        && !std::fs::read_to_string("/proc/meminfo").is_ok_and(|s| remote_memory_admitted(&s))
    {
        eprintln!("REMOTE_RESOURCE_REFUSED: fresh memory or swap floor not met");
        std::process::exit(2);
    }
    if guarded_remote {
        #[cfg(unix)]
        let valid = std::fs::File::open("/proc/self/exe")
            .and_then(|mut file| executable_matches(&mut file, args[3].to_str().unwrap()))
            .unwrap_or(false);
        #[cfg(not(unix))]
        let valid = false;
        if !valid {
            eprintln!("REMOTE_EXECUTABLE_REFUSED: executable digest mismatch or unreadable");
            std::process::exit(2);
        }
    }
    if let Err(error) = voltafabric::session::session_main() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn remote_executable_digest_rejects_wrong_identity_and_nonfiles() {
        let expected = voltafabric::sha256::hex(&std::fs::read("/proc/self/exe").unwrap());
        let mut file = std::fs::File::open("/proc/self/exe").unwrap();
        assert!(executable_matches(&mut file, &expected).unwrap());
        let mut file = std::fs::File::open("/proc/self/exe").unwrap();
        assert!(!executable_matches(&mut file, &"0".repeat(64)).unwrap());
        let mut file = std::fs::File::open("/proc/self/exe").unwrap();
        assert!(!executable_matches(&mut file, &expected.to_uppercase()).unwrap());
        let mut file = std::fs::File::open("/dev/null").unwrap();
        assert!(!executable_matches(&mut file, &expected).unwrap());
    }
    #[test]
    fn remote_memory_guard_fails_closed_at_the_exact_floors() {
        let good = "MemAvailable: 2359296 kB\nSwapFree: 1048576 kB\n";
        assert!(remote_memory_admitted(good));
        assert!(!remote_memory_admitted(
            "MemAvailable: 2359295 kB\nSwapFree: 1048576 kB\n"
        ));
        assert!(!remote_memory_admitted(
            "MemAvailable: 2359296 kB\nSwapFree: 1048575 kB\n"
        ));
        for bad in [
            "",
            "MemAvailable: 2359296 kB\n",
            "MemAvailable: 2359296 kB\nSwapFree: 1 MB\n",
            "MemAvailable: 2359296 kB\nSwapFree: 1048576 kB\nSwapFree: 1048576 kB\n",
            "MemAvailable: 18446744073709551615 kB\nSwapFree: 1048576 kB\n",
            "MemAvailable: +2359296 kB\nSwapFree: 1048576 kB\n",
        ] {
            assert!(!remote_memory_admitted(bad));
        }
    }
}
