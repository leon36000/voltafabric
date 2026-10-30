const REMOTE_MIN_AVAILABLE: u64 = (2 * 1024 + 256) * 1024 * 1024;
const REMOTE_MIN_SWAP: u64 = 1024 * 1024 * 1024;

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
    let guarded_remote = args.len() == 2 && args[1] == "--guarded-remote";
    if args.len() != 1 && !guarded_remote {
        eprintln!("USAGE: vf-rank-session [--guarded-remote] (CPU worker)");
        std::process::exit(2);
    }
    if guarded_remote
        && !std::fs::read_to_string("/proc/meminfo").is_ok_and(|s| remote_memory_admitted(&s))
    {
        eprintln!("REMOTE_RESOURCE_REFUSED: fresh memory or swap floor not met");
        std::process::exit(2);
    }
    if let Err(error) = voltafabric::session::session_main() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
