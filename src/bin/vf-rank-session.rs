fn main() {
    // This executable is the CPU session worker. Unknown flags must not silently
    // masquerade as an explicitly requested GPU backend before the HELLO frame.
    if std::env::args_os().len() != 1 {
        eprintln!("USAGE: vf-rank-session takes no arguments (CPU worker)");
        std::process::exit(2);
    }
    if let Err(error) = voltafabric::session::session_main() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
