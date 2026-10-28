fn main() {
    if let Err(error) = voltafabric::process::worker_main() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
