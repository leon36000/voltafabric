fn main() {
    if std::env::args_os().len() != 1 {
        eprintln!("USAGE: vf-strata-native-bridge takes no arguments");
        std::process::exit(2);
    }
    if let Err(error) = voltafabric::strata_native_bridge::main() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
