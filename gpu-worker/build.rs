//! No CUDA compilation here: link exactly the separately qualified native ABI.
#[allow(dead_code)]
#[path = "../src/sha256.rs"]
mod sha256;
use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=VF_MXFP4_NATIVE_DIR");
    println!("cargo:rerun-if-changed=../src/sha256.rs");
    let dir = PathBuf::from(
        env::var_os("VF_MXFP4_NATIVE_DIR")
            .expect("VF_MXFP4_NATIVE_DIR must identify the qualified native R13 ABI"),
    )
    .canonicalize()
    .expect("native directory");
    let library = dir.join("libvf_r13.so");
    let metadata = fs::metadata(&library).expect("qualified library");
    assert!(
        metadata.is_file() && metadata.len() <= 16 * 1024 * 1024,
        "bounded native library"
    );
    let bytes = fs::read(&library).expect("read qualified native library");
    println!("cargo:rerun-if-changed={}", library.display());
    println!(
        "cargo:rustc-env=VF_MXFP4_LIBRARY_SHA256={}",
        sha256::hex(&bytes)
    );
    println!("cargo:rustc-link-search=native={}", dir.display());
    // Relocatable package: the actual native .so is installed alongside the binary.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
