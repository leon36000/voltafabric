use std::{env, path::PathBuf};
fn main() {
    assert_eq!(
        env::var("HOST").unwrap(),
        env::var("TARGET").unwrap(),
        "host CPU only"
    );
    println!("cargo:rerun-if-env-changed=VF_GGML_NATIVE_BUILD");
    let base = PathBuf::from(
        env::var_os("VF_GGML_NATIVE_BUILD")
            .expect("set VF_GGML_NATIVE_BUILD to a verified isolated CPU CMake build"),
    );
    if env::var_os("CARGO_FEATURE_SCALAR_REFERENCE").is_some() {
        let lib = base.join("libvf_native_scalar_reference.a");
        println!("cargo:rerun-if-changed={}", lib.display());
        assert!(lib.is_file(), "missing opt-in scalar reference library");
        println!("cargo:rustc-link-search=native={}", base.display());
        println!("cargo:rustc-link-lib=static=vf_native_scalar_reference");
    }
    for (dir, name) in [
        (&base, "vf_native_ggml"),
        (&base.join("ggml/src"), "ggml-cpu"),
        (&base.join("ggml/src"), "ggml-base"),
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            dir.join(format!("lib{name}.a")).display()
        );
        assert!(
            dir.join(format!("lib{name}.a")).is_file(),
            "missing static library {name}"
        );
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
}
