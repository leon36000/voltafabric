use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
fn execute(program: &std::ffi::OsStr, args: &[&std::ffi::OsStr]) {
    let status = Command::new(program)
        .args(args)
        .status()
        .expect("start native CPU build tool");
    assert!(
        status.success(),
        "native CPU build tool failed: {program:?}"
    );
}
fn main() {
    assert_eq!(
        env::var("HOST").unwrap(),
        env::var("TARGET").unwrap(),
        "native CPU ABI validation only"
    );
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let cuda = root.join("../../cuda");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let cc = env::var_os("CC").unwrap_or_else(|| "gcc".into());
    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    let header = cuda.join("qpn2_moe_ffi.h");
    let validator = cuda.join("qpn2_moe_validate.c");
    let probe = root.join("abi_probe.c");
    for variable in ["CC", "AR"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for path in [&header, &validator, &probe, &root.join("build.rs")] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let object = out.join("qpn2_moe_validate.o");
    let archive = out.join("libvf_qpn2_cpu.a");
    let executable = out.join("abi_probe");
    let compile = |source: &Path, target: &Path, object_only: bool| {
        let mut args = vec![
            std::ffi::OsStr::new("-std=c11"),
            std::ffi::OsStr::new("-Wall"),
            std::ffi::OsStr::new("-Wextra"),
            std::ffi::OsStr::new("-Werror"),
            std::ffi::OsStr::new("-pedantic"),
            std::ffi::OsStr::new("-I"),
            cuda.as_os_str(),
        ];
        if object_only {
            args.push(std::ffi::OsStr::new("-c"));
        }
        args.extend([
            source.as_os_str(),
            std::ffi::OsStr::new("-o"),
            target.as_os_str(),
        ]);
        execute(&cc, &args);
    };
    compile(&validator, &object, true);
    execute(
        &ar,
        &[
            std::ffi::OsStr::new("crs"),
            archive.as_os_str(),
            object.as_os_str(),
        ],
    );
    compile(&probe, &executable, false);
    let measured = Command::new(&executable)
        .output()
        .expect("execute independent C ABI probe");
    assert!(measured.status.success(), "independent C ABI probe failed");
    let text = String::from_utf8(measured.stdout).expect("C ABI probe ASCII");
    fs::write(out.join("abi-layout.txt"), &text).unwrap();
    let mut generated = String::new();
    let mut names = Vec::new();
    let mut offsets = Vec::new();
    let mut sizes = Vec::new();
    let mut alignments = Vec::new();
    for line in text.lines() {
        let words: Vec<_> = line.split_whitespace().collect();
        match words[0] {
            "ABI" | "SIZE" | "ALIGN" => {
                generated.push_str(&format!("const C_{}: usize = {};\n", words[0], words[1]))
            }
            "FIELD" => {
                names.push(words[1].to_owned());
                offsets.push(words[2].parse::<usize>().unwrap());
                sizes.push(words[3].parse::<usize>().unwrap());
                alignments.push(words[4].parse::<usize>().unwrap());
            }
            "STATUSES" => generated.push_str(&format!(
                "const C_STATUSES: [i32; 9] = {:?};\n",
                words[1..]
                    .iter()
                    .map(|n| n.parse::<i32>().unwrap())
                    .collect::<Vec<_>>()
            )),
            _ => panic!("unrecognized C ABI probe line"),
        }
    }
    assert_eq!(
        names.len(),
        31,
        "all v1 fields must be independently measured"
    );
    generated.push_str(&format!(
        "#[cfg(test)]\nconst C_NAMES: [&str; 31] = {:?};\n",
        names
    ));
    generated.push_str(&format!("const C_OFFSETS: [usize; 31] = {:?};\n", offsets));
    generated.push_str(&format!("const C_SIZES: [usize; 31] = {:?};\n", sizes));
    generated.push_str(&format!(
        "const C_ALIGNMENTS: [usize; 31] = {:?};\n",
        alignments
    ));
    fs::write(out.join("abi.rs"), generated).unwrap();
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vf_qpn2_cpu");
}
