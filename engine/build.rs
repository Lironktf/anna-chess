// Build script: (1) embed nets/default.bin when present; (2) compile the vendored Fathom Syzygy
// prober (engine/csrc/fathom, MIT) into a static library using the system C compiler, with no
// extra crate dependencies.
use std::path::Path;
use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net = Path::new(&manifest).join("..").join("nets").join("default.bin");
    println!("cargo:rerun-if-changed={}", net.display());
    println!("cargo:rustc-check-cfg=cfg(embedded_net)");
    if net.exists() {
        println!("cargo:rustc-cfg=embedded_net");
        println!("cargo:rustc-env=EMBEDDED_NET_PATH={}", net.canonicalize().unwrap().display());
    }

    // Fathom.
    let src_dir = Path::new(&manifest).join("csrc").join("fathom");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out = Path::new(&out_dir);
    for f in ["tbprobe.c", "tbprobe.h", "tbconfig.h", "tbchess.c", "stdendian.h"] {
        println!("cargo:rerun-if-changed={}", src_dir.join(f).display());
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let obj = out.join("tbprobe.o");
    let status = Command::new(&cc)
        .args(["-O3", "-march=native", "-std=gnu11", "-fPIC", "-w", "-c"])
        .arg(format!("-I{}", src_dir.display()))
        .arg(src_dir.join("tbprobe.c"))
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to run the C compiler for Fathom (set CC or install gcc/clang)");
    assert!(status.success(), "Fathom compilation failed");
    let lib = out.join("libfathom.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").args(["rcs"]).arg(&lib).arg(&obj).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fathom");
}
