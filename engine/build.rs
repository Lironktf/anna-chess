// Embed nets/default.bin into the binary when it exists at build time.
use std::path::Path;
fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net = Path::new(&manifest).join("..").join("nets").join("default.bin");
    println!("cargo:rerun-if-changed={}", net.display());
    println!("cargo:rustc-check-cfg=cfg(embedded_net)");
    if net.exists() {
        println!("cargo:rustc-cfg=embedded_net");
        println!("cargo:rustc-env=EMBEDDED_NET_PATH={}", net.canonicalize().unwrap().display());
    }
}
