fn main() {
    // Also covers native runtime helper threads that inherit the executable's
    // default stack reservation. This is private to the ATen/HIP GUI flavor.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-arg-bin=primitive-painter-aten-hip=/STACK:16777216");
    }
}
