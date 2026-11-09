use std::env;

fn main() {
    for key in ["PAINTER_CUDA_PROFILE","CUDA_COMPUTE_CAP","PAINTER_CUDA_TOOLKIT_VERSION"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let profile=env::var("PAINTER_CUDA_PROFILE").unwrap_or_else(|_|"optimized".into());
    let expected=match profile.as_str(){"portable"=>"80","optimized"=>"120",_=>panic!("PAINTER_CUDA_PROFILE must be portable or optimized")};
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let actual=env::var("CUDA_COMPUTE_CAP").expect("CUDA workers must use scripts/build.ps1 with an explicit PTX profile");
        assert!(actual==expected || (profile=="optimized" && actual=="120a"),
            "CUDA_COMPUTE_CAP does not match PAINTER_CUDA_PROFILE; refusing a mislabeled worker");
    }
    println!("cargo:rustc-env=PAINTER_CUDA_PROFILE={profile}");
    println!("cargo:rustc-env=PAINTER_EMBEDDED_PTX_TARGET={}",if profile=="portable"{"sm_80"}else{"sm_120a"});
    let toolkit=env::var("PAINTER_CUDA_TOOLKIT_VERSION").unwrap_or_else(|_|"unknown".into());
    println!("cargo:rustc-env=PAINTER_CUDA_TOOLKIT_VERSION={toolkit}");
}
