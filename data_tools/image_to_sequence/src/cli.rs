//! Public command line. It only configures the existing production environment options.
use std::{ffi::OsString,path::PathBuf};
const HELP:&str=r#"POP 2 image-to-sequence converter (512px, packed CSV)
Usage:
  gpu_anime_vectorizer --input IMAGES --output NEW_OUTPUT [OPTIONS]

Required paths (or set SHAPE_RENDERER_INPUT_DIR / SHAPE_RENDERER_OUTPUT_DIR):
  --input DIR           Image folder; use unique file stems
  --output DIR          Dedicated output folder; do not point to input data

Safe execution defaults:
  --images N            Maximum images (default 16)
  --versions N          Independent sequence versions (default 1)
  --gpus N              Hardware GPUs to use (default 1)
  --batch-size N        Resident images per GPU (default 16)
  --dispatch-size N     Images per dispatch (default 16)
  --in-flight N         Pending submissions per GPU (default 1)
  --mode MODE           mixed / normal / perturbed (default mixed)
  --seed N              Base seed (default 20260916)
  -h, --help            Print help without initializing a GPU

The drawing algorithm and native token protocol are unchanged: global 85,
coarse 4x4 x20, fine 8x8 x10, regional RMSE 12.75, at most 896 CSV records.
CLI options override the corresponding SHAPE_RENDERER_* environment variables.
"#;
#[derive(Debug,PartialEq)]
enum Command{Help,Run(Vec<(&'static str,OsString)>)}
fn parse(args:impl IntoIterator<Item=OsString>)->Result<Command,String>{
    let mut args=args.into_iter();let mut values=Vec::new();
    while let Some(flag)=args.next(){
        if flag=="--help"||flag=="-h"{return Ok(Command::Help);}
        let key=match flag.to_str(){
            Some("--input")=>"SHAPE_RENDERER_INPUT_DIR",Some("--output")=>"SHAPE_RENDERER_OUTPUT_DIR",
            Some("--images")=>"SHAPE_RENDERER_IMAGE_LIMIT",Some("--versions")=>"SHAPE_RENDERER_NUM_VERSIONS",
            Some("--gpus")=>"SHAPE_RENDERER_GPU_COUNT",Some("--batch-size")=>"SHAPE_RENDERER_IMAGES_PER_GPU_BATCH",
            Some("--dispatch-size")=>"SHAPE_RENDERER_IMAGES_PER_GPU_DISPATCH",Some("--in-flight")=>"SHAPE_RENDERER_MAX_IN_FLIGHT_SUBMISSIONS",
            Some("--mode")=>"SHAPE_RENDERER_MODE",Some("--seed")=>"SHAPE_RENDERER_SEED",
            _=>return Err(format!("Unknown option: {}. Use --help.",flag.to_string_lossy())),
        };
        let value=args.next().filter(|v|!v.is_empty()).ok_or_else(||format!("{} requires a value",flag.to_string_lossy()))?;
        if key=="SHAPE_RENDERER_MODE" {
            if !matches!(value.to_str(),Some("normal"|"perturbed"|"mixed")){return Err("--mode must be mixed, normal or perturbed".into());}
        } else if !matches!(key,"SHAPE_RENDERER_INPUT_DIR"|"SHAPE_RENDERER_OUTPUT_DIR") {
            let n=value.to_str().and_then(|v|v.parse::<u64>().ok()).ok_or_else(||format!("{} requires a non-negative integer",flag.to_string_lossy()))?;
            if n==0&&key!="SHAPE_RENDERER_SEED"{return Err(format!("{} must be positive",flag.to_string_lossy()));}
        }
        values.push((key,value));
    }
    Ok(Command::Run(values))
}
pub fn configure()->Result<bool,String>{
    match parse(std::env::args_os().skip(1))?{
        Command::Help=>{println!("{HELP}");return Ok(false);},
        Command::Run(values)=>for(key,value)in values{std::env::set_var(key,value);},
    }
    let path=|key:&str|std::env::var_os(key).filter(|v|!v.is_empty()).map(PathBuf::from).ok_or_else(||format!("Explicit input and output folders are required. Missing {key}; use --help."));
    let input=path("SHAPE_RENDERER_INPUT_DIR")?;let output=path("SHAPE_RENDERER_OUTPUT_DIR")?;
    if !input.is_dir(){return Err(format!("Input directory does not exist: {}",input.display()));}
    if output.exists()&&!output.is_dir(){return Err("Output path is not a directory".into());}
    if output.exists()&&std::fs::canonicalize(&input).ok()==std::fs::canonicalize(&output).ok(){return Err("Input and output directories must be different".into());}
    Ok(true)
}
#[cfg(test)]mod tests{
    use super::*;
    fn args(s:&[&str])->Vec<OsString>{s.iter().map(OsString::from).collect()}
    #[test]fn help_and_paths_are_pure_and_unicode_preserving(){
        assert_eq!(parse(args(&["--help"])),Ok(Command::Help));
        let Command::Run(values)=parse(args(&["--input","C:/图像 空格","--output","D:/结果","--gpus","1","--seed","0"])).unwrap()else{panic!()};
        assert_eq!(values[0].1,"C:/图像 空格");assert_eq!(values[3],("SHAPE_RENDERER_SEED","0".into()));
    }
    #[test]fn reject_unknown_missing_and_invalid_execution_settings(){
        for a in [vec!["--unknown"],vec!["--input"],vec!["--images","0"],vec!["--gpus","-1"],vec!["--versions","bad"],vec!["--mode","arbitrary"]]{assert!(parse(args(&a)).is_err(),"{a:?}");}
        assert!(parse(args(&["--images","16","--versions","1","--batch-size","16","--dispatch-size","16","--in-flight","1","--mode","mixed"])).is_ok());
    }
    #[test]fn release_changes_scheduling_defaults_not_drawing_protocol(){
        assert_eq!((crate::DEFAULT_GPU_COUNT,crate::DEFAULT_IMAGE_LIMIT,crate::NUM_VERSIONS),(1,16,1));
        assert_eq!((crate::DEFAULT_IMAGES_PER_GPU_BATCH,crate::DEFAULT_IMAGES_PER_GPU_DISPATCH,crate::DEFAULT_MAX_IN_FLIGHT_SUBMISSIONS),(16,16,1));
        assert_eq!((crate::PHASE_TWO_RANDOM_STEP_MIN,crate::PHASE_TWO_RANDOM_STEP_MAX,crate::DEFAULT_GRID_MAX_SUCCESSFUL_SHAPES),(85,85,20));
        assert_eq!(crate::DEFAULT_MAX_HISTORY_RECORDS+1,896);
    }
}
