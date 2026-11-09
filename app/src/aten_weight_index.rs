//! Native metadata bridge: tensor bytes remain in the user's original ZIP.
//! Candle only parses data.pkl here; no tensor is evaluated or moved to a GPU.
use anyhow::{bail, Context, Result};
use candle_core::{pickle::PthTensors, DType};
use std::{fs::{self, OpenOptions}, io::{BufWriter, Write}, path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering}};

static INDEX_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn string(out: &mut impl Write, text: &str) -> Result<()> {
    let length = u32::try_from(text.len()).context("weight index string is too long")?;
    out.write_all(&length.to_le_bytes())?;
    out.write_all(text.as_bytes())?;
    Ok(())
}

pub fn write_weight_index(model: &Path, destination: &Path) -> Result<usize> {
    let source = fs::metadata(model).context("无法读取原始权重文件")?;
    if !source.is_file() { bail!("权重必须是本地文件"); }
    let loader = match PthTensors::new(model, Some("ema_state_dict")) {
        Ok(loader) if !loader.tensor_infos().is_empty() => loader,
        _ => PthTensors::new(model, Some("model_state_dict"))
            .context("无法解析原始权重的模型张量目录")?,
    };
    let mut entries: Vec<_> = loader.tensor_infos().iter().collect();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    if entries.is_empty() || entries.len() > 100_000 { bail!("权重张量目录数量无效"); }
    let count = entries.len();
    if let Some(parent) = destination.parent() { fs::create_dir_all(parent)?; }
    let file = OpenOptions::new().write(true).create_new(true).open(destination)
        .context("无法创建新的权重元数据目录文件")?;
    let mut out = BufWriter::new(file);
    out.write_all(b"PPATENI1")?;
    out.write_all(&1u32.to_le_bytes())?;
    out.write_all(&(count as u32).to_le_bytes())?;
    out.write_all(&source.len().to_le_bytes())?;
    for (name, entry) in entries {
        let dtype = match entry.dtype { DType::BF16 => 0u32, DType::F32 => 1,
            DType::F16 => 2, DType::F64 => 3, DType::I64 => 4, DType::U8 => 5,
            other => bail!("ATen权重目录暂不支持 {other:?}"), };
        let dims = entry.layout.dims();
        let strides = entry.layout.stride();
        if dims.len() != strides.len() || dims.len() > 16 { bail!("张量维度无效：{name}"); }
        let item_size = entry.dtype.size_in_bytes();
        // The pickle parser stores this offset in BYTES, unlike Tensor layout
        // element offsets. Preserve it exactly rather than multiplying again.
        let byte_offset = entry.layout.start_offset();
        if byte_offset % item_size != 0 { bail!("张量偏移未按数据类型对齐：{name}"); }
        let span = if dims.contains(&0) { 0 } else {
            dims.iter().zip(strides).try_fold(1usize, |span, (&dim, &stride)| {
                dim.saturating_sub(1).checked_mul(stride).and_then(|n| span.checked_add(n))
            }).context("张量跨度溢出")?
        };
        let end = span.checked_mul(item_size).and_then(|n| byte_offset.checked_add(n))
            .context("张量字节范围溢出")?;
        let storage_bytes = entry.storage_size.checked_mul(item_size).context("张量存储长度溢出")?;
        if end > storage_bytes { bail!("张量超出原始存储：{name}"); }
        string(&mut out, name)?;
        string(&mut out, &entry.path.replace('\\', "/"))?;
        out.write_all(&dtype.to_le_bytes())?;
        out.write_all(&(dims.len() as u32).to_le_bytes())?;
        out.write_all(&(entry.storage_size as u64).to_le_bytes())?;
        out.write_all(&(byte_offset as u64).to_le_bytes())?;
        for &dim in dims { out.write_all(&(dim as u64).to_le_bytes())?; }
        for &stride in strides { out.write_all(&(stride as u64).to_le_bytes())?; }
    }
    out.flush()?;
    let after = fs::metadata(model)?;
    if source.len() != after.len() || source.modified().ok() != after.modified().ok() {
        bail!("生成元数据期间权重发生变化，请在保存完成后重新加载");
    }
    Ok(count)
}

pub fn ensure_weight_index(model: &Path) -> Result<PathBuf> {
    let root = if let Some(path) = std::env::var_os("PAINTER_ATEN_CACHE_DIR") {
        let path = PathBuf::from(path);
        if !path.is_absolute() { bail!("ATen缓存目录必须使用绝对路径"); }
        path
    } else if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(path).join("PrimitivePainter").join("Cache").join("aten")
    } else { std::env::temp_dir().join("primitive-painter-aten") };
    fs::create_dir_all(&root)?;
    let sequence = INDEX_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let destination = root.join(format!("weights-{}-{now}-{sequence}.ppateni", std::process::id()));
    write_weight_index(model, &destination)?;
    Ok(destination)
}
