//! Managed local project catalog. Merely opening or listing it never writes files.
use crate::{document::Document, history_context::RawHistoryStore, tree_layout::{self, LayoutOptions, TreeDisplayMode, TreeNode}};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, HashSet}, fs::{self, File, Metadata, OpenOptions}, io::Read, path::{Path, PathBuf}, sync::atomic::{AtomicU64, Ordering}, time::{SystemTime, UNIX_EPOCH}};

const PROJECT: &str = "project.pproj";
const METADATA: &str = "metadata.json";
const PREVIEW: &str = "preview.png";
const SESSION_LOCK: &str = ".session.lock";
const METADATA_LIMIT: u64 = 16 * 1024;
pub(crate) const PREVIEW_WIDTH: u32 = 720;
pub(crate) const PREVIEW_HEIGHT: u32 = 280;

#[derive(Clone, Debug)]
pub(crate) struct Library { root: PathBuf }

/// Hold for the entire editing session, not merely one autosave. The operating
/// system releases the lock if a window/process exits; the empty marker is not
/// itself proof of ownership and is deliberately never removed.
#[derive(Debug)]
pub(crate) struct SaveLease { _file: File }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveEntry {
    pub id: String,
    pub display_name: String,
    pub updated_unix_ms: u64,
    pub node_count: usize,
    pub primitive_count: usize,
    #[serde(default)]pub workspace:crate::workspace::WorkspaceState,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LibraryListing {
    pub entries: Vec<SaveEntry>,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataFile {
    schema_version: u32,
    id: String,
    display_name: String,
    updated_unix_ms: u64,
    node_count: usize,
    primitive_count: usize,
    #[serde(default)]workspace:crate::workspace::WorkspaceState,
}
impl MetadataFile {
    fn from_entry(entry: &SaveEntry) -> Self { Self { schema_version: 1, id: entry.id.clone(),
        display_name: entry.display_name.clone(), updated_unix_ms: entry.updated_unix_ms,
        node_count: entry.node_count, primitive_count: entry.primitive_count, workspace:entry.workspace.clone() } }
    fn into_entry(self) -> SaveEntry { SaveEntry { id: self.id, display_name: self.display_name,
        updated_unix_ms: self.updated_unix_ms, node_count: self.node_count, primitive_count: self.primitive_count, workspace:self.workspace } }
}

impl Library {
    pub(crate) fn at(path: impl Into<PathBuf>) -> Self { Self { root: path.into() } }
    pub(crate) fn system() -> Result<Self, String> {
        crate::resources::user_config_dir().map(|path| Self::at(path.join("saves")))
            .ok_or_else(|| "无法定位当前用户的本地存档目录".into())
    }
    pub(crate) fn root(&self) -> &Path { &self.root }

    pub(crate) fn claim(&self, id: &str) -> Result<SaveLease, String> {
        let directory = self.directory(id)?;
        self.member(id, PROJECT)?;
        let path = directory.join(SESSION_LOCK);
        check_optional_file(&path)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(windows)] {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0);
        }
        let file = options.open(path).map_err(|error| {
            if matches!(error.raw_os_error(), Some(32 | 33)) {
                "这份存档已在另一个窗口中打开，请先关闭那个窗口".to_owned()
            } else { format!("无法独占打开这份存档：{error}") }
        })?;
        #[cfg(unix)] {
            use std::os::fd::AsRawFd;
            extern "C" { fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int; }
            if unsafe { flock(file.as_raw_fd(), 2 | 4) } != 0 {
                return Err("这份存档已在另一个窗口中打开，或无法取得独占权限".into());
            }
        }
        #[cfg(not(any(windows, unix)))] {
            return Err("当前平台尚不支持安全地独占打开存档".into());
        }
        Ok(SaveLease { _file: file })
    }

    pub(crate) fn list(&self) -> Result<LibraryListing, String> {
        match fs::symlink_metadata(&self.root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(LibraryListing::default()),
            Err(error) => return Err(format!("无法读取本地存档目录：{error}")),
            Ok(metadata) => check_directory(&metadata)?,
        }
        let mut listing = LibraryListing::default();
        for item in fs::read_dir(&self.root).map_err(|error| format!("无法浏览本地存档：{error}"))? {
            let item = match item { Ok(item) => item, Err(_) => { listing.warnings.push("有一项存档无法读取".into()); continue; } };
            let Some(id) = item.file_name().to_str().map(str::to_owned) else { continue; };
            // Interrupted new saves remain hidden and are never auto-published.
            if !valid_id(&id) { continue; }
            match self.read_entry(&id) {
                Ok(entry) => listing.entries.push(entry),
                Err(error) => listing.warnings.push(format!("存档 {id}：{error}")),
            }
        }
        listing.entries.sort_by(|a, b| b.updated_unix_ms.cmp(&a.updated_unix_ms).then_with(|| a.id.cmp(&b.id)));
        Ok(listing)
    }

    /// Publish a new stable ID only after all three files have been completed.
    pub(crate) fn create(&self, name: &str, doc: &Document, history: &RawHistoryStore) -> Result<SaveEntry, String> {
        self.create_workspace(name,&crate::workspace::WorkspaceState::default(),doc,history)
    }
    pub(crate) fn create_workspace(&self,name:&str,workspace:&crate::workspace::WorkspaceState,doc:&Document,history:&RawHistoryStore)->Result<SaveEntry,String>{
        workspace.validate()?;
        let name = checked_name(name)?;
        self.create_root()?;
        for _ in 0..32 {
            let id = new_id()?;
            let target = self.root.join(&id);
            if target.try_exists().map_err(|error| format!("无法检查存档位置：{error}"))? { continue; }
            let staging = self.root.join(format!(".pending-{id}"));
            match fs::create_dir(&staging) {
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("无法创建存档临时目录：{error}")),
                Ok(()) => {},
            }
            let mut pending = PendingDirectory { path: staging, published: false };
            let mut entry = make_entry(id, name.clone(), doc)?;entry.workspace=workspace.clone();
            write_complete(&pending.path, &entry, doc, history)?;
            // This destination is a fresh generated ID; no existing save moves.
            fs::rename(&pending.path, &target).map_err(|error| format!("无法发布新存档，临时内容已保留至本次清理：{error}"))?;
            pending.published = true;
            return Ok(entry);
        }
        Err("无法取得独立的存档编号，请重试".into())
    }

    /// Project replacement is durable and atomic. Auxiliary failures are errors,
    /// but cannot discard the newly saved project or the caller's live document.
    pub(crate) fn save(&self, id: &str, name: &str, doc: &Document, history: &RawHistoryStore) -> Result<SaveEntry, String> {
        let workspace=self.read_entry(id)?.workspace;
        self.save_workspace(id,name,&workspace,doc,history)
    }
    pub(crate) fn save_workspace(&self,id:&str,name:&str,workspace:&crate::workspace::WorkspaceState,doc:&Document,history:&RawHistoryStore)->Result<SaveEntry,String>{
        workspace.validate()?;
        let previous = self.read_entry(id)?;
        let mut entry = make_entry(previous.id, checked_name(name)?, doc)?;entry.workspace=workspace.clone();
        entry.updated_unix_ms = entry.updated_unix_ms.max(previous.updated_unix_ms.saturating_add(1));
        let directory = self.directory(id)?;
        check_optional_file(&directory.join(PREVIEW))?;
        write_complete(&directory, &entry, doc, history)?;
        Ok(entry)
    }

    pub(crate) fn update_workspace(&self,id:&str,workspace:&crate::workspace::WorkspaceState)->Result<SaveEntry,String>{
        workspace.validate()?;let mut entry=self.read_entry(id)?;entry.workspace=workspace.clone();
        entry.updated_unix_ms=now_ms()?.max(entry.updated_unix_ms.saturating_add(1));
        write_metadata(&self.directory(id)?.join(METADATA),&entry)?;Ok(entry)
    }
    pub(crate) fn rename(&self, id: &str, name: &str) -> Result<SaveEntry, String> {
        let _lease = self.claim(id)?;
        let mut entry = self.read_entry(id)?;
        entry.display_name = checked_name(name)?;
        entry.updated_unix_ms = now_ms()?.max(entry.updated_unix_ms.saturating_add(1));
        write_metadata(&self.directory(id)?.join(METADATA), &entry)?;
        Ok(entry)
    }

    pub(crate) fn project_path(&self, id: &str) -> Result<PathBuf, String> { self.member(id, PROJECT) }
    pub(crate) fn preview_path(&self, id: &str) -> Result<PathBuf, String> { self.member(id, PREVIEW) }

    fn create_root(&self) -> Result<(), String> {
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) => check_directory(&metadata),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(|error| format!("无法创建本地存档目录：{error}"))?;
                check_directory(&fs::symlink_metadata(&self.root).map_err(|error| error.to_string())?)
            },
            Err(error) => Err(format!("无法访问本地存档目录：{error}")),
        }
    }
    fn directory(&self, id: &str) -> Result<PathBuf, String> {
        if !valid_id(id) { return Err("存档编号无效".into()); }
        check_directory(&fs::symlink_metadata(&self.root).map_err(|_| "本地存档目录不存在")?)?;
        let directory = self.root.join(id);
        check_directory(&fs::symlink_metadata(&directory).map_err(|_| "找不到这份存档")?)?;
        Ok(directory)
    }
    fn member(&self, id: &str, name: &str) -> Result<PathBuf, String> {
        let path = self.directory(id)?.join(name);
        check_file(&fs::symlink_metadata(&path).map_err(|_| "存档缺少所需文件")?)?;
        Ok(path)
    }
    fn read_entry(&self, id: &str) -> Result<SaveEntry, String> {
        let metadata_path = self.member(id, METADATA)?;
        self.member(id, PROJECT)?;
        let file = File::open(metadata_path).map_err(|_| "无法打开存档说明")?;
        let mut bytes = Vec::new();
        file.take(METADATA_LIMIT + 1).read_to_end(&mut bytes).map_err(|_| "无法读取存档说明")?;
        if bytes.len() as u64 > METADATA_LIMIT { return Err("存档说明超过大小上限".into()); }
        let metadata: MetadataFile = serde_json::from_slice(&bytes).map_err(|_| "存档说明损坏")?;
        if metadata.schema_version != 1 || metadata.id != id || metadata.node_count == 0 {
            return Err("存档说明与目录不一致".into());
        }
        checked_name(&metadata.display_name)?;metadata.workspace.validate()?;
        Ok(metadata.into_entry())
    }
}

fn valid_id(id: &str) -> bool {
    let Some(body) = id.strip_prefix("save-") else { return false; };
    let pieces = body.split('-').collect::<Vec<_>>();
    pieces.len() == 3 && pieces.iter().zip([32, 8, 16]).all(|(piece, size)| piece.len() == size
        && piece.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
}
fn new_id() -> Result<String, String> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| "系统时间早于有效存档时间")?.as_nanos();
    Ok(format!("save-{nanos:032x}-{:08x}-{:016x}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)))
}
fn now_ms() -> Result<u64, String> {
    SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| "系统时间无效".to_owned())?
        .as_millis().try_into().map_err(|_| "系统时间超出存档范围".into())
}
fn checked_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.len() > 512 || name.chars().count() > 120 || name.chars().any(char::is_control) {
        return Err("存档名称须为1至120个普通字符".into());
    }
    Ok(name.to_owned())
}
fn make_entry(id: String, name: String, doc: &Document) -> Result<SaveEntry, String> {
    Ok(SaveEntry { id, display_name: name, updated_unix_ms: now_ms()?, node_count: doc.nodes.len(), primitive_count: doc.primitives.len(),workspace:Default::default() })
}
fn is_link(metadata: &Metadata) -> bool {
    if metadata.file_type().is_symlink() { return true; }
    #[cfg(windows)] { use std::os::windows::fs::MetadataExt; if metadata.file_attributes() & 0x400 != 0 { return true; } }
    false
}
fn check_directory(metadata: &Metadata) -> Result<(), String> {
    if is_link(metadata) || !metadata.is_dir() { Err("存档位置必须是普通文件夹，不能是链接".into()) } else { Ok(()) }
}
fn check_file(metadata: &Metadata) -> Result<(), String> {
    if is_link(metadata) || !metadata.is_file() { Err("存档内容必须是普通文件，不能是链接".into()) } else { Ok(()) }
}
fn check_optional_file(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) { Ok(metadata) => check_file(&metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()), Err(_) => Err("无法检查存档文件".into()) }
}
fn write_metadata(path: &Path, entry: &SaveEntry) -> Result<(), String> {
    crate::atomic_file::write_atomic(path, |file| serde_json::to_writer_pretty(file, &MetadataFile::from_entry(entry))
        .map_err(|error| format!("存档说明写入失败：{error}")))
}
fn write_complete(directory: &Path, entry: &SaveEntry, doc: &Document, history: &RawHistoryStore) -> Result<(), String> {
    crate::project_file::save_project(&directory.join(PROJECT), doc, history)?;
    let preview = composite_preview(doc)?;
    crate::atomic_file::write_atomic(&directory.join(PREVIEW), |file| {
        PngEncoder::new(file).write_image(preview.as_raw(), PREVIEW_WIDTH, PREVIEW_HEIGHT, ColorType::Rgb8)
            .map_err(|error| format!("存档预览写入失败：{error}"))
    })?;
    write_metadata(&directory.join(METADATA), entry)
}
struct PendingDirectory { path: PathBuf, published: bool }
impl Drop for PendingDirectory {
    fn drop(&mut self) {
        if self.published { return; }
        // Only files created by this transaction are touched, never arbitrary contents.
        for name in [PROJECT, METADATA, PREVIEW] { let _ = fs::remove_file(self.path.join(name)); }
        let _ = fs::remove_dir(&self.path);
    }
}

/// Left: current native canvas. Right: the actual branching structure, with
/// long chains sampled every fifty nodes and the first step of every fork kept.
pub(crate) fn composite_preview(doc: &Document) -> Result<RgbImage, String> {
    let mut output = RgbImage::from_pixel(PREVIEW_WIDTH, PREVIEW_HEIGHT, Rgb([244, 246, 250]));
    for y in 0..PREVIEW_HEIGHT { for x in PREVIEW_WIDTH / 2..PREVIEW_WIDTH { output.put_pixel(x, y, Rgb([248, 249, 252])); } }
    let canvas_size = PREVIEW_HEIGHT - 32;
    paste_canvas(&mut output, &doc.primitives, (PREVIEW_WIDTH / 2 - canvas_size) / 2, 16, canvas_size);
    for y in 12..PREVIEW_HEIGHT - 12 { output.put_pixel(PREVIEW_WIDTH / 2, y, Rgb([211, 218, 229])); }
    if doc.nodes.is_empty() || doc.current >= doc.nodes.len() { return Err("无法预览无效的状态树".into()); }
    let mut input = Vec::with_capacity(doc.nodes.len());
    for (index, node) in doc.nodes.iter().enumerate() {
        if node.id != index || node.parent.is_some_and(|parent| parent >= index) { return Err("无法预览含无效父链的状态树".into()); }
        input.push(TreeNode { id: index as u64, parent: node.parent.map(|parent| parent as u64) });
    }
    let sampled = tree_layout::build_layout(&input, LayoutOptions { display_mode: TreeDisplayMode::EmphasizeBranches, ..Default::default() });
    let mut retained = sampled.nodes.iter().map(|node| node.id as usize).collect::<HashSet<_>>();
    retained.insert(doc.current);
    for node in &doc.nodes { if node.children.len() > 1 { retained.extend(node.children.iter().copied()); } }
    let mut nearest = vec![None; doc.nodes.len()];
    let mut selected = Vec::with_capacity(retained.len());
    for (index, node) in doc.nodes.iter().enumerate() {
        let parent = node.parent.and_then(|parent| nearest[parent]);
        if retained.contains(&index) { selected.push(TreeNode { id: index as u64, parent }); nearest[index] = Some(index as u64); }
        else { nearest[index] = parent; }
    }
    let options = LayoutOptions { node_width: 34., node_height: 24., horizontal_gap: 14., vertical_gap: 22., ..Default::default() };
    let tree = tree_layout::build_layout(&selected, options);
    let width = tree.bounds.width().max(1.); let height = tree.bounds.height().max(1.);
    // Fit topology along both axes: a wide fan must not collapse to a flat line.
    let scale_x = (312. / width).min(1.7); let scale_y = (232. / height).min(1.7);
    let scale = scale_x.min(scale_y);
    let center = tree.bounds.center();
    let positions = tree.nodes.iter().map(|node| (node.id, (540. + (node.x-center.0)*scale_x, 140. + (node.y-center.1)*scale_y))).collect::<HashMap<_, _>>();
    for edge in &tree.edges {
        if let (Some(&(x0,y0)), Some(&(x1,y1))) = (positions.get(&edge.parent), positions.get(&edge.child)) {
            let mut last = (x0, y0); let middle = (y0+y1)*0.5;
            for step in 1..=20 { let t = step as f32/20.; let u = 1.-t;
                let point = (u*u*u*x0+3.*u*u*t*x0+3.*u*t*t*x1+t*t*t*x1,
                    u*u*u*y0+3.*u*u*t*middle+3.*u*t*t*middle+t*t*t*y1);
                line(&mut output, last, point, [133, 153, 181]); last = point;
            }
        }
    }
    let mut thumbnail_budget = 24;
    for node in &tree.nodes {
        let &(x,y) = positions.get(&node.id).ok_or("状态树预览缺少节点位置")?;
        let half_w = (options.node_width*scale*0.5).max(1.5); let half_h = (options.node_height*scale*0.5).max(1.5);
        let color = if node.id as usize == doc.current { [55, 112, 207] } else { [182, 195, 213] };
        rectangle(&mut output, x-half_w, y-half_h, x+half_w, y+half_h, color);
        if half_w >= 8. && half_h >= 8. && thumbnail_budget > 0 {
            let canvas = doc.materialize(node.id as usize)?; let side = ((half_h*2.-4.) as u32).max(1);
            paste_canvas(&mut output, &canvas, (x-side as f32*0.5) as u32, (y-side as f32*0.5) as u32, side);
            thumbnail_budget -= 1;
        }
    }
    Ok(output)
}
fn paste_canvas(image: &mut RgbImage, primitives: &[crate::document::Primitive], x: u32, y: u32, size: u32) {
    let canvas = crate::thumbnails::render_preview(primitives, size as usize);
    for row in 0..size { for column in 0..size { if x+column < image.width() && y+row < image.height() {
        let pixel = canvas.pixels[(row*size+column) as usize]; image.put_pixel(x+column, y+row, Rgb([pixel.r(),pixel.g(),pixel.b()]));
    } } }
}
fn rectangle(image: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8;3]) {
    for y in (y0.floor().max(0.) as u32)..=(y1.ceil().max(0.) as u32).min(image.height()-1) {
        for x in (x0.floor().max(361.) as u32)..=(x1.ceil().max(361.) as u32).min(image.width()-1) { image.put_pixel(x,y,Rgb(color)); }
    }
}
fn line(image: &mut RgbImage, from: (f32,f32), to: (f32,f32), color: [u8;3]) {
    let steps = ((to.0-from.0).abs().max((to.1-from.1).abs()).ceil() as u32).max(1).min(1024);
    for step in 0..=steps { let t = step as f32/steps as f32; let x = (from.0+(to.0-from.0)*t).round() as i32; let y = (from.1+(to.1-from.1)*t).round() as i32;
        if x>360 && x<image.width() as i32 && y>=0 && y<image.height() as i32 { image.put_pixel(x as u32,y as u32,Rgb(color)); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::{Primitive, ShapeKind}, region_state};
    struct TestDirectory(PathBuf);
    impl TestDirectory { fn new() -> Self { Self(std::env::temp_dir().join(format!("painter-library-test-{}", new_id().unwrap()))) } }
    impl Drop for TestDirectory { fn drop(&mut self) { if self.0.exists() { let _=fs::remove_dir_all(&self.0); } } }
    fn fixture(color: [u8;3]) -> (Document,RawHistoryStore) {
        let mut doc=Document::new();let mut raw=RawHistoryStore::new();
        let node=doc.commit_generated(Primitive::background(color)).unwrap();
        let tokens=doc.primitives[0].to_tokens().unwrap().map(u32::from);
        raw.attach(&doc,node,&[region_state::BOS_RECORD,tokens]).unwrap(); (doc,raw)
    }
    #[test]
    fn empty_listing_and_constructor_never_create_directory_or_save() {
        let directory=TestDirectory::new();let library=Library::at(&directory.0);
        assert!(!library.root().exists());assert!(library.list().unwrap().entries.is_empty());assert!(!library.root().exists());
        assert!(library.project_path("../outside").is_err());assert!(!library.root().exists());
    }
    #[test]
    fn workspace_selection_save_does_not_rewrite_the_tree_or_preview_and_old_metadata_defaults_to_paint(){
        let directory=TestDirectory::new();let library=Library::at(&directory.0);let(doc,raw)=fixture([40,80,120]);
        let entry=library.create("已有树",&doc,&raw).unwrap();
        let project=library.project_path(&entry.id).unwrap();let preview=library.preview_path(&entry.id).unwrap();
        let before=fs::read(&project).unwrap();let before_preview=fs::read(&preview).unwrap();
        let mut workspace=crate::workspace::WorkspaceState::redraw();
        workspace.selection.toggle(region_state::Stage::Fine,7,&doc.primitives,&std::collections::BTreeMap::new()).unwrap();
        let updated=library.update_workspace(&entry.id,&workspace).unwrap();
        assert_eq!(updated.workspace,workspace);assert_eq!(fs::read(project).unwrap(),before);assert_eq!(fs::read(preview).unwrap(),before_preview);
        let meta=directory.0.join(&entry.id).join(METADATA);let mut value:serde_json::Value=serde_json::from_slice(&fs::read(&meta).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("workspace");fs::write(&meta,serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(library.list().unwrap().entries[0].workspace,crate::workspace::WorkspaceState::default());
    }
    #[test]
    fn explicit_create_rename_and_save_preserve_id_and_full_project() {
        let directory=TestDirectory::new();let library=Library::at(&directory.0);let(doc,raw)=fixture([30,60,90]);
        let entry=library.create("我的项目",&doc,&raw).unwrap();assert!(valid_id(&entry.id));
        assert_eq!(library.list().unwrap().entries,vec![entry.clone()]);
        let renamed=library.rename(&entry.id,"不同名字 / 仍不改变路径").unwrap();assert_eq!(renamed.id,entry.id);
        assert_eq!(library.list().unwrap().entries[0].display_name,"不同名字 / 仍不改变路径");
        let(mut loaded,history)=crate::project_file::load_project(&library.project_path(&entry.id).unwrap()).unwrap();
        assert_eq!(loaded.primitives,doc.primitives);loaded.prompt="新草稿".into();
        let updated=library.save(&entry.id,&renamed.display_name,&loaded,&history).unwrap();assert!(updated.updated_unix_ms>renamed.updated_unix_ms);
        assert_eq!(crate::project_file::load_project(&library.project_path(&entry.id).unwrap()).unwrap().0.prompt,"新草稿");
        let png=image::open(library.preview_path(&entry.id).unwrap()).unwrap();assert_eq!((png.width(),png.height()),(720,280));
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(),1);
    }
    #[test]
    fn modification_save_preserves_nonstandard_canvas_and_workflow_without_standard_gate(){
        use crate::document::{OperationSource,PrimitiveStageAssignment,StageTier};
        let directory=TestDirectory::new();let library=Library::at(&directory.0);
        let mut doc=Document::new();let raw=RawHistoryStore::new();
        let mut fine=Primitive::new(ShapeKind::Ellipse,20.,20.,8.,8.,[120,80,40]);
        fine.stage_assignment=Some(PrimitiveStageAssignment{stage:StageTier::Fine,cell:0,queue_order:1});
        doc.commit_canvas("修改模式中的未完成画布",OperationSource::User,
            vec![Primitive::background([40,60,80]),fine]).unwrap();
        doc.prompt="保持修改模式，不替我改动画布。".into();
        assert!(crate::mode_transition::prepare_paint_resume(&doc,&raw,doc.current,&Default::default()).is_err(),
            "this fixture deliberately cannot return to standard mode");
        let mut workspace=crate::workspace::WorkspaceState::redraw();
        workspace.selection.toggle(region_state::Stage::Fine,0,&doc.primitives,&Default::default()).unwrap();
        let entry=library.create_workspace("未完成的修改",&workspace,&doc,&raw).unwrap();
        let updated=library.save_workspace(&entry.id,&entry.display_name,&workspace,&doc,&raw).unwrap();
        assert_eq!(updated.workspace,workspace);
        let (loaded,history)=crate::project_file::load_project(&library.project_path(&entry.id).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&loaded.nodes).unwrap(),serde_json::to_value(&doc.nodes).unwrap());
        assert_eq!(loaded.primitives,doc.primitives);
        assert_eq!(loaded.current,doc.current);assert_eq!(loaded.prompt,doc.prompt);assert_eq!(loaded.settings,doc.settings);
        assert_eq!(library.list().unwrap().entries[0].workspace,workspace);
        assert!(crate::mode_transition::prepare_paint_resume(&loaded,&history,loaded.current,&Default::default()).is_err());
    }
    #[test]
    fn catalog_skips_corrupt_entries_and_never_follows_traversal_or_replaces_other_files() {
        let directory=TestDirectory::new();let library=Library::at(&directory.0);let(doc,raw)=fixture([50,80,110]);
        let good=library.create("good",&doc,&raw).unwrap();let bad=library.create("bad",&doc,&raw).unwrap();
        fs::write(library.root.join(&bad.id).join(METADATA),b"broken").unwrap();
        let listing=library.list().unwrap();assert_eq!(listing.entries,vec![good.clone()]);assert_eq!(listing.warnings.len(),1);
        for id in ["..","../project.pproj","C:\\outside","save-../../outside",""] { assert!(library.project_path(id).is_err());assert!(library.rename(id,"oops").is_err()); }
        let sentinel=directory.0.join("unrelated.txt");fs::write(&sentinel,b"untouched").unwrap();
        assert!(library.create(" ",&doc,&raw).is_err());assert_eq!(fs::read(sentinel).unwrap(),b"untouched");
        assert!(library.rename(&good.id,"a\nb").is_err());assert_eq!(library.list().unwrap().entries[0],good);
    }
    #[test]
    fn composite_is_left_canvas_right_tree_and_wide_forks_stay_in_bounds() {
        let(mut doc,_)=fixture([201,33,55]);let origin=doc.current;
        for index in 0..180 { doc.jump(origin).unwrap();doc.commit_generated(Primitive::new(ShapeKind::Ellipse,index as f32*2.,100.,20.,20.,[40,100,200])).unwrap(); }
        let preview=composite_preview(&doc).unwrap();assert_eq!(preview.dimensions(),(720,280));
        assert_eq!(preview.get_pixel(60,20).0,doc.primitives[0].color);assert_eq!(preview.get_pixel(360,140).0,[211,218,229]);
        let tree_ink=(362..720).flat_map(|x|(0..280).map(move |y|(x,y))).filter(|&(x,y)|preview.get_pixel(x,y).0!=[248,249,252]).count();
        assert!(tree_ink>200,"right panel must show a real, nonempty tree");
        doc.nodes[0].parent=Some(0);assert!(composite_preview(&doc).is_err());
    }
    #[cfg(windows)]
    #[test]
    fn windows_session_lease_excludes_other_windows_and_drop_recovers_without_stale_lock() {
        let directory=TestDirectory::new();let first=Library::at(&directory.0);let second=first.clone();
        let(doc,raw)=fixture([30,60,90]);let entry=first.create("独占存档",&doc,&raw).unwrap();
        let path=directory.0.join(&entry.id).join(SESSION_LOCK);assert!(!path.exists());
        let lease=first.claim(&entry.id).unwrap();assert!(path.is_file());
        assert!(second.claim(&entry.id).is_err());assert!(second.rename(&entry.id,"不能改名").is_err());
        assert_eq!(second.list().unwrap().entries[0].display_name,"独占存档");
        drop(lease);assert!(path.is_file(),"the marker must not be treated as a stale lock");
        let recovered=second.claim(&entry.id).unwrap();assert!(first.claim(&entry.id).is_err());drop(recovered);
        assert_eq!(first.rename(&entry.id,"已恢复").unwrap().display_name,"已恢复");
        assert!(first.claim("../outside").is_err());
    }
}
