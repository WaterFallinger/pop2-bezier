//! Native primitive data and a delta-backed creative history tree.
//!
//! The model's actual nine-field order is x, y, angle, width, height, shape,
//! red, green, blue. Ordinary shapes use fixed 0.5 alpha; Background is an
//! opaque full-canvas fill regardless of its geometric carrier values.

use crate::acceleration::AccelerationSettings;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
#[path = "project_data.rs"]
pub(crate) mod project_data;

pub const CANVAS_SIZE: f32 = 512.0;
pub const SHAPE_ALPHA: f32 = 0.5;
pub const ATTRIBUTE_OFFSETS: [u16; 9] = [0, 1024, 2048, 2318, 4366, 6414, 6670, 6798, 6926];
pub const ATTRIBUTE_BINS: [u16; 9] = [1024, 1024, 270, 2048, 2048, 256, 128, 128, 128];
pub type NodeId = usize;

/// Editable ownership is application metadata, independent of model tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageTier { Global, Coarse, Fine }

impl StageTier {
    pub fn side(self)->usize {match self {Self::Global=>1,Self::Coarse=>4,Self::Fine=>8}}
    pub fn label(self)->&'static str {match self {Self::Global=>"全图",Self::Coarse=>"粗网格",Self::Fine=>"细网格"}}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PrimitiveStageAssignment {
    pub stage:StageTier,
    pub cell:usize,
    pub queue_order:u64,
}
impl<'de> Deserialize<'de> for PrimitiveStageAssignment {
    fn deserialize<D:serde::Deserializer<'de>>(deserializer:D)->Result<Self,D::Error>{
        #[derive(Deserialize)]#[serde(deny_unknown_fields)]
        struct Fields {stage:StageTier,cell:usize,queue_order:u64}
        let fields=Fields::deserialize(deserializer)?;
        let assignment=Self{stage:fields.stage,cell:fields.cell,queue_order:fields.queue_order};
        assignment.validate().map_err(serde::de::Error::custom)?;Ok(assignment)
    }
}
impl PrimitiveStageAssignment {
    pub fn validate(&self)->Result<(),String>{
        if self.cell>=self.stage.side()*self.stage.side(){return Err("图元阶段所属网格编号无效".into());}
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeKind {
    Background,
    Rectangle,
    Ellipse,
}

impl ShapeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Background => "背景",
            Self::Rectangle => "矩形",
            Self::Ellipse => "椭圆",
        }
    }

    pub fn bin(self) -> u16 {
        match self {
            Self::Background => 0,
            Self::Rectangle => 1,
            Self::Ellipse => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    pub angle_deg: f32,
    pub width: f32,
    pub height: f32,
    pub shape: ShapeKind,
    pub color: [u8; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_assignment:Option<PrimitiveStageAssignment>,
}

impl Primitive {
    /// ID zero means a newly drawn object; the document assigns a stable ID.
    pub fn new(shape: ShapeKind, x: f32, y: f32, width: f32, height: f32, color: [u8; 3]) -> Self {
        Self {
            id: 0,
            x,
            y,
            angle_deg: 0.0,
            width,
            height,
            shape,
            color,
            stage_assignment:None,
        }
    }

    pub fn background(color: [u8; 3]) -> Self {
        Self::new(ShapeKind::Background, 255.5, 255.5, 511.75, 511.75, color)
    }

    /// Project editable geometry to the exact right-open model lattice.
    /// A 90-degree turn is represented by swapping width/height, as in training.
    pub fn normalized(&self) -> Result<Self, String> {
        if let Some(assignment)=self.stage_assignment {assignment.validate()?;}
        if ![self.x, self.y, self.angle_deg, self.width, self.height]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err("图元包含非有限坐标或角度".into());
        }
        let mut p = self.clone();
        p.angle_deg = p.angle_deg.rem_euclid(180.0);
        if p.angle_deg >= 90.0 {
            p.angle_deg -= 90.0;
            std::mem::swap(&mut p.width, &mut p.height);
        }
        p.x = (p.x.clamp(0.0, 511.5) * 2.0).floor() / 2.0;
        p.y = (p.y.clamp(0.0, 511.5) * 2.0).floor() / 2.0;
        p.angle_deg = ((p.angle_deg * 3.0) + 0.00001).floor().min(269.0) / 3.0;
        p.width = (p.width.clamp(0.0, 511.75) * 4.0).floor() / 4.0;
        p.height = (p.height.clamp(0.0, 511.75) * 4.0).floor() / 4.0;
        for channel in &mut p.color {
            *channel = (*channel / 2) * 2;
        }
        Ok(p)
    }

    pub fn to_tokens(&self) -> Result<[u16; 9], String> {
        let p = self.normalized()?;
        let bins = [
            (p.x * 2.0).round() as u16,
            (p.y * 2.0).round() as u16,
            (p.angle_deg * 3.0).round() as u16,
            (p.width * 4.0).round() as u16,
            (p.height * 4.0).round() as u16,
            p.shape.bin(),
            (p.color[0] / 2) as u16,
            (p.color[1] / 2) as u16,
            (p.color[2] / 2) as u16,
        ];
        Ok(std::array::from_fn(|i| ATTRIBUTE_OFFSETS[i] + bins[i]))
    }

    pub fn from_tokens(id: u64, tokens: [u16; 9]) -> Result<Self, String> {
        let mut bins = [0u16; 9];
        for i in 0..9 {
            let bin = tokens[i]
                .checked_sub(ATTRIBUTE_OFFSETS[i])
                .ok_or_else(|| format!("属性 {} token 越界", i))?;
            if bin >= ATTRIBUTE_BINS[i] {
                return Err(format!("属性 {} token 越界", i));
            }
            bins[i] = bin;
        }
        let shape = match bins[5] {
            0 => ShapeKind::Background,
            1 => ShapeKind::Rectangle,
            2 => ShapeKind::Ellipse,
            _ => return Err("区域标记／控制记录不是可编辑图元".into()),
        };
        Ok(Self {
            id,
            x: bins[0] as f32 / 2.0,
            y: bins[1] as f32 / 2.0,
            angle_deg: bins[2] as f32 / 3.0,
            width: bins[3] as f32 / 4.0,
            height: bins[4] as f32 / 4.0,
            shape,
            color: [bins[6] as u8 * 2, bins[7] as u8 * 2, bins[8] as u8 * 2],
            stage_assignment:None,
        })
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        if self.shape == ShapeKind::Background {
            return (0.0..CANVAS_SIZE).contains(&x) && (0.0..CANVAS_SIZE).contains(&y);
        }
        let a = self.angle_deg.to_radians();
        let (sin, cos) = a.sin_cos();
        let dx = x - self.x;
        let dy = y - self.y;
        let lx = dx * cos + dy * sin;
        let ly = -dx * sin + dy * cos;
        let hx = (self.width / 2.0).max(0.125);
        let hy = (self.height / 2.0).max(0.125);
        match self.shape {
            ShapeKind::Ellipse => (lx / hx).powi(2) + (ly / hy).powi(2) <= 1.0,
            _ => lx.abs() <= hx && ly.abs() <= hy,
        }
    }
}

/// The contribution requested by the editor: squared RGB distance from the
/// background times geometric area, without opacity or overlap weighting.
/// Geometry and colors are inspected as supplied and are never quantized here.
pub fn primitive_contribution(primitive: &Primitive, background: [u8; 3]) -> Result<f64, String> {
    if ![primitive.x, primitive.y, primitive.angle_deg, primitive.width, primitive.height]
        .iter().all(|value| value.is_finite())
        || primitive.width < 0.0 || primitive.height < 0.0
    {
        return Err("无法重排序：图元的坐标、角度或尺寸无效".into());
    }
    let area = match primitive.shape {
        ShapeKind::Background => return Ok(0.0),
        ShapeKind::Rectangle => primitive.width as f64 * primitive.height as f64,
        ShapeKind::Ellipse => std::f64::consts::PI * primitive.width as f64 * primitive.height as f64 / 4.0,
    };
    let contrast = primitive.color.iter().zip(background).map(|(&color, background)| {
        let difference = color as f64 - background as f64;
        difference * difference
    }).sum::<f64>();
    Ok(contrast * area)
}

/// Low-level ordering helper for one region/bucket. The production caller must
/// preserve drawing-stage/region boundaries rather than sorting a whole
/// multi-region canvas through this function. Equal contributions retain their
/// previous order; a background is always first and never competes with shapes.
/// When the bucket omits the real canvas background, its caller must supply
/// that background while ordering (or use `primitive_contribution` directly).
/// An independently authored canvas without a background uses white.
pub fn reorder_by_contribution(primitives: &[Primitive]) -> Result<Vec<Primitive>, String> {
    let backgrounds: Vec<_> = primitives.iter().filter(|primitive| primitive.shape == ShapeKind::Background).collect();
    if backgrounds.len() > 1 {
        return Err("无法重排序：画布包含多个背景图元".into());
    }
    let background = backgrounds.first().map(|primitive| primitive.color).unwrap_or([255; 3]);
    let mut ordinary = Vec::with_capacity(primitives.len());
    for primitive in primitives {
        let contribution = primitive_contribution(primitive, background)?;
        if primitive.shape != ShapeKind::Background {
            ordinary.push((primitive, contribution));
        }
    }
    ordinary.sort_by(|left, right| right.1.total_cmp(&left.1));
    let mut ordered = Vec::with_capacity(primitives.len());
    if let Some(background) = backgrounds.first() {
        ordered.push((**background).clone());
    }
    ordered.extend(ordinary.into_iter().map(|(primitive, _)| primitive.clone()));
    Ok(ordered)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingBackend {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl SamplingBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "自动",
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub cfg: f32,
    pub renyi_alpha: f32,
    pub seed: u64,
    #[serde(default = "default_auto_seed")]
    pub auto_seed: bool,
    #[serde(default = "default_parallel_sequences")]
    pub parallel_sequences: usize,
    #[serde(default)]
    pub sampling_backend: SamplingBackend,
    #[serde(default = "default_context_overclock")]
    pub context_overclock: bool,
    #[serde(default)]
    pub pause_at_stage_end: bool,
    #[serde(default, skip_serializing_if = "disabled_half_region_switch")]
    pub half_region_switch: bool,
    #[serde(default)]
    pub auto_reorder_before_generation: bool,
    #[serde(default)]
    pub acceleration: AccelerationSettings,
}

fn default_context_overclock() -> bool {
    true
}

fn default_auto_seed() -> bool { true }

// Omit the disabled extension so otherwise unchanged schema-1 projects keep
// their previous settings representation and remain readable by older builds.
fn disabled_half_region_switch(value:&bool)->bool { !*value }

fn default_parallel_sequences() -> usize {
    8
}

/// Lane zero preserves the requested seed. Other lanes receive distinct,
/// deterministic seeds, including across u64 wraparound.
pub fn parallel_seed(base: u64, lane: usize) -> u64 {
    base.wrapping_add(lane as u64)
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            cfg: 1.0,
            renyi_alpha: 4.0,
            seed: 20261011,
            auto_seed: default_auto_seed(),
            parallel_sequences: default_parallel_sequences(),
            sampling_backend: SamplingBackend::Auto,
            context_overclock: true,
            pause_at_stage_end: false,
            half_region_switch: false,
            auto_reorder_before_generation: false,
            acceleration: AccelerationSettings::default(),
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        if !self.cfg.is_finite() || self.cfg < 0.0 {
            return Err("CFG 必须为有限非负数".into());
        }
        if !self.renyi_alpha.is_finite() || self.renyi_alpha <= 1.0 {
            return Err("Rényi α 必须大于 1 且为有限数".into());
        }
        if self.parallel_sequences == 0 {
            return Err("并行序列数必须至少为 1".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationSource {
    Initial,
    User,
    Model,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Patch {
    Add { index: usize, primitive: Primitive },
    Remove { id: u64 },
    Replace { primitive: Primitive },
    Move { id: u64, index: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub title: String,
    pub source: OperationSource,
    pub prompt: String,
    pub settings: Settings,
    pub patches: Vec<Patch>,
    pub depth: usize,
    pub primitive_count: usize,
}

/// `primitives` is the only materialized working canvas. Every node stores
/// only a parent, delta, and generation metadata; sparse in-memory checkpoints
/// bound replay cost without duplicating a whole canvas for every operation.
pub struct Document {
    pub nodes: Vec<HistoryNode>,
    pub current: NodeId,
    pub primitives: Vec<Primitive>,
    pub prompt: String,
    pub settings: Settings,
    next_primitive_id: u64,
    checkpoints: HashMap<NodeId, Vec<Primitive>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HistoryDeletion {
    pub selected_roots: Vec<NodeId>,
    pub removed_nodes: Vec<NodeId>,
    pub old_to_new: Vec<Option<NodeId>>,
    pub old_current: NodeId,
    pub new_current: NodeId,
    pub current_was_deleted: bool,
    pub root_reset: bool,
}

/// A compact immutable plan. Large patch/checkpoint payloads stay in the live
/// document until the atomic apply moves surviving payloads into their new IDs.
pub struct HistoryDeletionPlan {
    result: HistoryDeletion,
    source_fingerprint: [u8; 32],
    fallback_canvas: Option<Vec<Primitive>>,
    fallback_prompt: Option<String>,
    fallback_settings: Option<Settings>,
}

#[derive(Serialize)]
pub(crate) struct DocumentProjectRef<'a>{
    nodes:&'a [HistoryNode],current:NodeId,primitives:&'a [Primitive],prompt:&'a str,settings:&'a Settings,
    next_primitive_id:u64,checkpoints:Vec<CheckpointRef<'a>>,
}
#[derive(Serialize)]struct CheckpointRef<'a>{node:NodeId,canvas:&'a [Primitive]}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
pub(crate) struct DocumentProjectData{
    nodes:project_data::LimitedVec<project_data::Counted<ProjectNodeData,{project_data::NODES}>,{project_data::MAX_NODES}>,
    current:NodeId,
    primitives:project_data::LimitedVec<project_data::Counted<Primitive,{project_data::CURRENT_PRIMITIVES}>,{project_data::MAX_CANVAS}>,
    #[serde(deserialize_with="project_data::read_text")]prompt:String,
    #[serde(deserialize_with="project_data::strict_settings")]settings:Settings,
    next_primitive_id:u64,
    checkpoints:project_data::LimitedVec<project_data::Counted<CheckpointData,{project_data::CHECKPOINTS}>,{project_data::MAX_NODES}>,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct ProjectNodeData{
    id:NodeId,parent:Option<NodeId>,
    children:project_data::LimitedVec<project_data::Counted<NodeId,{project_data::EDGES}>,{project_data::MAX_NODES}>,
    #[serde(deserialize_with="project_data::read_title")]title:String,source:OperationSource,
    #[serde(deserialize_with="project_data::read_text")]prompt:String,
    #[serde(deserialize_with="project_data::strict_settings")]settings:Settings,
    patches:project_data::LimitedVec<project_data::Counted<Patch,{project_data::PATCHES}>,{project_data::MAX_PATCHES}>,
    depth:usize,primitive_count:usize,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct CheckpointData{
    node:NodeId,canvas:project_data::LimitedVec<project_data::Counted<Primitive,{project_data::CHECKPOINT_PRIMITIVES}>,{project_data::MAX_CANVAS}>,
}

impl HistoryDeletionPlan {
    pub fn result(&self) -> &HistoryDeletion { &self.result }

    pub(crate) fn validate_source(&self, doc: &Document) -> Result<(), String> {
        if self.result.old_to_new.len() != doc.nodes.len() || doc.deletion_fingerprint()? != self.source_fingerprint {
            return Err("状态树或当前编辑内容已变化，请重新选择后删除".into());
        }
        Ok(())
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    pub(crate) fn project_ref(&self)->DocumentProjectRef<'_>{
        let mut checkpoints=self.checkpoints.iter().map(|(&node,canvas)|CheckpointRef{node,canvas}).collect::<Vec<_>>();
        checkpoints.sort_by_key(|checkpoint|checkpoint.node);
        DocumentProjectRef{nodes:&self.nodes,current:self.current,primitives:&self.primitives,prompt:&self.prompt,
            settings:&self.settings,next_primitive_id:self.next_primitive_id,checkpoints}
    }
    pub(crate) fn has_stage_assignments(&self)->bool {
        self.primitives.iter().any(|p|p.stage_assignment.is_some())
            ||self.checkpoints.values().any(|canvas|canvas.iter().any(|p|p.stage_assignment.is_some()))
            ||self.nodes.iter().any(|node|node.patches.iter().any(|patch|match patch{
                Patch::Add{primitive,..}|Patch::Replace{primitive}=>primitive.stage_assignment.is_some(),_=>false}))
    }
    pub(crate) fn from_project_data(data:DocumentProjectData)->Result<Self,String>{
        let mut checkpoints=HashMap::new();
        for entry in data.checkpoints.0{let entry=entry.0;
            if checkpoints.insert(entry.node,entry.canvas.0.into_iter().map(|p|p.0).collect()).is_some(){return Err("项目含重复检查点".into());}}
        Ok(Self{nodes:data.nodes.0.into_iter().map(|node|{let node=node.0;HistoryNode{id:node.id,parent:node.parent,
            children:node.children.0.into_iter().map(|id|id.0).collect(),title:node.title,source:node.source,prompt:node.prompt,
            settings:node.settings,patches:node.patches.0.into_iter().map(|patch|patch.0).collect(),depth:node.depth,
            primitive_count:node.primitive_count}}).collect(),current:data.current,
            primitives:data.primitives.0.into_iter().map(|p|p.0).collect(),prompt:data.prompt,settings:data.settings,
            next_primitive_id:data.next_primitive_id,checkpoints})
    }

    pub(crate) fn validate_project(&self,budget:&mut project_data::Budget,work:&mut project_data::Work)
        ->Result<Vec<project_data::Projection>,String>{
        self.validate_history_structure()?;
        if self.nodes[0].source!=OperationSource::Initial||!self.nodes[0].patches.is_empty()
            ||self.nodes[0].primitive_count!=0||self.next_primitive_id==0||self.next_primitive_id==u64::MAX{return Err("项目初始状态或对象编号无效".into());}
        budget.charge(project_data::NODES,self.nodes.len())?;budget.charge(project_data::CURRENT_PRIMITIVES,self.primitives.len())?;
        budget.charge(project_data::CHECKPOINTS,self.checkpoints.len())?;
        fn settings(settings:&Settings)->Result<(),String>{settings.validate()}
        fn text(b:&mut project_data::Budget,s:&str,max:usize)->Result<(),String>{
            if s.len()>max{return Err("项目文本超过安全上限".into());}b.charge(project_data::TEXT,s.len())}
        settings(&self.settings)?;text(budget,&self.prompt,project_data::MAX_TEXT)?;
        for node in &self.nodes{
            settings(&node.settings)?;text(budget,&node.title,project_data::MAX_TITLE)?;text(budget,&node.prompt,project_data::MAX_TEXT)?;
            budget.charge(project_data::PATCHES,node.patches.len())?;budget.charge(project_data::EDGES,node.children.len())?;
            if node.primitive_count>project_data::MAX_CANVAS{return Err("项目单个画布超过安全上限".into());}
            if node.depth%64==0&&!self.checkpoints.contains_key(&node.id){return Err("项目缺少必要的稀疏检查点".into());}
        }
        for canvas in self.checkpoints.values(){budget.charge(project_data::CHECKPOINT_PRIMITIVES,canvas.len())?;}
        let mut walker=ProjectCanvas::default();let mut projections=vec![project_data::Projection::default();self.nodes.len()];
        struct Frame{node:usize,next:usize,undo:Vec<Patch>}
        let mut stack=vec![Frame{node:0,next:0,undo:Vec::new()}];
        fn state(doc:&Document,node:usize,walker:&ProjectCanvas,projections:&mut[project_data::Projection])->Result<(),String>{
            if walker.canvas.len()!=doc.nodes[node].primitive_count||walker.backgrounds>1
                ||(walker.backgrounds==1&&walker.canvas.first().is_none_or(|p|p.shape!=ShapeKind::Background)){
                return Err("项目画布图元数量、背景或叠放位置无效".into());}
            if doc.checkpoints.get(&node).is_some_and(|checkpoint|checkpoint!=&walker.canvas){return Err("项目检查点与增量历史不一致".into());}
            if node==doc.current&&doc.primitives!=walker.canvas{return Err("项目当前画布与当前状态不一致".into());}
            projections[node]=walker.prefix.last().cloned().unwrap_or_default();Ok(())
        }
        state(self,0,&walker,&mut projections)?;
        while let Some(frame)=stack.last_mut(){
            if let Some(&child)=self.nodes[frame.node].children.get(frame.next){frame.next+=1;
                let mut undo=Vec::new();for patch in &self.nodes[child].patches{undo.push(walker.apply(patch,work)?);}
                state(self,child,&walker,&mut projections)?;stack.push(Frame{node:child,next:0,undo});
            }else{let frame=stack.pop().unwrap();for patch in frame.undo.iter().rev(){walker.apply(patch,work)?;}}
        }
        if walker.maximum_id>=self.next_primitive_id{return Err("项目后续对象编号会与历史对象冲突".into());}
        Ok(projections)
    }

    fn deletion_fingerprint(&self) -> Result<[u8; 32], String> {
        use sha2::{Digest, Sha256};
        struct FingerprintWriter(Sha256);
        impl std::io::Write for FingerprintWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes); Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let mut writer = FingerprintWriter(Sha256::new());
        // Stream references; do not serialize into a second giant byte buffer.
        serde_json::to_writer(&mut writer, &(&self.nodes, self.current, &self.primitives,
            &self.prompt, &self.settings, self.next_primitive_id, &self.checkpoints))
            .map_err(|error| format!("无法校验状态树删除计划：{error}"))?;
        // Preserve distinctions between invalid in-progress scalar values too.
        writer.0.update(self.settings.cfg.to_bits().to_le_bytes());
        writer.0.update(self.settings.renyi_alpha.to_bits().to_le_bytes());
        Ok(writer.0.finalize().into())
    }

    fn validate_history_structure(&self) -> Result<(), String> {
        if self.nodes.is_empty() || self.current >= self.nodes.len() || !self.checkpoints.contains_key(&0) {
            return Err("状态树缺少有效的初始画布或当前节点".into());
        }
        let mut incoming = vec![0u8; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            if node.id != id { return Err("状态节点编号与存储位置不一致".into()); }
            if id == 0 {
                if node.parent.is_some() || node.depth != 0 { return Err("初始状态连接无效".into()); }
            } else {
                let parent = node.parent.ok_or("状态节点缺少父节点")?;
                if parent >= id || self.nodes[parent].depth.checked_add(1) != Some(node.depth) {
                    return Err("状态树父连接或层级无效".into());
                }
            }
            for &child in &node.children {
                if child >= self.nodes.len() || child <= id || incoming[child] != 0
                    || self.nodes[child].parent != Some(id) {
                    return Err("状态树子连接无效".into());
                }
                incoming[child] = 1;
            }
            let mut count = node.parent.map_or(0, |parent| self.nodes[parent].primitive_count);
            for patch in &node.patches {
                count = match patch {
                    Patch::Add { .. } => count.checked_add(1).ok_or("状态图元数量溢出")?,
                    Patch::Remove { .. } => count.checked_sub(1).ok_or("状态删除增量数量无效")?,
                    _ => count,
                };
            }
            if count != node.primitive_count || self.checkpoints.get(&id).is_some_and(|canvas| canvas.len() != count) {
                return Err("状态图元数量或检查点大小不一致".into());
            }
        }
        if incoming[0] != 0 || incoming.iter().skip(1).any(|&count| count != 1) {
            return Err("状态树父子连接不一致".into());
        }
        if self.checkpoints.keys().any(|&node| node >= self.nodes.len()) {
            return Err("状态树检查点引用不存在的节点".into());
        }
        Ok(())
    }

    pub fn plan_delete_subtrees(&self, selected: &[NodeId]) -> Result<HistoryDeletionPlan, String> {
        if selected.is_empty() { return Err("请先选择要删除的状态节点".into()); }
        if selected.iter().any(|&id| id >= self.nodes.len()) { return Err("要删除的状态节点不存在".into()); }
        self.validate_history_structure()?;
        let selected: HashSet<_> = selected.iter().copied().collect();
        let mut removed = vec![false; self.nodes.len()];
        let mut selected_roots = Vec::new();
        // Parents precede children, including all nodes hidden by tree filters.
        for node in &self.nodes {
            let inherited = node.parent.is_some_and(|parent| removed[parent]);
            removed[node.id] = inherited || selected.contains(&node.id);
            if selected.contains(&node.id) && !inherited { selected_roots.push(node.id); }
        }
        let root_reset = removed[0];
        let mut next = 0;
        let old_to_new = removed.iter().map(|&removed| if removed { None } else {
            let id=next;next+=1;Some(id)
        }).collect::<Vec<_>>();
        let current_was_deleted = removed[self.current];
        let mut fallback = self.current;
        while !root_reset && removed[fallback] { fallback = self.nodes[fallback].parent.ok_or("删除后没有可保留的祖先状态")?; }
        let new_current = if root_reset { 0 } else { old_to_new[fallback].ok_or("删除计划缺少当前状态")? };
        let (fallback_canvas, fallback_prompt, fallback_settings) = if current_was_deleted && !root_reset {
            (Some(self.materialize(fallback)?), Some(self.nodes[fallback].prompt.clone()), Some(self.nodes[fallback].settings.clone()))
        } else { (None, None, None) };
        Ok(HistoryDeletionPlan {
            result: HistoryDeletion { selected_roots, removed_nodes: removed.iter().enumerate().filter_map(|(id,&remove)|remove.then_some(id)).collect(),
                old_to_new, old_current:self.current, new_current, current_was_deleted, root_reset },
            source_fingerprint:self.deletion_fingerprint()?, fallback_canvas, fallback_prompt, fallback_settings,
        })
    }

    /// RawHistoryStore is the public transaction coordinator. Everything that
    /// can fail is checked before moving a single live document payload.
    pub(crate) fn apply_delete_subtrees_plan(&mut self, plan: HistoryDeletionPlan) -> Result<HistoryDeletion, String> {
        plan.validate_source(self)?;
        if plan.result.root_reset {
            let mut fresh = Document::new();
            fresh.next_primitive_id = self.next_primitive_id;
            *self = fresh;
            return Ok(plan.result);
        }
        let retained = self.nodes.len() - plan.result.removed_nodes.len();
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(retained).map_err(|_|"内存不足，无法压缩状态树；原历史已保留".to_owned())?;
        let mut checkpoints = HashMap::new();
        checkpoints.try_reserve(self.checkpoints.keys().filter(|&&id|plan.result.old_to_new[id].is_some()).count())
            .map_err(|_|"内存不足，无法更新状态树检查点；原历史已保留".to_owned())?;
        let HistoryDeletionPlan { result, fallback_canvas, fallback_prompt, fallback_settings, .. } = plan;
        for (old_id, mut node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
            let Some(id) = result.old_to_new[old_id] else { continue; };
            node.id = id;
            node.parent = node.parent.map(|parent|result.old_to_new[parent].expect("surviving parent was validated"));
            node.children.retain(|&child| result.old_to_new[child].is_some());
            for child in &mut node.children { *child = result.old_to_new[*child].expect("surviving child was validated"); }
            nodes.push(node);
        }
        for (id, canvas) in std::mem::take(&mut self.checkpoints) {
            if let Some(new_id) = result.old_to_new[id] { checkpoints.insert(new_id,canvas); }
        }
        self.nodes = nodes; self.checkpoints = checkpoints; self.current = result.new_current;
        if result.current_was_deleted {
            self.primitives = fallback_canvas.expect("deleted current fallback was prepared");
            self.prompt = fallback_prompt.expect("deleted current prompt was prepared");
            self.settings = fallback_settings.expect("deleted current settings were prepared");
        }
        Ok(result)
    }

    pub fn new() -> Self {
        let settings = Settings::default();
        Self {
            nodes: vec![HistoryNode {
                id: 0,
                parent: None,
                children: Vec::new(),
                title: "初始画布".into(),
                source: OperationSource::Initial,
                prompt: String::new(),
                settings: settings.clone(),
                patches: Vec::new(),
                depth: 0,
                primitive_count: 0,
            }],
            current: 0,
            primitives: Vec::new(),
            prompt: String::new(),
            settings,
            next_primitive_id: 1,
            checkpoints: HashMap::from([(0, Vec::new())]),
        }
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_primitive_id;
        self.next_primitive_id += 1;
        id
    }

    pub fn current_node(&self) -> &HistoryNode {
        &self.nodes[self.current]
    }

    pub fn commit_canvas(
        &mut self,
        title: impl Into<String>,
        source: OperationSource,
        next: Vec<Primitive>,
    ) -> Result<NodeId, String> {
        self.commit_canvas_impl(title,source,next,false)
    }
    pub fn commit_control_canvas(&mut self,title:impl Into<String>,next:Vec<Primitive>)->Result<NodeId,String>{
        self.commit_canvas_impl(title,OperationSource::User,next,true)
    }
    fn commit_canvas_impl(&mut self,title:impl Into<String>,source:OperationSource,next:Vec<Primitive>,force:bool)->Result<NodeId,String>{
        let normalized=self.preview_canvas(next)?;
        let next_id=normalized.iter().try_fold(self.next_primitive_id,|next,p|p.id.checked_add(1)
            .map(|following|next.max(following)).ok_or("图元 ID 用尽".to_owned()))?;
        let patches = diff_canvas(&self.primitives, &normalized)?;
        if patches.is_empty()&&!force {return Ok(self.current);}
        let parent = self.current;
        let id = self.nodes.len();
        let depth = self.nodes[parent].depth + 1;
        self.nodes.push(HistoryNode {
            id,parent:Some(parent),children:Vec::new(),title:title.into(),source,
            prompt:self.prompt.clone(),settings:self.settings.clone(),patches,depth,primitive_count:normalized.len(),
        });
        self.nodes[parent].children.push(id);self.current=id;self.primitives=normalized;self.next_primitive_id=next_id;
        if depth%64==0 {self.checkpoints.insert(id,self.primitives.clone());}
        Ok(id)
    }

    /// Normalize the exact proposed canvas and allocate prospective stable IDs
    /// without changing the live document, history, draft, or ID counter.
    pub fn preview_canvas(&self,next:Vec<Primitive>)->Result<Vec<Primitive>,String>{
        self.settings.validate()?;
        let mut next_id = self.next_primitive_id;
        let mut ids = HashSet::new();
        ids.try_reserve(next.len()).map_err(|_|"内存不足，无法验证画布对象编号".to_owned())?;
        let mut normalized = Vec::new();normalized.try_reserve_exact(next.len()).map_err(|_|"内存不足，无法预览画布修改".to_owned())?;
        for p in next {
            let mut p = p.normalized()?;
            if p.id == 0 {
                p.id = next_id;
                next_id = next_id.checked_add(1).ok_or("图元 ID 用尽")?;
            }
            if !ids.insert(p.id) {
                return Err("同一画布含重复图元 ID".into());
            }
            next_id = next_id.max(p.id.checked_add(1).ok_or("图元 ID 用尽")?);
            normalized.push(p);
        }
        if normalized
            .iter()
            .filter(|p| p.shape == ShapeKind::Background)
            .count()
            > 1
        {
            return Err("一幅画布只能包含一个原生背景记录".into());
        }
        // Background is intrinsically a bottom layer in the model representation.
        if let Some(index) = normalized
            .iter()
            .position(|p| p.shape == ShapeKind::Background)
        {
            if index != 0 {
                let bg = normalized.remove(index);
                normalized.insert(0, bg);
            }
        }
        Ok(normalized)
    }

    /// Roll back only a freshly committed current user leaf. Existing branches
    /// and the monotonically advanced primitive-ID counter remain untouched.
    pub fn rollback_last_canvas(&mut self,node:NodeId)->Result<(),String>{
        if self.current!=node||self.nodes.len().checked_sub(1)!=Some(node){return Err("只能回滚刚创建且当前选中的最后人工状态".into());}
        let state=self.nodes.get(node).ok_or("人工回滚状态不存在")?;
        if state.source!=OperationSource::User||!state.children.is_empty(){return Err("不能回滚非人工状态或已有后续历史的状态".into());}
        let parent=state.parent.ok_or("初始状态不能作为人工修改回滚")?;
        if parent>=node||self.nodes[parent].children.last()!=Some(&node){return Err("人工回滚状态父连接不一致".into());}
        let canvas=self.materialize(parent)?;
        let mut prompt=String::new();prompt.try_reserve_exact(self.nodes[parent].prompt.len()).map_err(|_|"内存不足，人工修改未回滚".to_owned())?;
        prompt.push_str(&self.nodes[parent].prompt);let settings=self.nodes[parent].settings.clone();
        self.nodes[parent].children.pop();self.nodes.pop();self.checkpoints.remove(&node);
        self.current=parent;self.primitives=canvas;self.prompt=prompt;self.settings=settings;Ok(())
    }

    pub fn commit_generated(&mut self, mut primitive: Primitive) -> Result<NodeId, String> {
        primitive.id = 0; // Model does not choose application object IDs.
        let label = format!(
            "模型添加{} · 第 {} 个图元",
            primitive.shape.label(),
            self.primitives.len() + 1
        );
        let mut next = self.primitives.clone();
        next.push(primitive);
        self.commit_canvas(label, OperationSource::Model, next)
    }

    /// Append a completed model primitive to an explicit parent without
    /// switching the user's selected canvas or overwriting its staged fields.
    /// This is the only document mutation needed by non-displayed lanes.
    pub fn commit_generated_at(
        &mut self,
        parent: NodeId,
        primitive: Primitive,
        prompt: String,
        settings: Settings,
    ) -> Result<NodeId, String> {
        self.commit_generated_at_ordered(parent,primitive,prompt,settings,false)
    }
    pub fn commit_generated_at_ordered(&mut self,parent:NodeId,primitive:Primitive,prompt:String,settings:Settings,stage_ordered:bool)->Result<NodeId,String>{
        settings.validate()?;
        let mut canvas = self.materialize(parent)?;
        let mut primitive = primitive.normalized()?;
        if primitive.shape == ShapeKind::Background
            && canvas.iter().any(|p| p.shape == ShapeKind::Background)
        {
            return Err("分支已有原生背景，不能再生成第二个背景".into());
        }
        let following_id = self
            .next_primitive_id
            .checked_add(1)
            .ok_or("图元 ID 用尽")?;
        primitive.id = self.next_primitive_id;
        let index = if primitive.shape == ShapeKind::Background {
            0
        } else if stage_ordered {
            let side=primitive.stage_assignment.map(|a|a.stage.side()).unwrap_or(1);
            canvas.iter().position(|p|p.stage_assignment.is_some_and(|a|a.stage.side()>side)).unwrap_or(canvas.len())
        } else {
            canvas.len()
        };
        let count = canvas.len() + 1;
        let patch = Patch::Add {
            index,
            primitive: primitive.clone(),
        };
        canvas.insert(index, primitive.clone());
        let node = self.nodes.len();
        let depth = self.nodes[parent].depth + 1;
        self.nodes.push(HistoryNode {
            id: node,
            parent: Some(parent),
            children: Vec::new(),
            title: format!("模型添加{} · 第 {} 个图元", primitive.shape.label(), count),
            source: OperationSource::Model,
            prompt,
            settings,
            patches: vec![patch],
            depth,
            primitive_count: count,
        });
        self.nodes[parent].children.push(node);
        self.next_primitive_id = following_id;
        if depth % 64 == 0 {
            self.checkpoints.insert(node, canvas);
        }
        Ok(node)
    }

    /// Roll back only the just-created, unexposed model leaf if attaching its
    /// validated model context unexpectedly fails. Older branches are untouched.
    pub(crate) fn rollback_last_generated(&mut self, node: NodeId) -> Result<(), String> {
        if self.nodes.len().checked_sub(1) != Some(node) || node == self.current {
            return Err("只能回滚尚未展示的最后一个模型状态".into());
        }
        let state = &self.nodes[node];
        if state.source != OperationSource::Model || !state.children.is_empty() {
            return Err("不能回滚已有后续历史的状态".into());
        }
        let parent = state.parent.ok_or("模型状态缺少父节点")?;
        if self.nodes[parent].children.last() != Some(&node) {
            return Err("模型状态父连接不一致".into());
        }
        let allocated = match state.patches.as_slice() {
            [Patch::Add { primitive, .. }] => primitive.id,
            _ => return Err("回滚目标不是单图元模型增量".into()),
        };
        self.nodes[parent].children.pop();
        self.nodes.pop();
        self.checkpoints.remove(&node);
        if self.next_primitive_id == allocated + 1 {
            self.next_primitive_id = allocated;
        }
        Ok(())
    }

    pub fn materialize(&self, id: NodeId) -> Result<Vec<Primitive>, String> {
        if id >= self.nodes.len() {
            return Err("状态节点不存在".into());
        }
        let mut cursor = id;
        let mut path = Vec::new();
        while !self.checkpoints.contains_key(&cursor) {
            path.push(cursor);
            cursor = self.nodes[cursor].parent.ok_or("历史树缺少初始状态")?;
        }
        let mut canvas = self.checkpoints[&cursor].clone();
        for step in path.into_iter().rev() {
            for patch in &self.nodes[step].patches {
                apply_patch(&mut canvas, patch)?;
            }
        }
        Ok(canvas)
    }

    pub fn jump(&mut self, id: NodeId) -> Result<(), String> {
        let canvas = self.materialize(id)?;
        let node = &self.nodes[id];
        self.prompt.clone_from(&node.prompt);
        self.settings.clone_from(&node.settings);
        self.primitives = canvas;
        self.current = id;
        Ok(())
    }

    pub fn undo(&mut self) -> Result<(), String> {
        if let Some(parent) = self.current_node().parent {
            self.jump(parent)?;
        }
        Ok(())
    }

    /// Redo follows the latest child. Other children remain directly selectable.
    pub fn redo(&mut self) -> Result<(), String> {
        if let Some(child) = self.current_node().children.last().copied() {
            self.jump(child)?;
        }
        Ok(())
    }

    pub fn stored_patch_count(&self) -> usize {
        self.nodes.iter().map(|n| n.patches.len()).sum()
    }
}

#[derive(Default)]struct ProjectCanvas{
    canvas:Vec<Primitive>,positions:HashMap<u64,usize>,prefix:Vec<project_data::Projection>,backgrounds:usize,maximum_id:u64,
}
impl ProjectCanvas{
    fn apply(&mut self,patch:&Patch,work:&mut project_data::Work)->Result<Patch,String>{
        // Bound both validation and the existing later Vec-based materializer's
        // search cost, even for a malicious node with millions of replacements.
        work.charge(self.canvas.len().saturating_add(1))?;
        fn primitive(p:&Primitive)->Result<(),String>{if p.id==0||p.normalized()?!=*p{return Err("项目图元不是有效原生栅格或对象编号为零".into());}Ok(())}
        let (index,undo)=match patch{
            Patch::Add{index,primitive:p}=>{primitive(p)?;
                if *index>self.canvas.len()||self.positions.contains_key(&p.id){return Err("项目添加增量位置或重复对象编号无效".into());}
                self.maximum_id=self.maximum_id.max(p.id);self.backgrounds+=usize::from(p.shape==ShapeKind::Background);
                self.canvas.insert(*index,p.clone());(*index,Patch::Remove{id:p.id})},
            Patch::Remove{id}=>{self.maximum_id=self.maximum_id.max(*id);let index=self.positions.remove(id).ok_or("项目删除增量引用不存在的对象")?;
                let p=self.canvas.remove(index);self.backgrounds-=usize::from(p.shape==ShapeKind::Background);(index,Patch::Add{index,primitive:p})},
            Patch::Replace{primitive:p}=>{primitive(p)?;self.maximum_id=self.maximum_id.max(p.id);
                let index=*self.positions.get(&p.id).ok_or("项目修改增量引用不存在的对象")?;
                let old=std::mem::replace(&mut self.canvas[index],p.clone());
                self.backgrounds=self.backgrounds-usize::from(old.shape==ShapeKind::Background)+usize::from(p.shape==ShapeKind::Background);
                (index,Patch::Replace{primitive:old})},
            Patch::Move{id,index}=>{self.maximum_id=self.maximum_id.max(*id);
                let old=*self.positions.get(id).ok_or("项目移动增量引用不存在的对象")?;
                if *index>=self.canvas.len(){return Err("项目移动增量位置越界".into());}
                let p=self.canvas.remove(old);self.canvas.insert(*index,p);(old.min(*index),Patch::Move{id:*id,index:old})},
        };
        if self.prefix.is_empty(){self.prefix.push(project_data::Projection::default());}
        self.prefix.truncate(index+1);
        for i in index..self.canvas.len(){self.positions.insert(self.canvas[i].id,i);
            let mut hash=self.prefix.last().cloned().unwrap_or_default();hash.push_with_id(&self.canvas[i].to_tokens()?.map(u32::from),self.canvas[i].id)?;self.prefix.push(hash);}
        Ok(undo)
    }
}

fn apply_patch(canvas: &mut Vec<Primitive>, patch: &Patch) -> Result<(), String> {
    match patch {
        Patch::Add { index, primitive } => {
            if *index > canvas.len() || canvas.iter().any(|p| p.id == primitive.id) {
                return Err("无效图元插入增量".into());
            }
            canvas.insert(*index, primitive.clone());
        }
        Patch::Remove { id } => {
            let i = canvas
                .iter()
                .position(|p| p.id == *id)
                .ok_or("删除增量引用不存在的图元")?;
            canvas.remove(i);
        }
        Patch::Replace { primitive } => {
            let target = canvas
                .iter_mut()
                .find(|p| p.id == primitive.id)
                .ok_or("修改增量引用不存在的图元")?;
            *target = primitive.clone();
        }
        Patch::Move { id, index } => {
            let i = canvas
                .iter()
                .position(|p| p.id == *id)
                .ok_or("层次增量引用不存在的图元")?;
            if *index >= canvas.len() {
                return Err("层次增量位置越界".into());
            }
            let p = canvas.remove(i);
            canvas.insert(*index, p);
        }
    }
    Ok(())
}

fn diff_canvas(old: &[Primitive], new: &[Primitive]) -> Result<Vec<Patch>, String> {
    let mut work = old.to_vec();
    let mut patches = Vec::new();
    let target_ids: HashSet<u64> = new.iter().map(|p| p.id).collect();
    for p in old {
        if !target_ids.contains(&p.id) {
            let patch = Patch::Remove { id: p.id };
            apply_patch(&mut work, &patch)?;
            patches.push(patch);
        }
    }
    for (index, p) in new.iter().enumerate() {
        let existing = work.iter().find(|v| v.id == p.id);
        let patch = match existing {
            Some(v) if v != p => Some(Patch::Replace {
                primitive: p.clone(),
            }),
            None => Some(Patch::Add {
                index: index.min(work.len()),
                primitive: p.clone(),
            }),
            _ => None,
        };
        if let Some(patch) = patch {
            apply_patch(&mut work, &patch)?;
            patches.push(patch);
        }
    }
    for (index, p) in new.iter().enumerate() {
        if work[index].id != p.id {
            let patch = Patch::Move { id: p.id, index };
            apply_patch(&mut work, &patch)?;
            patches.push(patch);
        }
    }
    debug_assert_eq!(work, new);
    Ok(patches)
}

/// Attribute predictions remain private until all nine are complete.
/// Dropping/cancelling this buffer cannot change the document or its tree.
#[derive(Default, Debug)]
pub struct PartialPrimitive {
    tokens: Vec<u16>,
}

impl PartialPrimitive {
    pub fn push(&mut self, token: u16) -> Result<(), String> {
        let i = self.tokens.len();
        if i >= 9 {
            return Err("当前图元已完成".into());
        }
        if token < ATTRIBUTE_OFFSETS[i] || token >= ATTRIBUTE_OFFSETS[i] + ATTRIBUTE_BINS[i] {
            return Err(format!("属性 {} token 越界", i));
        }
        self.tokens.push(token);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_complete(&self) -> bool {
        self.tokens.len() == 9
    }
    pub fn cancel(&mut self) {
        self.tokens.clear();
    }
    pub fn finish(self, id: u64) -> Result<Primitive, String> {
        let tokens: [u16; 9] = self
            .tokens
            .try_into()
            .map_err(|_| "图元九个属性尚未完整，不能提交".to_string())?;
        Primitive::from_tokens(id, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deletion_primitive(x: f32) -> Primitive {
        Primitive::new(ShapeKind::Ellipse, x, 100., 20., 12., [40,100,180])
    }
    fn deletion_child(doc:&mut Document,parent:NodeId,title:&str)->NodeId{
        doc.commit_generated_at(parent,deletion_primitive(parent as f32+10.),title.into(),Settings::default()).unwrap()
    }

    #[test]
    fn subtree_deletion_deduplicates_overlaps_compacts_hidden_descendants_and_keeps_draft() {
        let mut doc=Document::new();
        let a=deletion_child(&mut doc,0,"a");let a2=deletion_child(&mut doc,a,"a2");
        let a3=deletion_child(&mut doc,a2,"hidden a3");
        let b=deletion_child(&mut doc,0,"b");let b2=deletion_child(&mut doc,b,"b2");
        let canvases=[(b,doc.materialize(b).unwrap()),(b2,doc.materialize(b2).unwrap())];
        doc.jump(b2).unwrap();doc.prompt="尚未生成的草稿提示词".into();doc.settings.cfg=2.5;
        let working=doc.primitives.clone();let draft=doc.settings.clone();let next_id=doc.next_primitive_id;
        let plan=doc.plan_delete_subtrees(&[a2,a,a,a3]).unwrap();
        assert_eq!(plan.result().selected_roots,[a]);
        assert_eq!(plan.result().removed_nodes,[a,a2,a3]);
        let result=doc.apply_delete_subtrees_plan(plan).unwrap();
        assert_eq!(doc.nodes.len(),3);assert!(!result.current_was_deleted);
        assert_eq!(doc.current,2);assert_eq!(doc.prompt,"尚未生成的草稿提示词");assert_eq!(doc.settings,draft);
        assert_eq!(doc.primitives,working);assert_eq!(doc.next_primitive_id,next_id);
        for(old,canvas)in canvases{assert_eq!(doc.materialize(result.old_to_new[old].unwrap()).unwrap(),canvas);}
        for(id,node)in doc.nodes.iter().enumerate(){assert_eq!(node.id,id);for &child in &node.children{assert_eq!(doc.nodes[child].parent,Some(id));}}
        let generated=doc.commit_generated(deletion_primitive(300.)).unwrap();
        assert_eq!(doc.nodes[generated].parent,Some(2));
        assert_eq!(doc.primitives.last().unwrap().id,next_id);
    }

    #[test]
    fn deleting_current_subtree_returns_to_nearest_surviving_ancestor() {
        let mut doc=Document::new();let a=deletion_child(&mut doc,0,"ancestor");
        let b=deletion_child(&mut doc,a,"delete b");let c=deletion_child(&mut doc,b,"delete c");
        let sibling=deletion_child(&mut doc,a,"sibling");let sibling_canvas=doc.materialize(sibling).unwrap();
        doc.jump(c).unwrap();doc.prompt="discarded current draft".into();doc.settings.cfg=8.;
        let result=doc.apply_delete_subtrees_plan(doc.plan_delete_subtrees(&[b]).unwrap()).unwrap();
        assert!(result.current_was_deleted);assert_eq!(result.old_current,c);assert_eq!(doc.current,a);
        assert_eq!(doc.prompt,"ancestor");assert_eq!(doc.settings,doc.nodes[a].settings);
        assert_eq!(doc.primitives,doc.materialize(a).unwrap());
        assert_eq!(doc.materialize(result.old_to_new[sibling].unwrap()).unwrap(),sibling_canvas);
    }

    #[test]
    fn deleting_root_replaces_entire_tree_but_never_reuses_primitive_ids() {
        let mut doc=Document::new();let a=doc.commit_generated(deletion_primitive(1.)).unwrap();
        let b=doc.commit_generated(deletion_primitive(2.)).unwrap();let next_id=doc.next_primitive_id;
        let result=doc.apply_delete_subtrees_plan(doc.plan_delete_subtrees(&[0,a]).unwrap()).unwrap();
        assert!(result.root_reset&&result.current_was_deleted);assert_eq!(result.removed_nodes,[0,a,b]);
        assert!(result.old_to_new.iter().all(Option::is_none));assert_eq!(doc.nodes.len(),1);
        assert_eq!(doc.current,0);assert!(doc.nodes[0].children.is_empty());assert!(doc.nodes[0].parent.is_none());
        assert!(doc.primitives.is_empty()&&doc.prompt.is_empty());assert_eq!(doc.settings,Settings::default());
        assert_eq!(doc.next_primitive_id,next_id);assert_eq!(doc.checkpoints.len(),1);
        doc.commit_generated(deletion_primitive(3.)).unwrap();assert_eq!(doc.primitives[0].id,next_id);
    }

    #[test]
    fn deletion_invalid_ids_and_stale_topology_or_metadata_never_mutate_document() {
        let mut doc=Document::new();let a=deletion_child(&mut doc,0,"a");let b=deletion_child(&mut doc,a,"b");
        let original=doc.deletion_fingerprint().unwrap();
        for ids in [vec![],vec![doc.nodes.len()],vec![usize::MAX],vec![0,usize::MAX]]{
            assert!(doc.plan_delete_subtrees(&ids).is_err());assert_eq!(doc.deletion_fingerprint().unwrap(),original);
        }
        let stale=doc.plan_delete_subtrees(&[a]).unwrap();deletion_child(&mut doc,b,"new final callback");
        let after_callback=doc.deletion_fingerprint().unwrap();assert!(doc.apply_delete_subtrees_plan(stale).is_err());
        assert_eq!(doc.deletion_fingerprint().unwrap(),after_callback);
        let stale=doc.plan_delete_subtrees(&[a]).unwrap();doc.nodes[a].title.push_str(" renamed");
        let after_metadata=doc.deletion_fingerprint().unwrap();assert!(doc.apply_delete_subtrees_plan(stale).is_err());
        assert_eq!(doc.deletion_fingerprint().unwrap(),after_metadata);
        let stale=doc.plan_delete_subtrees(&[a]).unwrap();doc.prompt="new draft".into();doc.settings.parallel_sequences=64;
        let after_draft=doc.deletion_fingerprint().unwrap();assert!(doc.apply_delete_subtrees_plan(stale).is_err());
        assert_eq!(doc.deletion_fingerprint().unwrap(),after_draft);
    }

    #[test]
    fn subtree_compaction_moves_sparse_checkpoints_and_keeps_future_replay_valid() {
        let mut doc=Document::new();for n in 1..=130{doc.commit_generated(deletion_primitive(n as f32)).unwrap();}
        let sibling=deletion_child(&mut doc,64,"survives beyond removed vector range");
        let expected=doc.materialize(sibling).unwrap();let next_id=doc.next_primitive_id;
        let result=doc.apply_delete_subtrees_plan(doc.plan_delete_subtrees(&[80]).unwrap()).unwrap();
        let mapped=result.old_to_new[sibling].unwrap();assert_eq!(mapped,80);
        assert_eq!(doc.materialize(mapped).unwrap(),expected);assert_eq!(doc.current,79);
        assert!(doc.checkpoints.contains_key(&64));assert!(!doc.checkpoints.contains_key(&128));
        doc.jump(mapped).unwrap();let future=doc.commit_generated(deletion_primitive(400.)).unwrap();
        assert_eq!(doc.nodes[future].parent,Some(mapped));assert_eq!(doc.primitives.last().unwrap().id,next_id);
        assert_eq!(doc.materialize(future).unwrap(),doc.primitives);
    }

    #[test]
    fn settings_do_not_restore_or_serialize_a_legacy_generation_count_limit() {
        for legacy_limit in [0usize, 1, 128, 4096, usize::MAX] {
            let value = serde_json::json!({
                "cfg": 1.0, "renyi_alpha": 4.0, "seed": 42,
                "max_primitives": legacy_limit,
            });
            let settings: Settings = serde_json::from_value(value).unwrap();
            assert!(settings.validate().is_ok());
            assert!(serde_json::to_value(settings).unwrap().get("max_primitives").is_none());
        }
    }
    fn contribution_shape(id: u64, kind: ShapeKind, width: f32, height: f32, color: [u8; 3]) -> Primitive {
        let mut primitive = Primitive::new(kind, 100.25, 200.5, width, height, color);
        primitive.id = id;
        primitive.angle_deg = 37.25;
        primitive
    }

    #[test]
    fn contribution_uses_all_rgb_squared_differences_and_geometric_shape_area() {
        let rectangle = contribution_shape(1, ShapeKind::Rectangle, 4.0, 5.0, [12, 24, 36]);
        let background = [10, 20, 30];
        assert_eq!(primitive_contribution(&rectangle, background).unwrap(), (4.0 + 16.0 + 36.0) * 20.0);
        let mut ellipse = rectangle.clone();
        ellipse.shape = ShapeKind::Ellipse;
        assert!((primitive_contribution(&ellipse, background).unwrap() - 280.0 * std::f64::consts::PI).abs() < 1e-10);
        let mut moved = rectangle.clone();
        moved.x = -100.0;
        moved.y = 900.0;
        moved.angle_deg = 123.0;
        assert_eq!(primitive_contribution(&moved, background).unwrap(), primitive_contribution(&rectangle, background).unwrap());
    }

    #[test]
    fn contribution_sort_is_descending_stable_and_pins_background_without_mutating_shapes() {
        let mut background = Primitive::background([10, 20, 30]);
        background.id = 100;
        let low = contribution_shape(1, ShapeKind::Rectangle, 10.0, 10.0, [11, 20, 30]);
        let high = contribution_shape(2, ShapeKind::Rectangle, 10.0, 10.0, [20, 20, 30]);
        let tied = contribution_shape(3, ShapeKind::Rectangle, 10.0, 10.0, [0, 20, 30]);
        let input = vec![low.clone(), background.clone(), high.clone(), tied.clone()];
        let before = input.clone();
        let ordered = reorder_by_contribution(&input).unwrap();
        assert_eq!(ordered, vec![background, high, tied, low]);
        assert_eq!(input, before);
        assert_eq!(reorder_by_contribution(&ordered).unwrap(), ordered);
        assert_eq!(ordered.len(), input.len());
        for primitive in &input {
            assert_eq!(ordered.iter().find(|candidate| candidate.id == primitive.id), Some(primitive));
        }
    }

    #[test]
    fn contribution_sort_multiplies_area_instead_of_contrast_alone() {
        let near_large = contribution_shape(1, ShapeKind::Rectangle, 20.0, 20.0, [245, 255, 255]);
        let far_small = contribution_shape(2, ShapeKind::Rectangle, 1.0, 1.0, [155, 255, 255]);
        assert_eq!(primitive_contribution(&near_large, [255; 3]).unwrap(), 40_000.0);
        assert_eq!(primitive_contribution(&far_small, [255; 3]).unwrap(), 10_000.0);
        let ordered = reorder_by_contribution(&[far_small, near_large]).unwrap();
        assert_eq!(ordered.iter().map(|primitive| primitive.id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn contribution_sort_without_background_uses_white_and_never_inserts_one() {
        let light = contribution_shape(1, ShapeKind::Rectangle, 10.0, 10.0, [250; 3]);
        let dark = contribution_shape(2, ShapeKind::Rectangle, 10.0, 10.0, [20; 3]);
        let ordered = reorder_by_contribution(&[light.clone(), dark.clone()]).unwrap();
        assert_eq!(ordered, vec![dark, light]);
        assert!(ordered.iter().all(|primitive| primitive.shape != ShapeKind::Background));
        assert!(reorder_by_contribution(&[]).unwrap().is_empty());
    }

    #[test]
    fn contribution_zero_area_and_background_color_have_zero_score() {
        let zero_area = contribution_shape(1, ShapeKind::Ellipse, 0.0, 10.0, [0; 3]);
        let no_contrast = contribution_shape(2, ShapeKind::Rectangle, 500.0, 500.0, [255; 3]);
        let visible = contribution_shape(3, ShapeKind::Rectangle, 1.0, 1.0, [254; 3]);
        assert_eq!(primitive_contribution(&zero_area, [255; 3]).unwrap(), 0.0);
        assert_eq!(primitive_contribution(&no_contrast, [255; 3]).unwrap(), 0.0);
        let ordered = reorder_by_contribution(&[zero_area, no_contrast, visible]).unwrap();
        assert_eq!(ordered.iter().map(|primitive| primitive.id).collect::<Vec<_>>(), vec![3, 1, 2]);
    }

    #[test]
    fn contribution_invalid_geometry_or_duplicate_background_fails_without_mutation() {
        for invalid in [f32::NAN, f32::INFINITY, -1.0] {
            let primitive = contribution_shape(1, ShapeKind::Rectangle, invalid, 10.0, [0; 3]);
            assert!(reorder_by_contribution(&[primitive]).is_err());
        }
        let mut invalid_position = contribution_shape(1, ShapeKind::Rectangle, 10.0, 10.0, [0; 3]);
        invalid_position.x = f32::INFINITY;
        assert!(reorder_by_contribution(&[invalid_position]).is_err());
        let backgrounds = vec![Primitive::background([0; 3]), Primitive::background([255; 3])];
        let before = backgrounds.clone();
        assert!(reorder_by_contribution(&backgrounds).is_err());
        assert_eq!(backgrounds, before);
    }

    #[test]
    fn new_settings_enable_overclock_but_keep_automatic_reordering_opt_in() {
        let settings = Settings::default();
        assert!(settings.context_overclock);
        assert!(!settings.auto_reorder_before_generation);
        let legacy = r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42,"max_primitives":16}"#;
        let old: Settings = serde_json::from_str(legacy).unwrap();
        assert!(old.context_overclock);
        assert!(!old.auto_reorder_before_generation);
        let explicit = r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42,"max_primitives":16,"context_overclock":false,"auto_reorder_before_generation":true}"#;
        let selected: Settings = serde_json::from_str(explicit).unwrap();
        assert!(!selected.context_overclock);
        assert!(selected.auto_reorder_before_generation);
        assert_eq!(serde_json::from_str::<Settings>(&serde_json::to_string(&selected).unwrap()).unwrap(), selected);
    }

    fn rect(x: f32) -> Primitive {
        Primitive::new(ShapeKind::Rectangle, x, 100.0, 40.0, 20.0, [200, 50, 30])
    }

    #[test]
    fn stage_assignment_is_metadata_preserved_by_normalization_but_not_model_tokens() {
        let mut primitive=rect(32.2);let plain=primitive.to_tokens().unwrap();
        let assignment=PrimitiveStageAssignment{stage:StageTier::Coarse,cell:15,queue_order:7};
        primitive.stage_assignment=Some(assignment);
        assert_eq!(primitive.normalized().unwrap().stage_assignment,Some(assignment));
        assert_eq!(primitive.to_tokens().unwrap(),plain);
        assert_eq!(Primitive::from_tokens(1,plain).unwrap().stage_assignment,None);
        assert_eq!(rect(2.).stage_assignment,None);
        let legacy=serde_json::to_value(rect(2.)).unwrap();assert!(legacy.get("stage_assignment").is_none());
        assert_eq!(serde_json::from_value::<Primitive>(legacy).unwrap().stage_assignment,None);
        let encoded=serde_json::to_value(&primitive).unwrap();assert_eq!(encoded["stage_assignment"]["stage"],"coarse");
        assert_eq!(serde_json::from_value::<Primitive>(encoded).unwrap(),primitive);
        for(stage,cell)in[(StageTier::Global,1),(StageTier::Coarse,16),(StageTier::Fine,64)]{
            primitive.stage_assignment=Some(PrimitiveStageAssignment{stage,cell,queue_order:1});
            assert!(primitive.normalized().is_err());
        }
    }

    #[test]
    fn preview_canvas_is_nonmutating_and_commits_the_same_normalized_ids_and_background_order() {
        let mut doc=Document::new();let before=doc.deletion_fingerprint().unwrap();
        let next=vec![rect(11.2),Primitive::background([11,23,35])];
        let preview=doc.preview_canvas(next.clone()).unwrap();
        assert_eq!(doc.deletion_fingerprint().unwrap(),before);
        assert_eq!(doc.preview_canvas(next).unwrap(),preview);
        assert_eq!(preview[0].shape,ShapeKind::Background);assert_eq!(preview[0].id,2);assert_eq!(preview[1].id,1);
        assert!(doc.preview_canvas(vec![Primitive::background([0;3]),Primitive::background([2;3])]).is_err());
        assert_eq!(doc.deletion_fingerprint().unwrap(),before);
        doc.commit_canvas("confirmed",OperationSource::User,preview.clone()).unwrap();
        assert_eq!(doc.primitives,preview);assert_eq!(doc.next_primitive_id,3);
    }

    #[test]
    fn rollback_last_canvas_restores_only_the_fresh_user_leaf_and_never_reuses_ids() {
        let mut doc=Document::new();doc.prompt="parent".into();
        let parent=doc.commit_canvas("parent",OperationSource::User,vec![rect(20.)]).unwrap();
        let canvas=doc.primitives.clone();let parent_settings=doc.settings.clone();
        doc.prompt="uncommitted editing draft".into();doc.settings.seed=888;
        let mut next=canvas.clone();next.push(rect(40.));
        let leaf=doc.commit_canvas("candidate",OperationSource::User,next).unwrap();
        let abandoned=doc.primitives[1].id;let next_id=doc.next_primitive_id;
        doc.rollback_last_canvas(leaf).unwrap();
        assert_eq!(doc.current,parent);assert_eq!(doc.primitives,canvas);assert_eq!(doc.prompt,"parent");assert_eq!(doc.settings,parent_settings);
        assert_eq!(doc.next_primitive_id,next_id);assert!(doc.nodes[parent].children.is_empty());
        let model=doc.commit_generated(rect(60.)).unwrap();assert!(doc.primitives.last().unwrap().id>abandoned);
        let before=doc.deletion_fingerprint().unwrap();assert!(doc.rollback_last_canvas(model).is_err());assert_eq!(doc.deletion_fingerprint().unwrap(),before);
    }

    #[test]
    fn token_contract_matches_real_native_layout_and_background() {
        let p = Primitive {
            id: 8,
            x: 511.5,
            y: 0.5,
            angle_deg: 89.0,
            width: 511.75,
            height: 10.25,
            shape: ShapeKind::Ellipse,
            color: [254, 100, 0],
            stage_assignment:None,
        };
        let tokens = p.to_tokens().unwrap();
        assert_eq!(
            tokens,
            [1023, 1025, 2315, 4365, 4407, 6416, 6797, 6848, 6926]
        );
        assert_eq!(Primitive::from_tokens(8, tokens).unwrap(), p);
        let bg = Primitive::background([1, 255, 17]);
        let decoded = Primitive::from_tokens(4, bg.to_tokens().unwrap()).unwrap();
        assert_eq!(decoded.shape, ShapeKind::Background);
        assert_eq!(decoded.color, [0, 254, 16]);
        assert_eq!(bg.to_tokens().unwrap()[5], 6414);
        let mut marker = bg.to_tokens().unwrap();
        marker[5] = 6417;
        assert!(Primitive::from_tokens(4, marker).is_err());
        // All thirds-of-a-degree bins must survive floating point editor
        // coordinates without silently changing the model's completed token.
        for angle_bin in 0..270 {
            let mut t = tokens;
            t[2] = ATTRIBUTE_OFFSETS[2] + angle_bin;
            assert_eq!(
                Primitive::from_tokens(8, t).unwrap().to_tokens().unwrap(),
                t
            );
        }
    }

    #[test]
    fn normalization_rotates_swaps_and_quantizes_without_invalid_tokens() {
        let mut p = rect(-1.0);
        p.y = 999.0;
        p.angle_deg = 120.0;
        p.width = 30.3;
        p.height = 60.7;
        p.color = [1, 255, 19];
        let q = p.normalized().unwrap();
        assert_eq!(
            (q.x, q.y, q.angle_deg, q.width, q.height),
            (0.0, 511.5, 30.0, 60.5, 30.25)
        );
        assert_eq!(q.color, [0, 254, 18]);
        assert_eq!(
            q,
            Primitive::from_tokens(q.id, q.to_tokens().unwrap()).unwrap()
        );
        p.x = f32::NAN;
        assert!(p.normalized().is_err());
    }

    #[test]
    fn jumping_to_old_node_and_editing_preserves_both_branches_and_metadata() {
        let mut d = Document::new();
        d.prompt = "金发".into();
        d.settings.seed = 42;
        let a = d
            .commit_canvas("添加", OperationSource::User, vec![rect(100.0)])
            .unwrap();
        let mut next = d.primitives.clone();
        next[0].color = [0, 100, 200];
        d.prompt = "蓝发".into();
        d.settings.cfg = 2.0;
        let b = d
            .commit_canvas("改蓝", OperationSource::User, next)
            .unwrap();
        d.jump(a).unwrap();
        assert_eq!(d.prompt, "金发");
        assert_eq!(d.settings.seed, 42);
        assert_eq!(d.settings.cfg, 1.0);
        let mut other = d.primitives.clone();
        other[0].x = 250.0;
        let c = d
            .commit_canvas("移动", OperationSource::User, other)
            .unwrap();
        assert_eq!(d.nodes[a].children, vec![b, c]);
        d.jump(b).unwrap();
        assert_eq!(d.primitives[0].color, [0, 100, 200]);
        assert_eq!(d.primitives[0].x, 100.0);
        assert_eq!(d.prompt, "蓝发");
        assert_eq!(d.settings.cfg, 2.0);
        d.jump(c).unwrap();
        assert_eq!(d.primitives[0].x, 250.0);
        assert_eq!(d.primitives[0].color, [200, 50, 30]);
    }

    #[test]
    fn multi_edit_is_one_atomic_node_and_noop_parameter_edits_make_no_node() {
        let mut d = Document::new();
        d.commit_canvas(
            "添加两个",
            OperationSource::User,
            vec![rect(100.0), rect(200.0)],
        )
        .unwrap();
        let old = d.primitives.clone();
        let mut changed = old.clone();
        for p in &mut changed {
            p.x += 10.0;
        }
        d.commit_canvas("移动两个", OperationSource::User, changed.clone())
            .unwrap();
        assert_eq!(d.nodes.len(), 3);
        assert_eq!(d.current_node().patches.len(), 2);
        d.undo().unwrap();
        assert_eq!(d.primitives, old);
        d.redo().unwrap();
        assert_eq!(d.primitives, changed);
        d.prompt = "新提示词".into();
        d.settings.seed = 99;
        d.commit_canvas("空修改", OperationSource::User, changed)
            .unwrap();
        assert_eq!(d.nodes.len(), 3);
    }

    #[test]
    fn generated_nodes_store_add_delta_and_partial_stop_keeps_tree_untouched() {
        let mut d = Document::new();
        for i in 0..160 {
            d.commit_generated(rect(i as f32)).unwrap();
        }
        assert_eq!(d.nodes.len(), 161);
        assert_eq!(d.stored_patch_count(), 160);
        assert!(d.nodes[1..].iter().all(|n| n.patches.len() == 1));
        let canvas = d.primitives.clone();
        let node = d.current;
        let tokens = rect(20.0).to_tokens().unwrap();
        let mut partial = PartialPrimitive::default();
        for token in tokens.iter().take(7) {
            partial.push(*token).unwrap();
        }
        assert!(!partial.is_complete());
        partial.cancel();
        assert_eq!(partial.len(), 0);
        assert!(partial.finish(0).is_err());
        assert_eq!(d.current, node);
        assert_eq!(d.primitives, canvas);
        d.jump(65).unwrap();
        assert_eq!(d.primitives.len(), 65);
        d.jump(160).unwrap();
        assert_eq!(d.primitives, canvas);
    }

    #[test]
    fn deletion_insertion_reorder_and_invalid_commit_are_atomic() {
        let mut d = Document::new();
        d.commit_canvas(
            "三个",
            OperationSource::User,
            vec![rect(10.0), rect(20.0), rect(30.0)],
        )
        .unwrap();
        let initial = d.primitives.clone();
        let parent = d.current;
        let new = vec![initial[2].clone(), rect(40.0), initial[0].clone()];
        d.commit_canvas("删加排序", OperationSource::User, new)
            .unwrap();
        let expected = d.primitives.clone();
        let id = d.current;
        d.jump(parent).unwrap();
        assert_eq!(d.primitives, initial);
        d.jump(id).unwrap();
        assert_eq!(d.primitives, expected);
        let mut invalid = expected.clone();
        invalid.push(expected[0].clone());
        assert!(d
            .commit_canvas("无效重复", OperationSource::User, invalid)
            .is_err());
        assert_eq!(d.current, id);
        assert_eq!(d.primitives, expected);
    }

    #[test]
    fn background_remains_bottom_and_is_not_half_alpha_ordinary_shape() {
        let mut d = Document::new();
        d.commit_canvas(
            "背景和图元",
            OperationSource::User,
            vec![rect(50.0), Primitive::background([20, 40, 60])],
        )
        .unwrap();
        assert_eq!(d.primitives[0].shape, ShapeKind::Background);
        let snapshot = d.primitives.clone();
        let duplicate = vec![
            Primitive::background([20, 40, 60]),
            Primitive::background([30, 50, 70]),
        ];
        assert!(d
            .commit_canvas("双背景", OperationSource::User, duplicate)
            .is_err());
        assert_eq!(d.primitives, snapshot);
        assert_eq!(SHAPE_ALPHA, 0.5);
    }

    #[test]
    fn complete_generation_configuration_survives_deep_history_and_new_fork() {
        let mut d = Document::new();
        let first = Settings {
            cfg: 2.5,
            renyi_alpha: 3.5,
            seed: u64::MAX - 2,
            auto_seed: true,
            parallel_sequences: 4,
            sampling_backend: SamplingBackend::Gpu,
            context_overclock: true,
            pause_at_stage_end: true,
            half_region_switch: true,
            auto_reorder_before_generation: true,
            acceleration: AccelerationSettings {
                mode: crate::acceleration::AccelerationMode::Optimized,
                split_k: crate::acceleration::FeaturePreference::Enabled,
                primitive_graph: crate::acceleration::FeaturePreference::Disabled,
                ..Default::default()
            },
        };
        d.prompt = "第一条创作路线".into();
        d.settings = first.clone();
        let first_node = d.commit_generated(rect(12.0)).unwrap();

        let second = Settings {
            cfg: 1.0,
            renyi_alpha: 7.0,
            seed: 17,
            auto_seed: false,
            parallel_sequences: 2,
            sampling_backend: SamplingBackend::Cpu,
            context_overclock: false,
            pause_at_stage_end: false,
            half_region_switch: false,
            auto_reorder_before_generation: false,
            acceleration: AccelerationSettings {
                mode: crate::acceleration::AccelerationMode::Compatible,
                ..Default::default()
            },
        };
        d.prompt = "另一组完整设置".into();
        d.settings = second.clone();
        for i in 0..70 {
            d.commit_generated(rect(20.0 + i as f32)).unwrap();
        }
        let deep_node = d.current;
        let deep_canvas = d.primitives.clone();

        d.jump(first_node).unwrap();
        assert_eq!(d.settings, first);
        assert_eq!(d.prompt, "第一条创作路线");
        // A staged setting edit must not rewrite the already committed node.
        d.settings.seed = 999;
        d.settings.renyi_alpha = 2.0;
        d.prompt = "尚未生成的草稿".into();
        d.jump(deep_node).unwrap();
        assert_eq!(d.settings, second);
        assert_eq!(d.prompt, "另一组完整设置");
        assert_eq!(d.primitives, deep_canvas);

        d.jump(first_node).unwrap();
        d.settings.seed = 314159;
        d.prompt = "从旧状态重新分叉".into();
        let fork_settings = d.settings.clone();
        let fork = d.commit_generated(rect(88.0)).unwrap();
        assert_eq!(d.nodes[first_node].children.len(), 2);
        d.jump(deep_node).unwrap();
        assert_eq!(d.settings, second);
        d.jump(fork).unwrap();
        assert_eq!(d.settings, fork_settings);
        assert_eq!(d.prompt, "从旧状态重新分叉");
        assert_eq!(d.primitives.len(), 2);
    }

    #[test]
    fn automatic_seed_defaults_on_and_legacy_deserialization_preserves_the_recorded_seed(){
        let defaults=Settings::default();assert!(defaults.auto_seed);assert!(Document::new().settings.auto_seed);
        let legacy:Settings=serde_json::from_str(r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42}"#).unwrap();
        assert!(legacy.auto_seed);assert_eq!(legacy.seed,42);
        let mut strict_legacy=serde_json::to_value(&legacy).unwrap();strict_legacy.as_object_mut().unwrap().remove("auto_seed");
        let decoded=project_data::strict_settings(strict_legacy).unwrap();assert!(decoded.auto_seed);assert_eq!(decoded.seed,42);
        let manual=Settings{auto_seed:false,seed:123456789,..Default::default()};
        let encoded=serde_json::to_value(&manual).unwrap();assert_eq!(encoded["auto_seed"],false);
        assert_eq!(serde_json::from_value::<Settings>(encoded.clone()).unwrap(),manual);
        assert_eq!(project_data::strict_settings(encoded).unwrap(),manual);
        for malformed in [serde_json::json!("true"),serde_json::json!(1),serde_json::Value::Null]{
            let mut value=serde_json::to_value(&defaults).unwrap();value["auto_seed"]=malformed;
            assert!(serde_json::from_value::<Settings>(value.clone()).is_err());assert!(project_data::strict_settings(value).is_err());
        }
    }

    #[test]
    fn seed_mode_and_actual_seed_are_historical_values_not_refreshed_by_jump_or_import(){
        let mut doc=Document::new();doc.settings.seed=111111;doc.settings.auto_seed=true;
        let automatic=doc.commit_generated(rect(12.)).unwrap();
        doc.settings.seed=222222;doc.settings.auto_seed=false;let manual=doc.commit_generated(rect(30.)).unwrap();
        doc.jump(automatic).unwrap();assert_eq!(doc.settings.seed,111111);assert!(doc.settings.auto_seed);
        doc.settings.seed=333333;let fork=doc.commit_generated(rect(48.)).unwrap();
        assert_eq!(doc.nodes[automatic].settings.seed,111111);assert_eq!(doc.nodes[fork].settings.seed,333333);
        assert!(doc.nodes[automatic].settings.auto_seed&&doc.nodes[fork].settings.auto_seed);
        assert_eq!(doc.nodes[manual].settings.seed,222222);assert!(!doc.nodes[manual].settings.auto_seed);
        let mut loaded=Document::from_project_data(serde_json::from_value(serde_json::to_value(doc.project_ref()).unwrap()).unwrap()).unwrap();
        for(node,seed,automatic)in[(automatic,111111,true),(manual,222222,false),(fork,333333,true)]{
            loaded.jump(node).unwrap();assert_eq!(loaded.settings.seed,seed);assert_eq!(loaded.settings.auto_seed,automatic);
            assert_eq!(loaded.nodes[node].settings.seed,seed);
        }
        let mut legacy=serde_json::to_value(doc.project_ref()).unwrap();legacy["settings"].as_object_mut().unwrap().remove("auto_seed");
        for node in legacy["nodes"].as_array_mut().unwrap(){node["settings"].as_object_mut().unwrap().remove("auto_seed");}
        let old=Document::from_project_data(serde_json::from_value(legacy).unwrap()).unwrap();
        assert!(old.settings.auto_seed&&old.nodes.iter().all(|node|node.settings.auto_seed));
        assert_eq!(old.nodes[automatic].settings.seed,111111);assert_eq!(old.nodes[manual].settings.seed,222222);
        assert_eq!(old.nodes[fork].settings.seed,333333);
    }

    #[test]
    fn half_region_switch_defaults_off_and_strict_legacy_settings_remain_readable() {
        let default=Settings::default();assert!(!default.half_region_switch);
        let legacy=serde_json::to_value(&default).unwrap();
        assert!(legacy.get("half_region_switch").is_none(),"disabled extension must not change schema-1 defaults");
        assert!(!serde_json::from_value::<Settings>(legacy.clone()).unwrap().half_region_switch);
        assert!(!project_data::strict_settings(legacy.clone()).unwrap().half_region_switch);
        for enabled in [false,true] {
            let mut value=legacy.clone();value["half_region_switch"]=serde_json::json!(enabled);
            let settings:Settings=serde_json::from_value(value.clone()).unwrap();
            assert_eq!(settings.half_region_switch,enabled);
            assert_eq!(project_data::strict_settings(value).unwrap().half_region_switch,enabled);
            assert_eq!(serde_json::from_str::<Settings>(&serde_json::to_string(&settings).unwrap()).unwrap(),settings);
        }
        let mut bad=legacy;bad["half_region_switch"]=serde_json::json!("true");
        assert!(serde_json::from_value::<Settings>(bad.clone()).is_err());
        assert!(project_data::strict_settings(bad).is_err());
    }

    #[test]
    fn half_region_switch_project_payload_preserves_per_node_flags_and_old_missing_fields_default_off() {
        let mut doc=Document::new();doc.settings.half_region_switch=true;
        let enabled=doc.commit_generated(rect(12.)).unwrap();
        doc.settings.half_region_switch=false;let disabled=doc.commit_generated(rect(30.)).unwrap();
        let value=serde_json::to_value(doc.project_ref()).unwrap();
        assert_eq!(value["nodes"][enabled]["settings"]["half_region_switch"],true);
        assert!(value["nodes"][disabled]["settings"].get("half_region_switch").is_none());
        let mut loaded=Document::from_project_data(serde_json::from_value(value.clone()).unwrap()).unwrap();
        assert!(!loaded.settings.half_region_switch);loaded.jump(enabled).unwrap();assert!(loaded.settings.half_region_switch);
        loaded.jump(disabled).unwrap();assert!(!loaded.settings.half_region_switch);
        let mut legacy=value;legacy["settings"].as_object_mut().unwrap().remove("half_region_switch");
        for node in legacy["nodes"].as_array_mut().unwrap(){node["settings"].as_object_mut().unwrap().remove("half_region_switch");}
        let old=Document::from_project_data(serde_json::from_value(legacy).unwrap()).unwrap();
        assert!(!old.settings.half_region_switch&&old.nodes.iter().all(|node|!node.settings.half_region_switch));
        assert_eq!(old.primitives,doc.primitives);
    }

    #[test]
    fn invalid_sampler_configuration_never_commits_a_canvas_change() {
        let mut d = Document::new();
        d.commit_generated(rect(12.0)).unwrap();
        let old_node = d.current;
        let old_canvas = d.primitives.clone();
        d.settings.renyi_alpha = 1.0;
        assert!(d.commit_generated(rect(25.0)).is_err());
        assert_eq!(d.current, old_node);
        assert_eq!(d.primitives, old_canvas);
        d.settings.renyi_alpha = 4.0;
        d.settings.cfg = f32::NAN;
        assert!(d.commit_generated(rect(25.0)).is_err());
        assert_eq!(d.nodes.len(), 2);
    }

    #[test]
    fn explicit_parent_commit_never_changes_selected_canvas_or_staged_metadata() {
        let mut d = Document::new();
        let selected = d
            .commit_generated(Primitive::background([20, 40, 60]))
            .unwrap();
        d.prompt = "用户当前未生成的提示词".into();
        d.settings.seed = 999;
        d.settings.parallel_sequences = 4;
        let selected_canvas = d.primitives.clone();
        let staged = d.settings.clone();
        let other_settings = Settings {
            seed: 101,
            parallel_sequences: 2,
            ..Default::default()
        };
        let other = d
            .commit_generated_at(
                0,
                Primitive::background([120, 40, 20]),
                "另一条序列".into(),
                other_settings.clone(),
            )
            .unwrap();
        assert_eq!(d.current, selected);
        assert_eq!(d.primitives, selected_canvas);
        assert_eq!(d.prompt, "用户当前未生成的提示词");
        assert_eq!(d.settings, staged);
        assert_eq!(d.nodes[other].parent, Some(0));
        assert_eq!(d.nodes[other].settings, other_settings);
        assert_eq!(d.materialize(other).unwrap()[0].color, [120, 40, 20]);
        let count = d.nodes.len();
        assert!(d
            .commit_generated_at(999, rect(1.0), String::new(), Settings::default())
            .is_err());
        assert_eq!(d.nodes.len(), count);
        assert_eq!(d.current, selected);
    }

    #[test]
    fn missing_parallel_count_defaults_to_eight_and_parallel_limits_are_explicit() {
        let fields = vec![
            ("cfg", 1u64),
            ("renyi_alpha", 4u64),
            ("seed", 42u64),
            ("max_primitives", 16u64),
        ];
        let deserializer = serde::de::value::MapDeserializer::<_, serde::de::value::Error>::new(
            fields.into_iter(),
        );
        let old = Settings::deserialize(deserializer).unwrap();
        assert_eq!(old.parallel_sequences, 8);
        assert_eq!(Settings::default().parallel_sequences, 8);
        assert_eq!(Document::new().settings.parallel_sequences, 8);
        let explicit_single: Settings = serde_json::from_str(
            r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42,"parallel_sequences":1}"#,
        ).unwrap();
        assert_eq!(explicit_single.parallel_sequences, 1);
        assert!(old.context_overclock);
        assert!(!old.auto_reorder_before_generation);
        assert_eq!(old.sampling_backend, SamplingBackend::Auto);
        assert_eq!(old.seed, 42);
        assert!(old.validate().is_ok());
        for count in [0] {
            let invalid = Settings {
                parallel_sequences: count,
                ..Default::default()
            };
            assert!(invalid.validate().is_err());
        }
        for count in [17, 32, 1000, usize::MAX] {
            let expanded=Settings{parallel_sequences:count,..Default::default()};
            assert!(expanded.validate().is_ok());
            let encoded=serde_json::to_string(&expanded).unwrap();
            assert_eq!(serde_json::from_str::<Settings>(&encoded).unwrap().parallel_sequences,count);
        }
        let seeds: Vec<_> = (0..16)
            .map(|lane| parallel_seed(u64::MAX - 3, lane))
            .collect();
        assert_eq!(seeds[0], u64::MAX - 3);
        assert_eq!(seeds.iter().copied().collect::<HashSet<_>>().len(), 16);
    }

    #[test]
    fn stage_end_pause_defaults_off_for_new_and_legacy_settings_and_true_round_trips() {
        assert!(!Settings::default().pause_at_stage_end);
        assert!(!Document::new().settings.pause_at_stage_end);
        let legacy = r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42}"#;
        let old: Settings = serde_json::from_str(legacy).unwrap();
        assert!(!old.pause_at_stage_end);
        for enabled in [false, true] {
            let json = serde_json::json!({"cfg":1.0,"renyi_alpha":4.0,"seed":42,
                "pause_at_stage_end":enabled});
            let settings: Settings = serde_json::from_value(json).unwrap();
            assert_eq!(settings.pause_at_stage_end, enabled);
            let encoded = serde_json::to_string(&settings).unwrap();
            let restored: Settings = serde_json::from_str(&encoded).unwrap();
            assert_eq!(restored, settings);
            assert!(restored.validate().is_ok());
        }
    }

    #[test]
    fn stage_end_pause_is_staged_without_history_and_restored_with_each_canvas_node() {
        let mut document = Document::new();
        document.settings.pause_at_stage_end = true;
        assert_eq!(document.nodes.len(), 1);
        assert!(!document.nodes[0].settings.pause_at_stage_end);
        let enabled = document.commit_generated(rect(30.0)).unwrap();
        assert!(document.nodes[enabled].settings.pause_at_stage_end);
        document.settings.pause_at_stage_end = false;
        assert_eq!(document.nodes.len(), 2);
        assert!(document.nodes[enabled].settings.pause_at_stage_end);
        let disabled = document.commit_generated(rect(40.0)).unwrap();
        assert!(!document.nodes[disabled].settings.pause_at_stage_end);
        document.jump(enabled).unwrap();
        assert!(document.settings.pause_at_stage_end);
        document.jump(disabled).unwrap();
        assert!(!document.settings.pause_at_stage_end);
        document.jump(0).unwrap();
        assert!(!document.settings.pause_at_stage_end);
    }

    #[test]
    fn old_json_defaults_sampler_to_auto_and_explicit_backends_round_trip() {
        let legacy = r#"{"cfg":1.0,"renyi_alpha":4.0,"seed":42,"max_primitives":16}"#;
        let old: Settings = serde_json::from_str(legacy).unwrap();
        assert_eq!(old.sampling_backend, SamplingBackend::Auto);
        assert_eq!(old.parallel_sequences, 8);
        assert_eq!(old.acceleration, AccelerationSettings::default());
        for backend in [
            SamplingBackend::Auto,
            SamplingBackend::Gpu,
            SamplingBackend::Cpu,
        ] {
            let settings = Settings {
                sampling_backend: backend,
                ..Default::default()
            };
            let encoded = serde_json::to_string(&settings).unwrap();
            assert_eq!(
                serde_json::from_str::<Settings>(&encoded).unwrap(),
                settings
            );
        }
        assert!(serde_json::from_str::<Settings>(r#"{"cfg":1,"renyi_alpha":4,"seed":42,"max_primitives":16,"sampling_backend":"unknown"}"#).is_err());
    }

    #[test]
    fn sampler_choice_is_staged_without_a_node_and_restored_with_real_canvas_states() {
        let mut document = Document::new();
        document.settings.sampling_backend = SamplingBackend::Gpu;
        assert_eq!(document.nodes.len(), 1);
        assert_eq!(document.current, 0);
        assert_eq!(
            document.nodes[0].settings.sampling_backend,
            SamplingBackend::Auto
        );
        let gpu = document
            .commit_generated(Primitive::background([30, 50, 70]))
            .unwrap();
        assert_eq!(
            document.nodes[gpu].settings.sampling_backend,
            SamplingBackend::Gpu
        );
        document.settings.sampling_backend = SamplingBackend::Cpu;
        assert_eq!(document.nodes.len(), 2);
        assert_eq!(
            document.nodes[gpu].settings.sampling_backend,
            SamplingBackend::Gpu
        );
        let cpu = document.commit_generated(rect(40.)).unwrap();
        assert_eq!(
            document.nodes[cpu].settings.sampling_backend,
            SamplingBackend::Cpu
        );
        document.jump(gpu).unwrap();
        assert_eq!(document.settings.sampling_backend, SamplingBackend::Gpu);
        document.jump(cpu).unwrap();
        assert_eq!(document.settings.sampling_backend, SamplingBackend::Cpu);
        document.jump(0).unwrap();
        assert_eq!(document.settings.sampling_backend, SamplingBackend::Auto);
        document.settings.sampling_backend = SamplingBackend::Gpu;
        document
            .commit_canvas(
                "只改变采样设置",
                OperationSource::User,
                document.primitives.clone(),
            )
            .unwrap();
        assert_eq!(document.nodes.len(), 3);
        assert_eq!(document.current, 0);
        assert_eq!(
            document.nodes[0].settings.sampling_backend,
            SamplingBackend::Auto
        );
    }
}
