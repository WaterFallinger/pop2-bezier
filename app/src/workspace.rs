//! The last selected editing workflow, not a type of save. All saves use one
//! state-tree format and may switch between standard and modification workflows.
use serde::{Deserialize,Serialize};

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub(crate) enum WorkspaceMode{#[default] Paint,Redraw}

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkspaceState{
    #[serde(default)]pub mode:WorkspaceMode,
    #[serde(default)]pub selection:crate::redraw::RedrawSelection,
}
impl WorkspaceState{
    pub fn redraw()->Self{Self{mode:WorkspaceMode::Redraw,..Default::default()}}
    pub fn validate(&self)->Result<(),String>{self.selection.validate()}
}
