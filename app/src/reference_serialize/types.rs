// src/types.rs
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Config {
    pub image_offset: u32,
    pub dispatch_images: u32,
    pub num_images: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuShape {
    pub cx: f32,
    pub cy: f32,
    pub hw: f32,
    pub hh: f32,
    pub theta: f32,
    pub shape_type: u32,
    pub r: u32,
    pub g: u32,
    pub b: u32,
    pub delta_err: f32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ImageState {
    pub current_sse: f32,
    pub initial_sse: f32,
    pub region_sse: f32,
    pub phase_two_start_count: u32,
    pub is_done: u32,
    pub phase: u32,
    pub first_phase_shape_count: u32,
    pub successful_shape_count: u32,
    pub search_attempt_count: u32,
    pub history_count: u32,
    pub active_grid: u32,
    pub stage: u32,
    pub termination_reason: u32,
    pub grid_shape_count: u32,
    pub perturbed: u32,
    pub rng: u32,
    pub counts: [u32; 80],
    pub closed: [u32; 80],
    pub perturb: [u32; 80],
}
