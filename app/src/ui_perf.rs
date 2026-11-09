//! Opt-in fixed-work diagnostic counters. Normal sessions allocate none.
use crate::runtime::Event;
use serde_json::{json, Value};
use std::time::Instant;

pub struct UiPerformance {
    started: Instant,
    tokens: Vec<Vec<[u32; 9]>>,
    callbacks: Vec<Vec<f64>>,
    statuses: Vec<String>,
    worker_seconds: Option<f64>,
    event_seconds: f64,
    nodes_seconds: f64,
    ui_seconds: f64,
    frames: usize,
    events: usize,
}
impl UiPerformance {
    pub fn new(lanes: usize) -> Self {
        Self { started: Instant::now(), tokens: vec![Vec::new(); lanes], callbacks: vec![Vec::new(); lanes],
            statuses: Vec::new(), worker_seconds: None, event_seconds: 0., nodes_seconds: 0., ui_seconds: 0., frames: 0, events: 0 }
    }
    pub fn observe(&mut self, event: &Event) {
        self.events += 1;
        match event {
            Event::Primitive { tokens, .. } => self.primitive(0, *tokens),
            Event::ParallelPrimitive { lane, tokens, .. } => self.primitive(*lane, *tokens),
            Event::Status(message) => self.statuses.push(message.clone()),
            Event::Finished { elapsed_seconds, .. } | Event::ParallelFinished { elapsed_seconds, .. } => self.worker_seconds = Some(*elapsed_seconds),
            _ => {}
        }
    }
    fn primitive(&mut self, lane: usize, tokens: [u32; 9]) {
        self.tokens[lane].push(tokens);
        self.callbacks[lane].push(self.started.elapsed().as_secs_f64());
    }
    pub fn event_cpu(&mut self, seconds: f64) { self.event_seconds += seconds; }
    pub fn frame_cpu(&mut self, nodes: f64, ui: f64) { self.nodes_seconds += nodes; self.ui_seconds += ui; self.frames += 1; }
    pub fn report(&self) -> Value {
        let total: usize = self.tokens.iter().map(Vec::len).sum();
        let wall = self.started.elapsed().as_secs_f64();
        let warm_end = self.callbacks.iter().filter_map(|lane| lane.get(1)).copied().reduce(f64::max);
        let last = self.callbacks.iter().filter_map(|lane| lane.last()).copied().reduce(f64::max);
        let steady = warm_end.zip(last).and_then(|(start, end)| (end > start).then_some((total.saturating_sub(self.tokens.len()*2)) as f64 / (end-start)));
        json!({"scope":"actual native GUI; fixed diagnostic work with unlimited-production cache forecast",
            "wall_seconds_until_events_consumed":wall,"worker_seconds":self.worker_seconds,
            "worker_complete_primitives_per_second":self.worker_seconds.map(|s| total as f64/s),
            "display_complete_primitives_per_second":total as f64/wall,
            "display_post_two_rounds_primitives_per_second":steady,
            "complete_primitives":total,"logical_lanes":self.tokens.len(),
            "event_handler_cpu_seconds":self.event_seconds,"node_view_build_cpu_seconds":self.nodes_seconds,
            "egui_build_cpu_seconds":self.ui_seconds,"generation_frames":self.frames,"events":self.events,
            "callbacks_by_lane":self.callbacks,"tokens_by_lane":self.tokens,"statuses":self.statuses,
            "timing_note":"model load excluded; CPU egui build excludes GPU presentation; callback times are UI receipt, not GPU kernel times"})
    }
}
