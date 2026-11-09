use super::*;
use bytemuck::Zeroable;

async fn execute(
    engine: &GpuEngine,
    states: &[ImageState],
    capacity: usize,
    steps: usize,
    target: u32,
) -> (Vec<ImageState>, Vec<GpuShape>) {
    execute_scheduled(engine, states, capacity, steps, target, states.len(), 1).await
}

async fn execute_scheduled(
    engine: &GpuEngine,
    states: &[ImageState],
    capacity: usize,
    steps: usize,
    target: u32,
    dispatch_images: usize,
    max_in_flight: usize,
) -> (Vec<ImageState>, Vec<GpuShape>) {
    let n = states.len();
    let make = |label, bytes: &[u8], usage| {
        engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            })
    };
    let buffers = [
        make(
            "target",
            bytemuck::cast_slice(&vec![target; n * 512 * 512]),
            wgpu::BufferUsages::STORAGE,
        ),
        make(
            "canvas",
            bytemuck::cast_slice(&vec![0u32; n * 512 * 512]),
            wgpu::BufferUsages::STORAGE,
        ),
        make(
            "states",
            bytemuck::cast_slice(states),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ),
        make(
            "history",
            bytemuck::cast_slice(&vec![GpuShape::zeroed(); n * capacity]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ),
        make(
            "background",
            bytemuck::cast_slice(&vec![0u32; n]),
            wgpu::BufferUsages::STORAGE,
        ),
    ];
    let chunks = build_dispatches(
        engine,
        n,
        dispatch_images,
        [
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
        ],
    );
    let mut window = SubmissionWindow::new(max_in_flight);
    for _ in 0..steps {
        window.submit_step(engine, &chunks).unwrap();
        assert!(window.pending_count() < max_in_flight);
    }
    window.drain(engine).unwrap();
    assert_eq!(window.pending_count(), 0);
    let state = engine
        .read_buffer_range(&buffers[2], 0, (n * 1024) as u64, "states")
        .await
        .unwrap();
    let history = engine
        .read_buffer_range(&buffers[3], 0, (n * capacity * 48) as u64, "history")
        .await
        .unwrap();
    (state, history)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_state_machine_boundaries() {
    pollster::block_on(async {
        let mut c = default_shader_constants();
        c.initial_candidate_count = 4;
        c.survivor_count = 1;
        c.mutation_stagnation_limit = 4;
        c.max_mutations_per_survivor = 16;
        c.max_half_size = 8.0;
        let source = build_shader_source(&c).unwrap();
        let engines = GpuEngine::discover_all(&source).await.unwrap();
        let e = &engines[0];
        let base = || {
            let mut s = make_state(0.0, 1, 85);
            s.perturbed = 0;
            s.perturb = [0; 80];
            s
        };
        // Uneven sub-batches and queue depths preserve every byte of state and history.
        let samples: Vec<_> = (1..=7)
            .map(|seed| {
                let mut s = make_state(1e12, seed, 85);
                s.first_phase_shape_count = 85;
                s
            })
            .collect();
        let (base_states, base_history) =
            execute_scheduled(e, &samples, 895, 18, 0x808080, 7, 1).await;
        for (chunk, depth) in [(1, 1), (3, 1), (4, 2)] {
            let (states, history) =
                execute_scheduled(e, &samples, 895, 18, 0x808080, chunk, depth).await;
            assert_eq!(
                bytemuck::cast_slice::<ImageState, u8>(&base_states),
                bytemuck::cast_slice::<ImageState, u8>(&states)
            );
            assert_eq!(
                bytemuck::cast_slice::<GpuShape, u8>(&base_history),
                bytemuck::cast_slice::<GpuShape, u8>(&history)
            );
        }
        // Already satisfied regions: all 80 true-end, no shapes, ordered stage barrier.
        let mut s = base();
        s.first_phase_shape_count = 85;
        let (out, h) = execute(e, &[s], 895, 165, 0).await;
        assert_eq!(out[0].termination_reason, 1);
        assert_eq!(out[0].history_count, 160);
        assert!(out[0].closed.iter().all(|&x| x == 1));
        for (i, row) in h[..160].iter().enumerate() {
            assert_eq!(row.shape_type, if i % 2 == 0 { 2 } else { 3 });
            assert_eq!(row.pad1, if i < 32 { 1 } else { 2 });
        }
        // Last remaining region re-enters itself after interruption, retaining cumulative count.
        let mut s = base();
        s.stage = 2;
        s.phase = 4;
        s.active_grid = 16;
        s.counts[16] = 7;
        s.grid_shape_count = 7;
        s.closed = [1; 80];
        s.closed[16] = 0;
        s.region_sse = 999.0;
        let (out, h) = execute(e, &[s], 895, 2, 0).await;
        assert_eq!(h[0].shape_type, 4);
        assert_eq!(h[1].shape_type, 2);
        assert_eq!(h[1].pad2, 0);
        assert_eq!(out[0].counts[16], 7);
        assert_eq!(out[0].grid_shape_count, 7);
        assert_eq!(out[0].region_sse, 0.0);
        assert_eq!(out[0].closed[16], 0);
        // End marker closes coarse stage before entering fine; coarse counts survive.
        let mut s = base();
        s.stage = 1;
        s.phase = 3;
        s.active_grid = 15;
        s.closed[..15].fill(1);
        s.counts[15] = 20;
        let (out, h) = execute(e, &[s], 895, 2, 0).await;
        assert_eq!(h[0].pad1, 1);
        assert_eq!(h[0].shape_type, 3);
        assert_eq!(h[1].pad1, 2);
        assert_eq!(out[0].counts[15], 20);
        // A cap reached by every marker type stops without writing any additional row.
        let mut cases = Vec::new();
        for phase in [1, 3, 4] {
            let mut s = base();
            s.stage = 1;
            s.phase = phase;
            s.history_count = 894;
            cases.push(s);
        }
        let (out, h) = execute(e, &cases, 895, 3, 0).await;
        for (i, kind) in [2, 3, 4].into_iter().enumerate() {
            assert_eq!(out[i].history_count, 895);
            assert_eq!(out[i].termination_reason, 2);
            assert_eq!(h[i * 895 + 894].shape_type, kind);
        }
        // Search exhaustion is different from record-limit termination.
        let mut s = base();
        s.search_attempt_count = 2000;
        let (out, _) = execute(e, &[s], 895, 1, 0).await;
        assert_eq!(out[0].termination_reason, 3);
        assert_eq!(out[0].history_count, 0);
        // One successful shape fills the final slot; no fabricated end marker.
        let mut s = base();
        s.history_count = 894;
        let (out, h) = execute(e, &[s], 895, 3, 0x808080).await;
        assert_eq!(out[0].termination_reason, 2);
        assert_eq!(out[0].history_count, 895);
        assert!(h[894].shape_type < 2);
        // Force interventions only in this test shader. Normal and true-end precedence remain intact.
        let source = source.replace(
            "let probability = schedule_float(img_idx) * 0.10;",
            "let probability = 1.0;",
        );
        let forced = GpuEngine::discover_all(&source).await.unwrap();
        let mut cases = Vec::new();
        for (perturbed, count, id) in [(1, 0, 0), (0, 0, 0), (1, 19, 0), (1, 9, 16)] {
            let mut s = base();
            s.stage = if id < 16 { 1 } else { 2 };
            s.active_grid = id as u32;
            s.phase = 2;
            s.region_sse = 1e12;
            s.perturbed = perturbed;
            s.perturb[id] = 1;
            s.counts[id] = count;
            s.grid_shape_count = count;
            cases.push(s);
        }
        let (out, h) = execute(&forced[0], &cases, 895, 2, 0x808080).await;
        assert_eq!(h[1].shape_type, 4);
        assert!(h[895 + 1].shape_type < 2);
        assert_eq!(h[1790 + 1].shape_type, 3);
        assert_eq!(out[2].counts[0], 20);
        assert_eq!(h[2685 + 1].shape_type, 3);
        assert_eq!(out[3].counts[16], 10);
        println!("GPU state tests: ordering, 80 closures, reentry, stale RMSE, cap on all four record kinds, exhaustion, intervention and true-end priority passed");
    });
}
