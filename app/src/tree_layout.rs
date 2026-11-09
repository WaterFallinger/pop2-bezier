//! Stable, non-recursive layout of full, sampled, or branch-only state trees.
//! All coordinates are logical world units; camera pan/zoom belongs to the UI.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub id: u64,
    pub parent: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreeDisplayMode {
    #[default]
    All,
    EmphasizeBranches,
    BranchesOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct LayoutOptions {
    pub display_mode: TreeDisplayMode,
    /// Legacy callers using `compact: true` and the default mode retain the
    /// original branch-only preview, including each branch's first child.
    pub compact: bool,
    pub node_width: f32,
    pub node_height: f32,
    pub horizontal_gap: f32,
    pub vertical_gap: f32,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            display_mode: TreeDisplayMode::All,
            compact: false,
            node_width: 220.0,
            node_height: 66.0,
            horizontal_gap: 38.0,
            vertical_gap: 30.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayoutBounds {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl LayoutBounds {
    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }
    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }
    pub fn center(&self) -> (f32, f32) {
        (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutNode {
    pub id: u64,
    /// Center, not top-left, in logical world coordinates.
    pub x: f32,
    pub y: f32,
    /// Depth in the displayed tree. Hidden chain steps consume no row.
    pub depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutEdge {
    pub parent: u64,
    pub child: u64,
    /// Original nodes skipped between these two visible endpoints.
    pub hidden_steps: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TreeLayout {
    pub nodes: Vec<LayoutNode>,
    pub edges: Vec<LayoutEdge>,
    pub bounds: LayoutBounds,
    pub roots: Vec<u64>,
    /// Invalid references are repaired deterministically instead of hanging.
    pub diagnostics: Vec<String>,
}

impl TreeLayout {
    pub fn node(&self, id: u64) -> Option<&LayoutNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

fn positive(value: f32, default: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value.min(100_000.0)
    } else {
        default
    }
}

fn gap(value: f32, default: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value.min(100_000.0)
    } else {
        default
    }
}

/// Sibling subtrees occupy disjoint horizontal intervals. A parent's center is
/// the midpoint of its first/last displayed child centers, including when an
/// old long branch acquires a new sibling. Input order fixes sibling order.
/// A single root is always at x=0; forest root centers are centered as a group.
pub fn build_layout(input: &[TreeNode], options: LayoutOptions) -> TreeLayout {
    let defaults = LayoutOptions::default();
    let display_mode = if options.compact && options.display_mode == TreeDisplayMode::All {
        TreeDisplayMode::BranchesOnly
    } else {
        options.display_mode
    };
    let node_width = positive(options.node_width, defaults.node_width);
    let node_height = positive(options.node_height, defaults.node_height);
    let horizontal_gap = gap(options.horizontal_gap, defaults.horizontal_gap);
    let vertical_gap = gap(options.vertical_gap, defaults.vertical_gap);
    let mut result = TreeLayout::default();
    let mut ids = HashMap::new();
    let mut nodes = Vec::new();
    for node in input {
        if ids.contains_key(&node.id) {
            diagnostic(
                &mut result.diagnostics,
                format!("重复节点 {}：保留首次出现的记录", node.id),
            );
        } else {
            ids.insert(node.id, nodes.len());
            nodes.push(*node);
        }
    }
    let count = nodes.len();
    if count == 0 {
        return result;
    }
    let mut parents = Vec::with_capacity(count);
    for node in &nodes {
        let parent = node.parent.and_then(|id| ids.get(&id).copied());
        if node.parent.is_some() && parent.is_none() {
            diagnostic(
                &mut result.diagnostics,
                format!("节点 {} 的父节点不存在：作为根节点显示", node.id),
            );
        }
        parents.push(parent);
    }

    // Every item has at most one parent. Follow parent chains once and cut the
    // earliest input item inside each cycle. No recursive traversal is used.
    let mut states = vec![0u8; count];
    for start in 0..count {
        if states[start] != 0 {
            continue;
        }
        let mut path = Vec::new();
        let mut cursor = Some(start);
        while let Some(node) = cursor {
            match states[node] {
                0 => {
                    states[node] = 1;
                    path.push(node);
                    cursor = parents[node];
                }
                1 => {
                    let cycle_start = path.iter().position(|&i| i == node).unwrap_or(0);
                    let cut = *path[cycle_start..].iter().min().unwrap_or(&node);
                    parents[cut] = None;
                    diagnostic(
                        &mut result.diagnostics,
                        format!("节点 {} 所在父链成环：断开该节点的父连线", nodes[cut].id),
                    );
                    break;
                }
                _ => break,
            }
        }
        for node in path {
            states[node] = 2;
        }
    }

    let mut children = vec![Vec::new(); count];
    let mut roots = Vec::new();
    for (node, parent) in parents.iter().copied().enumerate() {
        if let Some(parent) = parent {
            children[parent].push(node);
        } else {
            roots.push(node);
        }
    }
    let mut preorder = Vec::with_capacity(count);
    let mut stack: Vec<usize> = roots.iter().rev().copied().collect();
    while let Some(node) = stack.pop() {
        preorder.push(node);
        stack.extend(children[node].iter().rev().copied());
    }

    let mut visible = vec![false; count];
    match display_mode {
        TreeDisplayMode::All => visible.fill(true),
        TreeDisplayMode::BranchesOnly => {
            // Retain the established compact-preview behavior: roots, leaves,
            // forks, and the immediate first step of each new branch.
            for node in 0..count {
                if parents[node].is_none() || children[node].is_empty() || children[node].len() >= 2
                {
                    visible[node] = true;
                }
                if children[node].len() >= 2 {
                    for &child in &children[node] {
                        visible[child] = true;
                    }
                }
            }
        }
        TreeDisplayMode::EmphasizeBranches => {
            // Count actual parent edges within each single-child chain, not
            // IDs or traversal positions. Each fork restarts all outgoing
            // chains, and each fiftieth step becomes a visible sample.
            let mut steps_since_anchor = vec![0usize; count];
            for &node in &preorder {
                let is_anchor = parents[node].is_none()
                    || children[node].is_empty()
                    || children[node].len() >= 2;
                let steps = parents[node]
                    .map(|parent| steps_since_anchor[parent] + 1)
                    .unwrap_or(0);
                if is_anchor || steps == 50 {
                    visible[node] = true;
                } else {
                    steps_since_anchor[node] = steps;
                }
            }
        }
    }
    let mut nearest_visible: Vec<Option<usize>> = vec![None; count];
    let mut hidden_since_visible = vec![0usize; count];
    let mut displayed_parents: Vec<Option<usize>> = vec![None; count];
    let mut displayed_children: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut depths = vec![0usize; count];
    for &node in &preorder {
        if visible[node] {
            if let Some(parent) = parents[node] {
                if let Some(ancestor) = nearest_visible[parent] {
                    displayed_parents[node] = Some(ancestor);
                    displayed_children[ancestor].push(node);
                    depths[node] = depths[ancestor] + 1;
                    result.edges.push(LayoutEdge {
                        parent: nodes[ancestor].id,
                        child: nodes[node].id,
                        hidden_steps: hidden_since_visible[parent],
                    });
                }
            }
            nearest_visible[node] = Some(node);
        } else if let Some(parent) = parents[node] {
            nearest_visible[node] = nearest_visible[parent];
            hidden_since_visible[node] = hidden_since_visible[parent] + 1;
        }
    }

    // Compute each complete subtree's left/right extents relative to its root.
    // Child roots need not sit in the middle of their subtree intervals, so
    // centering only by a count of leaves would misplace asymmetric branches.
    let mut left_extent = vec![node_width * 0.5; count];
    let mut right_extent = vec![node_width * 0.5; count];
    let mut child_offsets: Vec<Vec<f32>> = vec![Vec::new(); count];
    for &node in preorder.iter().rev() {
        if !visible[node] || displayed_children[node].is_empty() {
            continue;
        }
        let mut cursor = 0.0;
        let mut child_centers = Vec::with_capacity(displayed_children[node].len());
        for &child in &displayed_children[node] {
            child_centers.push(cursor + left_extent[child]);
            cursor += left_extent[child] + right_extent[child] + horizontal_gap;
        }
        let children_right = cursor - horizontal_gap;
        let parent_center = (child_centers[0] + child_centers[child_centers.len() - 1]) * 0.5;
        let left = 0.0f32.min(parent_center - node_width * 0.5);
        let right = children_right.max(parent_center + node_width * 0.5);
        left_extent[node] = parent_center - left;
        right_extent[node] = right - parent_center;
        child_offsets[node] = child_centers
            .into_iter()
            .map(|x| x - parent_center)
            .collect();
    }

    let displayed_roots: Vec<usize> = preorder
        .iter()
        .copied()
        .filter(|&n| visible[n] && displayed_parents[n].is_none())
        .collect();
    let mut positions = vec![0.0; count];
    let mut cursor = 0.0;
    for &root in &displayed_roots {
        positions[root] = cursor + left_extent[root];
        cursor += left_extent[root] + right_extent[root] + horizontal_gap;
    }
    let center = displayed_roots
        .first()
        .zip(displayed_roots.last())
        .map(|(&a, &b)| (positions[a] + positions[b]) * 0.5)
        .unwrap_or(0.0);
    for &root in &displayed_roots {
        positions[root] -= center;
    }
    for &node in &preorder {
        if !visible[node] {
            continue;
        }
        for (&child, &offset) in displayed_children[node].iter().zip(&child_offsets[node]) {
            positions[child] = positions[node] + offset;
        }
    }

    let mut bounds = LayoutBounds {
        min_x: f32::INFINITY,
        min_y: f32::INFINITY,
        max_x: f32::NEG_INFINITY,
        max_y: f32::NEG_INFINITY,
    };
    // Keep node output in canonical input order for stable draw/pick order.
    for node in 0..count {
        if !visible[node] {
            continue;
        }
        let x = positions[node];
        let y = depths[node] as f32 * (node_height + vertical_gap);
        bounds.min_x = bounds.min_x.min(x - node_width * 0.5);
        bounds.max_x = bounds.max_x.max(x + node_width * 0.5);
        bounds.min_y = bounds.min_y.min(y - node_height * 0.5);
        bounds.max_y = bounds.max_y.max(y + node_height * 0.5);
        result.nodes.push(LayoutNode {
            id: nodes[node].id,
            x,
            y,
            depth: depths[node],
        });
    }
    result.bounds = bounds;
    result.roots = displayed_roots.iter().map(|&root| nodes[root].id).collect();
    result
}

fn diagnostic(messages: &mut Vec<String>, message: String) {
    // Corrupt input must not turn a preview into an unbounded diagnostic list.
    if messages.len() < 32 {
        messages.push(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, parent: Option<u64>) -> TreeNode {
        TreeNode { id, parent }
    }
    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 0.01, "{a} != {b}");
    }

    #[test]
    fn two_and_three_children_are_centered_around_root() {
        let two = [node(0, None), node(1, Some(0)), node(2, Some(0))];
        let layout = build_layout(&two, LayoutOptions::default());
        close(layout.node(0).unwrap().x, 0.0);
        assert!(layout.node(1).unwrap().x < 0.0 && layout.node(2).unwrap().x > 0.0);
        close(
            (layout.node(1).unwrap().x + layout.node(2).unwrap().x) / 2.0,
            layout.node(0).unwrap().x,
        );
        let three = [
            node(0, None),
            node(1, Some(0)),
            node(2, Some(0)),
            node(3, Some(0)),
        ];
        let layout = build_layout(&three, LayoutOptions::default());
        close(layout.node(2).unwrap().x, 0.0);
        close(layout.node(1).unwrap().x, -layout.node(3).unwrap().x);
    }

    #[test]
    fn adding_a_fork_moves_old_and_new_subtrees_to_opposite_sides() {
        let before = [
            node(0, None),
            node(1, Some(0)),
            node(2, Some(1)),
            node(3, Some(2)),
        ];
        let original = build_layout(&before, LayoutOptions::default());
        close(original.node(2).unwrap().x, 0.0);
        let after = [
            node(0, None),
            node(1, Some(0)),
            node(2, Some(1)),
            node(3, Some(2)),
            node(4, Some(1)),
            node(5, Some(4)),
        ];
        let forked = build_layout(&after, LayoutOptions::default());
        let branch = forked.node(1).unwrap().x;
        assert!(forked.node(2).unwrap().x < branch && forked.node(4).unwrap().x > branch);
        assert!(forked.node(3).unwrap().x < branch && forked.node(5).unwrap().x > branch);
        close(
            (forked.node(2).unwrap().x + forked.node(4).unwrap().x) * 0.5,
            branch,
        );
    }

    #[test]
    fn asymmetric_subtrees_do_not_overlap_and_all_parents_center_on_children() {
        let input = [
            node(0, None),
            node(1, Some(0)),
            node(2, Some(0)),
            node(3, Some(1)),
            node(4, Some(1)),
            node(5, Some(1)),
            node(6, Some(2)),
            node(7, Some(6)),
            node(8, Some(6)),
            node(9, Some(8)),
            node(10, Some(8)),
        ];
        let options = LayoutOptions::default();
        let layout = build_layout(&input, options);
        for parent in &input {
            let children: Vec<_> = input
                .iter()
                .filter(|n| n.parent == Some(parent.id))
                .collect();
            if !children.is_empty() {
                let first = layout.node(children[0].id).unwrap().x;
                let last = layout.node(children[children.len() - 1].id).unwrap().x;
                close(layout.node(parent.id).unwrap().x, (first + last) * 0.5);
            }
        }
        for a in &layout.nodes {
            for b in &layout.nodes {
                if a.id != b.id && a.depth == b.depth {
                    assert!(
                        (a.x - b.x).abs() >= options.node_width + options.horizontal_gap - 0.01
                    );
                }
            }
        }
    }

    #[test]
    fn compact_chain_retains_only_root_and_leaf_and_compresses_depth() {
        let input: Vec<_> = (0..8)
            .map(|id| node(id, if id == 0 { None } else { Some(id - 1) }))
            .collect();
        let layout = build_layout(
            &input,
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        assert_eq!(
            layout.nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
            vec![0, 7]
        );
        assert_eq!(
            layout.edges,
            vec![LayoutEdge {
                parent: 0,
                child: 7,
                hidden_steps: 6
            }]
        );
        assert_eq!(layout.node(7).unwrap().depth, 1);
        assert!(layout.node(7).unwrap().y < 2.0 * (66.0 + 30.0));
    }

    #[test]
    fn compact_branch_keeps_immediate_children_and_nearest_visible_ancestors() {
        let input = [
            node(0, None),
            node(1, Some(0)),
            node(2, Some(1)),
            node(3, Some(2)),
            node(4, Some(3)),
            node(5, Some(4)),
            node(6, Some(2)),
            node(7, Some(6)),
            node(8, Some(7)),
        ];
        let layout = build_layout(
            &input,
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        assert_eq!(
            layout.nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
            vec![0, 2, 3, 5, 6, 8]
        );
        assert!(layout.edges.contains(&LayoutEdge {
            parent: 0,
            child: 2,
            hidden_steps: 1
        }));
        assert!(layout.edges.contains(&LayoutEdge {
            parent: 2,
            child: 3,
            hidden_steps: 0
        }));
        assert!(layout.edges.contains(&LayoutEdge {
            parent: 2,
            child: 6,
            hidden_steps: 0
        }));
        assert!(layout.edges.contains(&LayoutEdge {
            parent: 3,
            child: 5,
            hidden_steps: 1
        }));
        assert!(layout.edges.contains(&LayoutEdge {
            parent: 6,
            child: 8,
            hidden_steps: 1
        }));
        assert_eq!(layout.node(8).unwrap().depth, 3);
        assert!(layout.node(3).unwrap().x < layout.node(2).unwrap().x);
        assert!(layout.node(6).unwrap().x > layout.node(2).unwrap().x);
    }

    #[test]
    fn emphasized_chains_sample_exactly_every_fifty_parent_edges() {
        for last in [0, 1, 49, 50, 51, 99, 100, 101] {
            let input: Vec<_> = (0..=last).map(|id| node(id, id.checked_sub(1))).collect();
            let layout = build_layout(
                &input,
                LayoutOptions {
                    display_mode: TreeDisplayMode::EmphasizeBranches,
                    ..Default::default()
                },
            );
            let expected: Vec<_> = (0..=last)
                .filter(|id| *id == 0 || *id == last || id % 50 == 0)
                .collect();
            assert_eq!(
                layout.nodes.iter().map(|item| item.id).collect::<Vec<_>>(),
                expected,
                "chain ending at step {last}"
            );
            assert_eq!(layout.edges.len(), expected.len().saturating_sub(1));
            for (depth, &id) in expected.iter().enumerate() {
                assert_eq!(layout.node(id).unwrap().depth, depth);
            }
            for pair in expected.windows(2) {
                assert!(layout.edges.contains(&LayoutEdge {
                    parent: pair[0],
                    child: pair[1],
                    hidden_steps: (pair[1] - pair[0] - 1) as usize,
                }));
            }
        }
    }

    #[test]
    fn emphasized_sampling_restarts_at_each_fork_for_each_outgoing_chain() {
        let mut input: Vec<_> = (0..=30).map(|id| node(id, id.checked_sub(1))).collect();
        // A second fork occurs only 37 steps after the first fork. Its two
        // outgoing chains must both restart their own fifty-step counters.
        for step in 1..=101 {
            input.push(node(
                1000 + step,
                Some(if step == 1 { 30 } else { 999 + step }),
            ));
        }
        for step in 1..=49 {
            input.push(node(
                2000 + step,
                Some(if step == 1 { 30 } else { 1999 + step }),
            ));
        }
        for step in 1..=51 {
            input.push(node(
                9000 + step,
                Some(if step == 1 { 1037 } else { 8999 + step }),
            ));
        }
        let layout = build_layout(
            &input,
            LayoutOptions {
                display_mode: TreeDisplayMode::EmphasizeBranches,
                ..Default::default()
            },
        );
        assert_eq!(
            layout.nodes.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 30, 1037, 1087, 1101, 2049, 9050, 9051]
        );
        for (parent, child, hidden_steps) in [
            (0, 30, 29),
            (30, 1037, 36),
            (1037, 1087, 49),
            (1087, 1101, 13),
            (30, 2049, 48),
            (1037, 9050, 49),
            (9050, 9051, 0),
        ] {
            assert!(layout.edges.contains(&LayoutEdge {
                parent,
                child,
                hidden_steps
            }));
        }
        for id in [1, 31, 1001, 1038, 1050, 2001, 9001] {
            assert!(
                layout.node(id).is_none(),
                "short chain or fork's first child {id} must remain hidden"
            );
        }
        assert_eq!(layout.node(1101).unwrap().depth, 4);
        let fork = layout.node(1037).unwrap();
        close(
            fork.x,
            (layout.node(1087).unwrap().x + layout.node(9050).unwrap().x) * 0.5,
        );
    }

    #[test]
    fn emphasized_sampling_uses_parent_links_with_sparse_ids_and_reversed_input() {
        let id_for_step = |step: u64| 10_000 + (step * 137) % 10_007;
        let mut input: Vec<_> = (0..=101)
            .map(|step| node(id_for_step(step), step.checked_sub(1).map(id_for_step)))
            .collect();
        input.reverse();
        let layout = build_layout(
            &input,
            LayoutOptions {
                display_mode: TreeDisplayMode::EmphasizeBranches,
                ..Default::default()
            },
        );
        assert!(layout.diagnostics.is_empty());
        assert_eq!(layout.roots, vec![id_for_step(0)]);
        assert_eq!(
            layout.nodes.iter().map(|item| item.id).collect::<Vec<_>>(),
            [101, 100, 50, 0].map(id_for_step)
        );
        assert_eq!(
            layout.edges,
            vec![
                LayoutEdge {
                    parent: id_for_step(0),
                    child: id_for_step(50),
                    hidden_steps: 49
                },
                LayoutEdge {
                    parent: id_for_step(50),
                    child: id_for_step(100),
                    hidden_steps: 49
                },
                LayoutEdge {
                    parent: id_for_step(100),
                    child: id_for_step(101),
                    hidden_steps: 0
                },
            ]
        );
        assert_eq!(layout.node(id_for_step(101)).unwrap().depth, 3);
    }

    #[test]
    fn explicit_modes_and_legacy_compact_options_remain_compatible() {
        let mut input: Vec<_> = (0..=60).map(|id| node(id, id.checked_sub(1))).collect();
        input.push(node(999, Some(2)));
        assert_eq!(TreeDisplayMode::default(), TreeDisplayMode::All);
        assert_eq!(
            build_layout(&input, LayoutOptions::default()).nodes.len(),
            input.len()
        );
        let legacy = build_layout(
            &input,
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        let branch_only = build_layout(
            &input,
            LayoutOptions {
                display_mode: TreeDisplayMode::BranchesOnly,
                ..Default::default()
            },
        );
        assert_eq!(legacy.nodes, branch_only.nodes);
        assert_eq!(legacy.edges, branch_only.edges);
        assert_eq!(
            branch_only
                .nodes
                .iter()
                .map(|item| item.id)
                .collect::<Vec<_>>(),
            vec![0, 2, 3, 60, 999]
        );
        let emphasized = build_layout(
            &input,
            LayoutOptions {
                display_mode: TreeDisplayMode::EmphasizeBranches,
                compact: true,
                ..Default::default()
            },
        );
        assert_eq!(
            emphasized
                .nodes
                .iter()
                .map(|item| item.id)
                .collect::<Vec<_>>(),
            vec![0, 2, 52, 60, 999]
        );
        assert_eq!(
            emphasized.nodes,
            build_layout(
                &input,
                LayoutOptions {
                    display_mode: TreeDisplayMode::EmphasizeBranches,
                    ..Default::default()
                }
            )
            .nodes
        );
    }

    #[test]
    fn invalid_forest_is_bounded_deterministic_and_keeps_unique_nodes() {
        let input = [
            node(10, Some(11)),
            node(11, Some(10)),
            node(12, Some(99)),
            node(13, Some(13)),
            node(10, None),
            node(14, Some(11)),
        ];
        let options = LayoutOptions {
            node_width: f32::NAN,
            vertical_gap: -5.0,
            ..Default::default()
        };
        let a = build_layout(&input, options);
        let b = build_layout(&input, options);
        assert_eq!(a.nodes.len(), 5);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.edges, b.edges);
        assert!(!a.diagnostics.is_empty());
        assert!(a.bounds.width().is_finite() && a.bounds.height().is_finite());
        assert_eq!(a.roots, vec![10, 12, 13]);
        assert_eq!(a.edges.len(), a.nodes.len() - a.roots.len());
    }

    #[test]
    fn long_chain_uses_no_recursive_stack_and_singleton_is_valid() {
        let input: Vec<_> = (0..50_000)
            .map(|id| node(id, if id == 0 { None } else { Some(id - 1) }))
            .collect();
        let compact = build_layout(
            &input,
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        assert_eq!(compact.nodes.len(), 2);
        assert_eq!(compact.edges[0].hidden_steps, 49_998);
        let sampled = build_layout(
            &input,
            LayoutOptions {
                display_mode: TreeDisplayMode::EmphasizeBranches,
                ..Default::default()
            },
        );
        assert_eq!(sampled.nodes.len(), 1001);
        assert_eq!(sampled.edges.len(), 1000);
        assert_eq!(
            sampled.edges[0],
            LayoutEdge {
                parent: 0,
                child: 50,
                hidden_steps: 49
            }
        );
        assert_eq!(
            sampled.edges.last().unwrap(),
            &LayoutEdge {
                parent: 49_950,
                child: 49_999,
                hidden_steps: 48
            }
        );
        assert_eq!(sampled.node(49_999).unwrap().depth, 1000);
        let singleton = build_layout(
            &[node(42, None)],
            LayoutOptions {
                compact: true,
                ..Default::default()
            },
        );
        assert_eq!(
            singleton.nodes,
            vec![LayoutNode {
                id: 42,
                x: 0.0,
                y: 0.0,
                depth: 0
            }]
        );
        assert_eq!(
            singleton.bounds.width(),
            LayoutOptions::default().node_width
        );
        assert!(build_layout(&[], LayoutOptions::default()).nodes.is_empty());
    }
}
