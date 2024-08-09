use std::collections::HashMap;

use bevy::prelude::*;
use candle_core::{DType, Device, Tensor};

use crate::{
    agents::{Agent, PursuerAgent},
    gridworld::{LevelLayout, GRID_CELL_SIZE},
    observer::{Observable, Observer, VMSeenData},
    world_objs::NoiseSource,
};

const OBJ_DIM: usize = 8;
const MAX_OBJS: usize = 16;

#[derive(Clone, Copy)]
pub struct ObservableObject {
    pub pos: Vec2,
}

#[derive(Clone, Copy)]
pub struct NoiseSourceObject {
    pub pos: Vec2,
    pub active_radius: f32,
}

/// Encodes game data into observations for the pursuer.
///
/// The last element in the grid observation is zeroed out, this must be replaced with the localization probabilities
/// for the agent.
pub fn encode_obs(
    player_e: Entity,
    level: &Res<LevelLayout>,
    agent_state: &AgentState,
    filter_probs: &Tensor,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    // Set up observations
    let device = Device::Cpu;
    let mut obs_vec = vec![0.; 4];
    obs_vec[0] = 0.5 + agent_state.pos.x / (level.size as f32 * GRID_CELL_SIZE);
    obs_vec[1] = 0.5 + agent_state.pos.y / (level.size as f32 * GRID_CELL_SIZE);
    obs_vec[2] = agent_state.dir.x;
    obs_vec[3] = agent_state.dir.y;

    let walls = Tensor::from_slice(
        &level
            .walls
            .iter()
            .map(|x| *x as u8 as f32)
            .collect::<Vec<_>>(),
        &[level.size * level.size],
        &device,
    )?
    .reshape((level.size, level.size))?;

    let mut obs_vecs = vec![vec![0.; OBJ_DIM]; MAX_OBJS];
    for (i, e) in agent_state.observing.iter().enumerate() {
        if agent_state.vm_data.contains_key(e) {
            let obs_obj = agent_state.objects.get(e).unwrap();
            let mut obj_features = vec![0.; OBJ_DIM];
            obj_features[0] = 0.5 + obs_obj.pos.x / (level.size as f32 * GRID_CELL_SIZE);
            obj_features[1] = 0.5 + obs_obj.pos.y / (level.size as f32 * GRID_CELL_SIZE);
            obj_features[2] = 1.;
            let vm_data = agent_state.vm_data[e];
            obj_features[5] = vm_data.last_seen_elapsed / 10.0;
            obj_features[6] = obs_obj.pos.x - vm_data.last_pos.x;
            obj_features[7] = obs_obj.pos.y - vm_data.last_pos.y;
            obs_vecs[i] = obj_features;
        }
    }
    for (i, e) in agent_state.listening.iter().enumerate() {
        let obj_noise = agent_state.noise_sources.get(e).unwrap();
        let mut obj_features = vec![0.; OBJ_DIM];
        obj_features[0] = 0.5 + obj_noise.pos.x / (level.size as f32 * GRID_CELL_SIZE);
        obj_features[1] = 0.5 + obj_noise.pos.y / (level.size as f32 * GRID_CELL_SIZE);
        obj_features[3] = 1.;
        obj_features[4] = obj_noise.active_radius;
        obs_vecs[i + agent_state.observing.len()] = obj_features;
    }

    let mut attn_mask = vec![0.; MAX_OBJS];
    let num_objs = agent_state.observing.len() + agent_state.listening.len();
    for i in num_objs..attn_mask.len() {
        attn_mask[i] = 1.;
    }
    let filter_probs = filter_probs.reshape(&[level.size, level.size])?;
    let grid = Tensor::stack(
        &[
            &walls,
            &filter_probs,
            &Tensor::zeros(walls.shape(), DType::F32, &device).unwrap(),
        ],
        0,
    )?;

    // Combine scalar observations with grid
    let scalar_grid = Tensor::from_slice(&obs_vec, &[obs_vec.len()], &device)?
        .reshape(&[4, 1, 1])?
        .repeat(&[1, level.size, level.size])?;
    let grid = Tensor::cat(&[&scalar_grid, &grid], 0)?;

    Ok((
        grid,
        Tensor::stack(
            &obs_vecs
                .iter()
                .map(|s| Tensor::from_slice(s, &[OBJ_DIM], &device).unwrap())
                .collect::<Vec<_>>(),
            0,
        )?,
        Tensor::from_slice(&attn_mask, &[MAX_OBJS], &device)?,
    ))
}

#[derive(Clone)]
pub struct AgentState {
    pub pos: Vec2,
    pub dir: Vec2,
    pub observing: Vec<Entity>,
    pub listening: Vec<Entity>,
    pub vm_data: HashMap<Entity, VMSeenData>,
    pub visible_cells: Vec<f32>,
    pub objects: HashMap<Entity, ObservableObject>,
    pub noise_sources: HashMap<Entity, NoiseSourceObject>,
}

/// Encodes information from the world into an agent's state.
/// This can be further processed to yield Tensor observations.
pub fn encode_state(
    pursuer_query: &Query<(&Agent, &GlobalTransform, &Observer), With<PursuerAgent>>,
    listening_query: &Query<(Entity, &GlobalTransform, &NoiseSource)>,
    level: &Res<LevelLayout>,

    observable_query: &Query<(Entity, &GlobalTransform), With<Observable>>,
    noise_query: &Query<(Entity, &GlobalTransform, &NoiseSource)>,
) -> AgentState {
    // Encode global state stuff
    let mut objects = HashMap::new();
    for (e, xform) in observable_query.iter() {
        objects.insert(
            e,
            ObservableObject {
                pos: xform.translation().xy(),
            },
        );
    }

    let mut noise_sources = HashMap::new();
    for (e, xform, noise_src) in noise_query.iter() {
        noise_sources.insert(
            e,
            NoiseSourceObject {
                pos: xform.translation().xy(),
                active_radius: noise_src.active_radius,
            },
        );
    }

    let (agent, &xform, observer) = pursuer_query.single();
    let vis_mesh = observer.vis_mesh.clone();
    let pos = xform.translation().xy();
    let dir = agent.dir;
    let observing = observer.observing.clone();
    let vm_data = observer
        .seen_markers
        .iter()
        .map(|(e, vm_data)| (*e, *vm_data))
        .collect();

    let listening = listening_query
        .iter()
        .filter(|(_, noise_xform, noise_src)| {
            (xform.translation().xy() - noise_xform.translation().xy()).length_squared()
                <= noise_src.noise_radius.powi(2)
                && noise_src.activated_by_player
        })
        .map(|(e, _, _)| e)
        .collect();
    let size = level.size;

    // Compute intersection of agent visible area with grid.
    // We need to supersample to handle edges.
    let visible_scale = 4;
    let mut visible_cells_ss = vec![false; (size * visible_scale).pow(2)];
    for tri in &vis_mesh {
        let mut points = tri.to_vec();
        points.sort_by(|p1, p2| p1.y.total_cmp(&p2.y)); // 2 is top, 0 is bottom
        let slope = (points[2].x - points[0].x) / (points[2].y - points[0].y);
        let mid_point = Vec2::new(
            points[0].x + slope * (points[1].y - points[0].y),
            points[1].y,
        );

        let mut mid_points = [points[1], mid_point];
        mid_points.sort_by(|p1, p2| p1.x.total_cmp(&p2.x));

        fill_tri_half(
            &mut visible_cells_ss,
            mid_points[0],
            mid_points[1],
            points[2],
            true,
            size * visible_scale,
            GRID_CELL_SIZE / visible_scale as f32,
        );
        fill_tri_half(
            &mut visible_cells_ss,
            mid_points[0],
            mid_points[1],
            points[0],
            false,
            size * visible_scale,
            GRID_CELL_SIZE / visible_scale as f32,
        );
    }
    let mut visible_cells = vec![0.; size.pow(2)];
    for y in 0..size {
        for x in 0..size {
            let mut value = 0.;
            for sy in 0..visible_scale {
                for sx in 0..visible_scale {
                    value += visible_cells_ss[(y * visible_scale + sy) * (size * visible_scale)
                        + (x * visible_scale + sx)] as u8 as f32;
                }
            }
            visible_cells[y * size + x] = value / visible_scale.pow(2) as f32;
        }
    }

    AgentState {
        pos,
        dir,
        observing,
        listening,
        vm_data,
        visible_cells,
        objects,
        noise_sources,
    }
}

/// Fills in half a triangle.
pub fn fill_tri_half(
    visible_cells: &mut [bool],
    mid1: Vec2,
    mid2: Vec2,
    other: Vec2,
    is_top: bool,
    size: usize,
    cell_size: f32,
) {
    let slope1 = (other.x - mid1.x) / (other.y - mid1.y);
    let slope2 = (other.x - mid2.x) / (other.y - mid2.y);
    let dy = cell_size;
    let (mut last1, mut last2) = if is_top { (mid1, mid2) } else { (other, other) };
    for _ in 0..((if is_top {
        other.y - mid1.y
    } else {
        mid1.y - other.y
    } / dy)
        .ceil() as u32)
    {
        let y = ((last1.y / cell_size).round() as usize).clamp(0, size - 1);
        for x in ((last1.x / cell_size).floor() as usize)..((last2.x / cell_size).ceil() as usize) {
            visible_cells[y * size + x.clamp(0, size - 1)] = true;
        }

        last1.x += slope1 * dy;
        last1.y += dy;
        last2.x += slope2 * dy;
        last2.y += dy;
    }
}
