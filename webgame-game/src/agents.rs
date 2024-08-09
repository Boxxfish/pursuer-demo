use std::{
    collections::{HashMap, VecDeque},
    f32::consts::PI,
    time::Duration,
};

use bevy::{prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::control::KinematicCharacterController;
use candle_core::Tensor;
use rand::distributions::Distribution;

use crate::{
    filter::{pos_to_grid, BayesFilter},
    gridworld::{LevelLayout, ShouldRun, GRID_CELL_SIZE},
    models::PolicyNet,
    net::{load_weights_into_net, NNWrapper},
    observations::{encode_obs, encode_state, AgentState},
    observer::{update_vm_data, Observable, Observer},
    world_objs::{NoiseSource, QuadMesh},
};

/// Plugin for agent stuff.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            ((
                move_agents,
                visualize_agent::<PursuerAgent>(Color::RED),
                visualize_agent::<PlayerAgent>(Color::GREEN),
            )
                .run_if(resource_exists::<ShouldRun>),),
        );
    }
}

/// Adds playable functionality for agents.
pub struct AgentPlayPlugin;

impl Plugin for AgentPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, add_materials).add_systems(
            Update,
            (
                (
                    set_player_action,
                    set_pursuer_action_neural,
                    set_pursuer_action_pathfinding,
                    visualize_act_probs,
                    update_observations.after(update_vm_data),
                )
                    .run_if(resource_exists::<ShouldRun>),
                load_weights_into_net::<PolicyNet>,
            ),
        );
    }
}

/// State used by all agents.
#[derive(Component, Clone, Copy)]
pub struct Agent {
    /// The direction the agent is currently looking at.
    pub dir: Vec2,
}

impl Default for Agent {
    fn default() -> Self {
        Self { dir: Vec2::X }
    }
}

// Indicates the Pursuer agent.
#[derive(Component)]
pub struct PursuerAgent {
    pub observations: Option<(Tensor, Option<Tensor>, Option<Tensor>)>,
    pub obs_timer: Timer,
    pub agent_state: Option<AgentState>,
    pub action_probs: Vec<f32>,
}

impl Default for PursuerAgent {
    fn default() -> Self {
        Self {
            observations: None,
            obs_timer: Timer::from_seconds(0.4, TimerMode::Repeating),
            agent_state: None,
            action_probs: Vec::new(),
        }
    }
}

/// Denotes that the Pursuer should be controlled by a neural network.
#[derive(Component)]
pub struct NeuralPolicy;

/// Denotes that the Pursuer should use pathfinding.
#[derive(Component)]
pub struct PathfindingPolicy;

/// Updates the Pursuer's observations.
#[allow(clippy::too_many_arguments)]
pub fn update_observations(
    mut pursuer_query: Query<&mut PursuerAgent>,
    p_query: Query<(&Agent, &GlobalTransform, &Observer), With<PursuerAgent>>,
    time: Res<Time>,
    observable_query: Query<(Entity, &GlobalTransform), With<Observable>>,
    filter_query: Query<&BayesFilter>,
    noise_query: Query<(Entity, &GlobalTransform, &NoiseSource)>,
    level: Res<LevelLayout>,
    player_query: Query<Entity, With<PlayerAgent>>,
    listening_query: Query<(Entity, &GlobalTransform, &NoiseSource)>,
) {
    for mut pursuer in pursuer_query.iter_mut() {
        if let Ok(player_e) = player_query.get_single() {
            pursuer.obs_timer.tick(time.delta());
            if pursuer.obs_timer.just_finished() {
                // Encode observations
                let agent_state = encode_state(
                    &p_query,
                    &listening_query,
                    &level,
                    &observable_query,
                    &noise_query,
                );
                let filter = filter_query.single();
                let (grid, objs, objs_attn) =
                    encode_obs(player_e, &level, &agent_state, &filter.probs).unwrap();
                pursuer.observations = Some((grid, Some(objs), Some(objs_attn)));
                pursuer.agent_state = Some(agent_state);
            }
        }
    }
}

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// The child of an `Agent` that contains its visuals.
#[derive(Component)]
pub struct AgentVisuals;

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(
    Commands<'_, '_>,
    Query<'_, '_, Entity, Added<T>>,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
) {
    move |mut commands: Commands,
          agent_query: Query<Entity, Added<T>>,
          mut meshes: ResMut<Assets<Mesh>>,
          mut materials: ResMut<Assets<ColorMaterial>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Mesh2dHandle(meshes.add(Circle::new(GRID_CELL_SIZE * 0.25))),
                materials.add(color),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
        }
    }
}

/// A visual for showing action probabilities.
#[derive(Component)]
struct ActProbVis {
    pub action_id: usize,
}

#[derive(Resource)]
struct ArrowMat(Handle<StandardMaterial>);

fn add_materials(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let material = StandardMaterial {
        base_color: Color::BLACK,
        base_color_texture: Some(asset_server.load("arrow.png")),
        alpha_mode: AlphaMode::Blend,
        ..default()
    };
    let mat = ArrowMat(materials.add(material));
    commands.insert_resource(mat);
}

/// Shows action probabilities.
fn visualize_act_probs(
    pursuer_query: Query<(Entity, &PursuerAgent)>,
    mut vis_query: Query<(&ActProbVis, &Parent, &mut Transform)>,
    mut commands: Commands,
    quad_mesh: Res<QuadMesh>,
    arrow_mat: Res<ArrowMat>,
) {
    let offset = 16.;
    let offset_z = 10.;
    if vis_query.is_empty() {
        for (pursuer_e, _) in pursuer_query.iter() {
            commands.entity(pursuer_e).with_children(|p| {
                for action_id in 0..9 {
                    let transform = match action_id {
                        0 => Transform::default(),
                        1 => Transform::from_translation((Vec2::Y * offset).extend(offset_z))
                            .with_rotation(Quat::from_rotation_z(PI)),
                        2 => Transform::from_translation(
                            ((Vec2::Y + Vec2::X).normalize() * offset).extend(offset_z),
                        )
                        .with_rotation(Quat::from_rotation_z(3. * PI / 4.)),
                        3 => Transform::from_translation((Vec2::X * offset).extend(offset_z))
                            .with_rotation(Quat::from_rotation_z(PI / 2.)),
                        4 => Transform::from_translation(
                            ((-Vec2::Y + Vec2::X).normalize() * offset).extend(offset_z),
                        )
                        .with_rotation(Quat::from_rotation_z(PI / 4.)),
                        5 => Transform::from_translation((-Vec2::Y * offset).extend(offset_z)),
                        6 => Transform::from_translation(
                            ((-Vec2::Y + -Vec2::X).normalize() * offset).extend(offset_z),
                        )
                        .with_rotation(Quat::from_rotation_z(7. * PI / 4.)),
                        7 => Transform::from_translation((-Vec2::X * offset).extend(offset_z))
                            .with_rotation(Quat::from_rotation_z(3. * PI / 2.)),
                        8 => Transform::from_translation(
                            ((Vec2::Y + -Vec2::X).normalize() * offset).extend(offset_z),
                        )
                        .with_rotation(Quat::from_rotation_z(5. * PI / 4.)),
                        _ => unreachable!(),
                    };
                    p.spawn((
                        ActProbVis { action_id },
                        PbrBundle {
                            mesh: quad_mesh.0.clone(),
                            material: arrow_mat.0.clone(),
                            transform,
                            ..default()
                        },
                    ));
                }
            });
        }
    }
    for (pursuer_e, pursuer) in pursuer_query.iter() {
        if pursuer.action_probs.is_empty() {
            continue;
        }
        for (vis, parent, mut xform) in vis_query.iter_mut() {
            if parent.get() == pursuer_e {
                let action_id = vis.action_id;
                let prob = pursuer.action_probs[action_id];
                let scale = 1. + prob * 16.;
                xform.scale = Vec3::new(scale, scale, 1.);
            }
        }
    }
}

const AGENT_SPEED: f32 = GRID_CELL_SIZE * 2.;

/// Holds the next action for an agent.
#[derive(Default, Component)]
pub struct NextAction {
    /// Which direction the agent will move in.
    pub dir: Vec2,
    /// Whether the agent should toggle nearby objects this frame.
    pub toggle_objs: bool,
}

/// Allows the player to set the Player's next action.
fn set_player_action(
    inpt: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut NextAction, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::KeyW) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::KeyS) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::KeyA) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::KeyD) {
        dir.x += 1.;
    }
    if let Ok(mut next_action) = player_query.get_single_mut() {
        next_action.dir = dir;
        next_action.toggle_objs = false;
        if inpt.just_pressed(KeyCode::KeyF) {
            next_action.toggle_objs = true;
        }
    }
}

/// Updates the Pursuer's next action with a neural policy.
fn set_pursuer_action_neural(
    net_query: Query<&NNWrapper<PolicyNet>>,
    mut pursuer_query: Query<
        (&mut NextAction, &mut PursuerAgent, &GlobalTransform),
        With<NeuralPolicy>,
    >,
    player_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
) {
    if let Ok((mut next_action, mut pursuer, pursuer_xform)) = pursuer_query.get_single_mut() {
        let p_net = net_query.single();
        if let Some(net) = &p_net.net {
            if pursuer.obs_timer.just_finished() {
                if let Some((grid, objs, objs_attn_mask)) = &pursuer.observations {
                    let logits = net
                        .forward(
                            &grid.unsqueeze(0).unwrap(),
                            objs.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                            objs_attn_mask
                                .as_ref()
                                .map(|t| t.unsqueeze(0).unwrap())
                                .as_ref(),
                        )
                        .unwrap()
                        .squeeze(0)
                        .unwrap();
                    let probs = (logits
                        .exp()
                        .unwrap()
                        .broadcast_div(&logits.exp().unwrap().sum_all().unwrap()))
                    .unwrap();
                    let probs = probs.to_vec1::<f32>().unwrap();
                    pursuer.action_probs = probs.clone();
                    let index = rand::distributions::WeightedIndex::new(probs).unwrap();
                    let mut rng = rand::thread_rng();
                    let action = index.sample(&mut rng);

                    let action_map = [
                        Vec2::ZERO,
                        Vec2::Y,
                        (Vec2::Y + Vec2::X).normalize(),
                        Vec2::X,
                        (-Vec2::Y + Vec2::X).normalize(),
                        -Vec2::Y,
                        (-Vec2::Y + -Vec2::X).normalize(),
                        -Vec2::X,
                        (Vec2::Y + -Vec2::X).normalize(),
                    ];
                    let mut dir = action_map[action];

                    // Beeline towards player if in sight
                    let (player_e, player_xform) = player_query.single();
                    if pursuer
                        .agent_state
                        .as_ref()
                        .unwrap()
                        .observing
                        .contains(&player_e)
                    {
                        let offset =
                            player_xform.translation().xy() - pursuer_xform.translation().xy();
                        dir = offset.normalize();
                    }

                    next_action.dir = dir;
                    next_action.toggle_objs = false;
                }
            }
        }
    }
}

/// Updates the Pursuer's next action with a pathfinding policy.
fn set_pursuer_action_pathfinding(
    net_query: Query<&NNWrapper<PolicyNet>>,
    mut pursuer_query: Query<
        (&mut NextAction, &mut PursuerAgent, &GlobalTransform),
        With<PathfindingPolicy>,
    >,
    filter_query: Query<&BayesFilter>,
    player_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
    level: Res<LevelLayout>,
) {
    if let Ok((mut next_action, mut pursuer, pursuer_xform)) = pursuer_query.get_single_mut() {
        let p_net = net_query.single();
        if let Some(net) = &p_net.net {
            if pursuer.obs_timer.just_finished() {
                // Identify most probable tile
                let tile_idx = filter_query
                    .single()
                    .probs
                    .flatten(0, 1)
                    .unwrap()
                    .argmax(0)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap() as usize;
                let x = tile_idx % level.size;
                let y = tile_idx / level.size;

                // Pathfind from current tile to target tile
                let mut queue = VecDeque::new();
                let mut parents = HashMap::new();
                let xlation = pursuer_xform.translation();
                let goal_pos = pos_to_grid(xlation.x, xlation.y, GRID_CELL_SIZE);
                queue.push_back((x, y));
                loop {
                    let curr_tile = queue.pop_front().unwrap();
                    if curr_tile == goal_pos {
                        break;
                    }
                    let neighbors = [
                        (curr_tile.0 as i32 + 1, curr_tile.1 as i32),
                        (curr_tile.0 as i32 - 1, curr_tile.1 as i32),
                        (curr_tile.0 as i32, curr_tile.1 as i32 + 1),
                        (curr_tile.0 as i32, curr_tile.1 as i32 - 1),
                    ];
                    for n in neighbors {
                        if n.0 > 0
                            && n.0 < level.size as i32 - 1
                            && n.1 > 0
                            && n.1 < level.size as i32 - 1
                            && !level.walls[n.1 as usize * level.size + n.0 as usize]
                            && !parents.contains_key(&n)
                        {
                            queue.push_back((n.0 as usize, n.1 as usize));
                            parents.insert(n, curr_tile);
                        }
                    }
                }
                let next_tile = parents
                    .get(&(goal_pos.0 as i32, goal_pos.1 as i32))
                    .unwrap();
                let mut dir = Vec2::new(next_tile.0 as f32, next_tile.1 as f32) * GRID_CELL_SIZE + 0.5
                    - pursuer_xform.translation().xy();
                dir = dir.normalize();

                // Beeline towards player if in sight
                let (player_e, player_xform) = player_query.single();
                if pursuer
                    .agent_state
                    .as_ref()
                    .unwrap()
                    .observing
                    .contains(&player_e)
                {
                    let offset = player_xform.translation().xy() - pursuer_xform.translation().xy();
                    dir = offset.normalize();
                }

                next_action.dir = dir;
                next_action.toggle_objs = false;
            }
        }
    }
}

/// If present, causes positions to be locked to a grid.
#[derive(Resource)]
pub struct UseGridPositions;

/// Moves agents around.
pub fn move_agents(
    mut agent_query: Query<(
        Entity,
        &mut Agent,
        &mut KinematicCharacterController,
        &NextAction,
        &Children,
        &GlobalTransform,
    )>,
    child_query: Query<(Entity, Option<&Name>, Option<&Children>)>,
    mut vis_query: Query<&mut Transform, With<AgentVisuals>>,
    mut anim_query: Query<&mut AnimationPlayer>,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
    use_grid: Option<Res<UseGridPositions>>,
) {
    for (agent_e, mut agent, mut controller, next_action, children, xform) in agent_query.iter_mut()
    {
        let dir = next_action.dir;
        let anim_e = get_entity(&agent_e, &["", "", "Root"], &child_query);
        if dir.length_squared() > 0.1 {
            let dir = dir.normalize();
            agent.dir = dir;
            if use_grid.is_some() {
                let pos = xform.translation().xy() + dir * GRID_CELL_SIZE;
                let (x, y) = pos_to_grid(pos.x, pos.y, GRID_CELL_SIZE);
                let new_pos = Vec2::new(
                    x as f32 * GRID_CELL_SIZE + 0.5,
                    y as f32 * GRID_CELL_SIZE + 0.5,
                );
                controller.translation = Some(new_pos - xform.translation().xy());
            } else {
                let xlation = dir * AGENT_SPEED * time.delta_seconds();
                controller.translation = Some(xlation);
            }
            if let Some(anim_e) = anim_e {
                if let Ok(mut anim) = anim_query.get_mut(anim_e) {
                    for child in children.iter() {
                        if let Ok(mut xform) = vis_query.get_mut(*child) {
                            xform.look_to(-dir.extend(0.), Vec3::Z);
                            {
                                anim.play_with_transition(
                                    asset_server.load("characters/cyborgFemaleA.glb#Animation1"),
                                    Duration::from_secs_f32(0.2),
                                )
                                .repeat();
                            }
                            break;
                        }
                    }
                }
            }
        } else if let Some(anim_e) = anim_e {
            if let Ok(mut anim) = anim_query.get_mut(anim_e) {
                {
                    anim.play_with_transition(
                        asset_server.load("characters/cyborgFemaleA.glb#Animation0"),
                        Duration::from_secs_f32(0.2),
                    )
                    .repeat();
                }
            }
        }
    }
}

/// Returns the entity at this path.
pub fn get_entity(
    parent: &Entity,
    path: &[&str],
    child_query: &Query<(Entity, Option<&Name>, Option<&Children>)>,
) -> Option<Entity> {
    if path.is_empty() {
        return Some(*parent);
    }
    if let Ok((_, _, Some(children))) = child_query.get(*parent) {
        for child in children {
            let (_, name, _) = child_query.get(*child).unwrap();
            if (name.is_none() && path[0].is_empty()) || (name.unwrap().as_str() == path[0]) {
                let e = get_entity(child, &path[1..], child_query);
                if e.is_some() {
                    return e;
                }
            }
        }
    }
    None
}
