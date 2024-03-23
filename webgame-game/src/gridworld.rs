use bevy::{prelude::*, sprite::Anchor};
use bevy_rapier2d::{
    control::KinematicCharacterController, dynamics::RigidBody, geometry::Collider,
};
use rand::Rng;

use crate::{
    observer::{DebugObserver, Observable, Observer, Wall},
    world_objs::Door,
};

/// Plugin for basic game features, such as moving around and not going through walls.
pub struct GridworldPlugin;

impl Plugin for GridworldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LevelLayout::random(DEFAULT_LEVEL_SIZE))
            .add_systems(Startup, setup_entities)
            .add_systems(Update, move_agents);
    }
}

/// Adds playable functionality for `GridworldPlugin`.
pub struct GridworldPlayPlugin;

impl Plugin for GridworldPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_entities_playable)
            .add_systems(
                Update,
                (
                    visualize_agent::<PursuerAgent>(Color::RED),
                    visualize_agent::<PlayerAgent>(Color::GREEN),
                    set_player_action,
                ),
            );
    }
}

/// The width and height of the level by default.
pub const DEFAULT_LEVEL_SIZE: usize = 8;
/// The probability of a door spawning in an empty cell.
pub const DOOR_PROB: f64 = 0.05;

/// Stores the layout of the level.
#[derive(Resource)]
pub struct LevelLayout {
    /// Stores `true` if a wall exists, `false` for empty spaces. The first element is the top right corner.
    pub walls: Vec<bool>,
    pub size: usize,
}

impl LevelLayout {
    /// Generates a randomized level.
    pub fn random(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            walls: (0..(size * size)).map(|_| rng.gen_bool(0.2)).collect(),
            size,
        }
    }
}

/// State used by all agents.
#[derive(Component)]
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
pub struct PursuerAgent;

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// Sets up all entities in the game.
fn setup_entities(mut commands: Commands, level: Res<LevelLayout>) {
    commands.spawn((
        PursuerAgent,
        Agent::default(),
        NextAction::default(),
        Collider::ball(GRID_CELL_SIZE * 0.25),
        RigidBody::KinematicPositionBased,
        KinematicCharacterController::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(20., 0., 0.))),
        Observer::default(),
        Observable,
    ));
    commands.spawn((
        PlayerAgent,
        Agent::default(),
        NextAction::default(),
        Collider::ball(GRID_CELL_SIZE * 0.25),
        RigidBody::KinematicPositionBased,
        KinematicCharacterController::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(40., 30., 0.))),
        Observer::default(),
        Observable,
        DebugObserver,
    ));

    // Set up walls and doors
    let mut rng = rand::thread_rng();
    for y in 0..level.size {
        for x in 0..level.size {
            if level.walls[y * level.size + x] {
                commands.spawn((
                    Wall,
                    Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                    TransformBundle::from_transform(Transform::from_translation(
                        Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE,
                    )),
                ));
            } else if rng.gen_bool(DOOR_PROB) {
                commands.spawn((
                    Door::default(),
                    Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                    TransformBundle::from_transform(Transform::from_translation(
                        Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE,
                    )),
                ));
            }
        }
    }

    // Set up the sides of the game world
    let half_sizes = [GRID_CELL_SIZE / 2., GRID_CELL_SIZE * level.size as f32 / 2.];
    let wall_positions = [-GRID_CELL_SIZE, GRID_CELL_SIZE * level.size as f32];
    let wall_pos_offset = GRID_CELL_SIZE * (level.size - 1) as f32 / 2.;
    for i in 0..4 {
        let positions = [wall_positions[i % 2], wall_pos_offset];
        commands.spawn((
            Wall,
            Collider::cuboid(half_sizes[i / 2], half_sizes[1 - i / 2]),
            TransformBundle::from_transform(Transform::from_translation(Vec3::new(
                positions[i / 2],
                positions[1 - i / 2],
                0.,
            ))),
        ));
    }
}

pub const GRID_CELL_SIZE: f32 = 25.;

/// Sets up entities for playable mode.
fn setup_entities_playable(mut commands: Commands, level: Res<LevelLayout>) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_translation(Vec3::new(
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            0.,
        )),
        ..default()
    });

    for y in 0..level.size {
        for x in 0..level.size {
            if level.walls[y * level.size + x] {
                commands.spawn(SpriteBundle {
                    sprite: Sprite {
                        color: Color::BLACK,
                        custom_size: Some(Vec2::ONE * GRID_CELL_SIZE),
                        ..default()
                    },
                    transform: Transform::from_translation(
                        Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE,
                    ),
                    ..default()
                });
            }
        }
    }

    // Set up the sides of the game world
    let wall_positions = [
        -GRID_CELL_SIZE * 0.5,
        GRID_CELL_SIZE * (level.size as f32 - 0.5),
    ];
    let wall_pos_offset = GRID_CELL_SIZE * (level.size as f32 / 2. - 0.5);
    let anchors = [
        Anchor::CenterRight,
        Anchor::CenterLeft,
        Anchor::TopCenter,
        Anchor::BottomCenter,
    ];
    for i in 0..4 {
        let positions = [wall_positions[i % 2], wall_pos_offset];
        commands.spawn((SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::ONE * GRID_CELL_SIZE * level.size as f32 * 2.),
                anchor: anchors[i],
                ..default()
            },
            transform: Transform::from_translation(Vec3::new(
                positions[i / 2],
                positions[1 - i / 2],
                0.,
            )),
            ..default()
        },));
    }
}

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(Commands<'_, '_>, Query<'_, '_, Entity, Added<T>>) {
    move |mut commands: Commands, agent_query: Query<Entity, Added<T>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Sprite {
                    color,
                    custom_size: Some(Vec2::ONE * GRID_CELL_SIZE * 0.25),
                    ..default()
                },
                Handle::<Image>::default(),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
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

/// Allows the player to set the Players next action.
fn set_player_action(
    inpt: Res<Input<KeyCode>>,
    mut player_query: Query<&mut NextAction, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::W) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::S) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::A) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::D) {
        dir.x += 1.;
    }
    let mut next_action = player_query.single_mut();
    next_action.dir = dir;
    next_action.toggle_objs = false;
    if inpt.just_pressed(KeyCode::F) {
        next_action.toggle_objs = true;
    }
}

/// Moves agents around.
fn move_agents(
    mut agent_query: Query<(&mut Agent, &mut KinematicCharacterController, &NextAction)>,
    time: Res<Time>,
) {
    for (mut agent, mut controller, next_action) in agent_query.iter_mut() {
        let dir = next_action.dir;
        if dir.length_squared() > 0.1 {
            let dir = dir.normalize();
            agent.dir = dir;
            controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
        }
    }
}
