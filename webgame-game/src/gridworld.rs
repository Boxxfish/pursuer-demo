use std::f32::consts::PI;

use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    utils::BoxedFuture,
};
use bevy_rapier2d::{
    control::KinematicCharacterController,
    dynamics::{Damping, LockedAxes, RigidBody},
    geometry::Collider,
};
use rand::{seq::IteratorRandom, Rng};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    agents::{Agent, AgentVisuals, NeuralPolicy, NextAction, PathfindingPolicy, PlayerAgent, PursuerAgent},
    configs::IsPlayable,
    filter::BayesFilter,
    observer::{DebugObserver, Observable, Observer, Wall},
    screens::GameScreen,
    world_objs::{Door, DoorVisual, Key, NoiseSource, PickupEffect, VisualMarker},
};

/// Plugin for basic game features, such as moving around and not going through walls.
pub struct GridworldPlugin;

impl Plugin for GridworldPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ResetEvent>().add_systems(
            Update,
            (
                reset_game,
                setup_entities.run_if(resource_added::<LevelLayout>),
            ),
        );
    }
}

/// Adds playable functionality for `GridworldPlugin`.
pub struct GridworldPlayPlugin;

impl Plugin for GridworldPlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<LoadedLevelData>()
            .init_asset_loader::<LoadedLevelDataLoader>()
            .add_event::<GameEndEvent>()
            .add_systems(
                Update,
                (
                    load_level,
                    (player_reached_door, pursuer_sees_player).run_if(resource_exists::<ShouldRun>),
                ),
            );
    }
}

pub const GRID_CELL_SIZE: f32 = 25.;

/// Data for objects in levels.
#[derive(Deserialize, Clone)]
pub struct LoadedObjData {
    pub name: String,
    pub pos: (usize, usize),
    #[serde(default)]
    pub dir: Option<String>,
    #[serde(default)]
    pub movable: bool,
}

/// Data for loaded levels.
#[derive(Deserialize, Asset, TypePath)]
pub struct LoadedLevelData {
    pub size: usize,
    pub walls: Vec<u8>,
    pub objects: Vec<LoadedObjData>,
    pub key_pos: (usize, usize),
    pub door_pos: (usize, usize),
    pub player_start: (usize, usize),
    pub pursuer_start: (usize, usize),
}

/// Indicates that a level should be loaded.
#[derive(Resource)]
pub enum LevelLoader {
    Path(String),
    Asset(Handle<LoadedLevelData>),
}

#[derive(Default)]
struct LoadedLevelDataLoader;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LoadedLevelDataLoaderError {
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse JSON: {0}")]
    JSONSpannedError(#[from] serde_json::error::Error),
}

impl AssetLoader for LoadedLevelDataLoader {
    type Asset = LoadedLevelData;
    type Settings = ();
    type Error = LoadedLevelDataLoaderError;
    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut buf = String::new();
            reader.read_to_string(&mut buf).await?;
            let data: LoadedLevelData = serde_json::from_str(&buf)?;
            Ok(data)
        })
    }

    fn extensions(&self) -> &[&str] {
        &[".json"]
    }
}

/// Loads levels.
fn load_level(
    level: Option<Res<LevelLoader>>,
    level_data: Res<Assets<LoadedLevelData>>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
) {
    if let Some(level) = level {
        match level.as_ref() {
            LevelLoader::Path(path) => {
                commands.insert_resource(LevelLoader::Asset(asset_server.load(path)));
            }
            LevelLoader::Asset(handle) => {
                if let Some(level) = level_data.get(handle.clone()) {
                    let mut walls = Vec::new();
                    for y in 0..level.size {
                        for x in 0..level.size {
                            walls.push(level.walls[(level.size - y - 1) * level.size + x] != 0);
                        }
                    }
                    commands.insert_resource(LevelLayout {
                        walls,
                        size: level.size,
                        objects: level.objects.clone(),
                        key_pos: Some(level.key_pos),
                        door_pos: Some(level.door_pos),
                        player_start: Some(level.player_start),
                        pursuer_start: Some(level.pursuer_start),
                    });
                    commands.remove_resource::<LevelLoader>();
                }
            }
        }
    }
}

/// Indicates that the game should begin running.
#[derive(Resource)]
pub struct ShouldRun;

/// Stores the layout of the level.
#[derive(Resource, Clone)]
pub struct LevelLayout {
    /// Stores `true` if a wall exists, `false` for empty spaces. The first element is the top right corner.
    pub walls: Vec<bool>,
    pub size: usize,
    pub objects: Vec<LoadedObjData>,
    pub key_pos: Option<(usize, usize)>,
    pub door_pos: Option<(usize, usize)>,
    pub player_start: Option<(usize, usize)>,
    pub pursuer_start: Option<(usize, usize)>,
}

impl LevelLayout {
    /// Generates a randomized level.
    pub fn random(size: usize, wall_prob: f64, max_items: usize) -> Self {
        let mut rng = rand::thread_rng();
        let orig = Self {
            walls: (0..(size * size))
                .map(|_| rng.gen_bool(wall_prob))
                .collect(),
            size,
            objects: Vec::new(),
            key_pos: None,
            door_pos: None,
            player_start: None,
            pursuer_start: None,
        };
        let mut objects = Vec::new();
        if max_items > 0 {
            for _ in 0..rng.gen_range(0..max_items) {
                let tile_idx = orig.get_empty();
                let y = tile_idx / size;
                let x = tile_idx % size;
                objects.push(LoadedObjData {
                    name: "".into(),
                    pos: (x, y),
                    dir: Some("left".into()),
                    movable: true,
                });
            }
        }
        Self {
            walls: orig.walls,
            size,
            objects,
            key_pos: None,
            door_pos: None,
            player_start: None,
            pursuer_start: None,
        }
    }

    /// Returns a random empty tile index.
    pub fn get_empty(&self) -> usize {
        let mut rng = rand::thread_rng();
        let tile_idx = self
            .walls
            .iter()
            .enumerate()
            .filter(|(_, x)| !**x)
            .map(|(x, _)| x)
            .choose(&mut rng)
            .unwrap();
        tile_idx
    }
}

/// Sets up all entities in the game.
fn setup_entities(
    mut commands: Commands,
    level: Res<LevelLayout>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    is_playable: Option<Res<IsPlayable>>,
    mut filter_query: Query<&mut BayesFilter>,
) {
    for mut filter in filter_query.iter_mut() {
        filter.reset();
    }

    commands
        .spawn((
            GameScreen,
            VisibilityBundle::default(),
            TransformBundle::default(),
        ))
        .with_children(|p| {
            // Add light
            p.spawn(DirectionalLightBundle {
                directional_light: DirectionalLight {
                    illuminance: 6000.,
                    ..default()
                },
                transform: Transform::from_rotation(Quat::from_rotation_x(PI / 4.)),
                ..default()
            });

            let pursuer_tile_idx = match &level.pursuer_start {
                Some((x, y)) => (level.size - y - 1) * level.size + x,
                None => level.get_empty(),
            };
            p.spawn((
                PursuerAgent::default(),
                Agent::default(),
                NextAction::default(),
                Collider::ball(GRID_CELL_SIZE * 0.25),
                RigidBody::KinematicPositionBased,
                KinematicCharacterController::default(),
                TransformBundle::from_transform(Transform::from_translation(
                    Vec3::new(
                        (pursuer_tile_idx % level.size) as f32,
                        (pursuer_tile_idx / level.size) as f32,
                        0.,
                    ) * GRID_CELL_SIZE,
                )),
                NeuralPolicy,
                Observer::default(),
                Observable,
                DebugObserver,
            ))
            .with_children(|p| {
                if is_playable.is_some() {
                    p.spawn((
                        AgentVisuals,
                        SceneBundle {
                            scene: asset_server.load("characters/cyborgFemaleA.glb#Scene0"),
                            transform: Transform::default()
                                .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                                .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.4),
                            ..default()
                        },
                    ));
                }
            });
            let player_tile_idx = match &level.player_start {
                Some((x, y)) => (level.size - y - 1) * level.size + x,
                None => level.get_empty(),
            };
            p.spawn((
                PlayerAgent,
                Agent::default(),
                NextAction::default(),
                Collider::ball(GRID_CELL_SIZE * 0.25),
                RigidBody::KinematicPositionBased,
                KinematicCharacterController::default(),
                TransformBundle::from_transform(Transform::from_translation(
                    Vec3::new(
                        (player_tile_idx % level.size) as f32,
                        (player_tile_idx / level.size) as f32,
                        0.,
                    ) * GRID_CELL_SIZE,
                )),
                Observer::default(),
                Observable,
            ))
            .with_children(|p| {
                if is_playable.is_some() {
                    p.spawn((
                        AgentVisuals,
                        SceneBundle {
                            scene: asset_server.load("characters/skaterMaleA.glb#Scene0"),
                            transform: Transform::default()
                                .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                                .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.4),
                            ..default()
                        },
                    ));
                }
            });

            // Add floor
            p.spawn(SceneBundle {
                scene: asset_server.load("furniture/floorFull.glb#Scene0"),
                transform: Transform::default()
                    .with_translation(Vec3::new(-1., -1., 0.) * GRID_CELL_SIZE / 2.)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                    .with_scale(
                        Vec3::new(level.size as f32, 1., level.size as f32) * GRID_CELL_SIZE,
                    ),
                ..default()
            });

            // Set up walls and doors
            let wall_mesh = meshes.add(Cuboid::new(GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1));
            let wall_mat = materials.add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..default()
            });
            for y in 0..level.size {
                for x in 0..level.size {
                    if level.walls[y * level.size + x] {
                        p.spawn((
                            Wall,
                            Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                            TransformBundle::from_transform(Transform::from_translation(
                                Vec3::new(x as f32, y as f32, 0.) * GRID_CELL_SIZE,
                            )),
                            VisibilityBundle::default(),
                        ))
                        .with_children(|p| {
                            if is_playable.is_some() {
                                let offsets = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y];
                                let base_xform = Transform::default()
                                    .with_translation(-Vec3::X * GRID_CELL_SIZE / 2.)
                                    .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                                    .with_scale(Vec3::ONE * GRID_CELL_SIZE);
                                for (i, offset) in offsets.iter().enumerate() {
                                    let should_spawn = match i {
                                        3 => (y > 0) && !level.walls[(y - 1) * level.size + x],
                                        2 => {
                                            (y < level.size - 1)
                                                && !level.walls[(y + 1) * level.size + x]
                                        }
                                        1 => (x > 0) && !level.walls[y * level.size + (x - 1)],
                                        0 => {
                                            (x < level.size - 1)
                                                && !level.walls[y * level.size + (x + 1)]
                                        }
                                        _ => unreachable!(),
                                    };
                                    if should_spawn {
                                        let rot = if i >= 2 {
                                            Quat::IDENTITY
                                        } else {
                                            Quat::from_rotation_z(std::f32::consts::PI / 2.)
                                        };
                                        p.spawn(SceneBundle {
                                            scene: asset_server.load("furniture/wall.glb#Scene0"),
                                            transform: Transform::default()
                                                .with_rotation(rot)
                                                .with_translation(
                                                    *offset * (GRID_CELL_SIZE / 2. + 0.1),
                                                )
                                                * base_xform,
                                            ..default()
                                        });
                                    }
                                }
                            }
                            p.spawn(PbrBundle {
                                mesh: wall_mesh.clone(),
                                material: wall_mat.clone(),
                                transform: Transform::from_translation(
                                    Vec3::Z * GRID_CELL_SIZE / 2.,
                                ),
                                ..default()
                            });
                        });
                    }
                }
            }

            // Set up the sides of the game world
            let half_sizes = [GRID_CELL_SIZE / 2., GRID_CELL_SIZE * level.size as f32 / 2.];
            let wall_positions = [-GRID_CELL_SIZE, GRID_CELL_SIZE * level.size as f32];
            let wall_pos_offset = GRID_CELL_SIZE * (level.size - 1) as f32 / 2.;
            for i in 0..4 {
                let positions = [wall_positions[i % 2], wall_pos_offset];
                p.spawn((
                    Wall,
                    Collider::cuboid(half_sizes[i / 2], half_sizes[1 - i / 2]),
                    TransformBundle::from_transform(Transform::from_translation(Vec3::new(
                        positions[i / 2],
                        positions[1 - i / 2],
                        0.,
                    ))),
                    VisibilityBundle::default(),
                ))
                .with_children(|p| {
                    if is_playable.is_some() {
                        // let offsets = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y];
                        // let base_xform = Transform::default()
                        //     .with_translation(-Vec3::X * GRID_CELL_SIZE * level.size as f32 / 2.)
                        //     .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                        //     .with_scale(Vec3::new(level.size as f32, 1., 1.) * GRID_CELL_SIZE);
                        // let rot = if i >= 2 {
                        //     Quat::IDENTITY
                        // } else {
                        //     Quat::from_rotation_z(std::f32::consts::PI / 2.)
                        // };
                        // p.spawn(SceneBundle {
                        //     scene: asset_server.load("furniture/wall.glb#Scene0"),
                        //     transform: Transform::default()
                        //         .with_rotation(rot)
                        //         .with_translation(offsets[i] * GRID_CELL_SIZE / 2.)
                        //         * base_xform,
                        //     ..default()
                        // });
                    } else {
                        p.spawn(PbrBundle {
                            mesh: meshes.add(Cuboid::new(
                                half_sizes[i / 2] * 2.,
                                half_sizes[1 - i / 2] * 2.,
                                0.1,
                            )),
                            material: wall_mat.clone(),
                            ..default()
                        });
                    }
                });
            }

            // Add objects, which may include noise sources and visual markers
            let obj_mat = materials.add(StandardMaterial {
                base_color: Color::BLUE,
                unlit: true,
                ..default()
            });
            for obj in &level.objects {
                let (x, y) = obj.pos;
                let pos = Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE;
                let collider_size = GRID_CELL_SIZE * 0.8;
                let mut child_builder = p.spawn((
                    Collider::cuboid(collider_size / 2., collider_size / 2.),
                    TransformBundle::from_transform(Transform::from_translation(pos)),
                    VisibilityBundle::default(),
                ));
                if obj.movable {
                    child_builder.insert((
                        RigidBody::Dynamic,
                        Damping {
                            linear_damping: 10.,
                            ..default()
                        },
                        LockedAxes::ROTATION_LOCKED,
                        NoiseSource {
                            noise_radius: GRID_CELL_SIZE * 4.,
                            active_radius: GRID_CELL_SIZE * 0.8,
                            activated_by_player: false,
                        },
                        VisualMarker,
                        Observable,
                    ));
                }
                child_builder.with_children(|p| {
                    if is_playable.is_some() {
                        let base_xform = Transform::default()
                            .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                            .with_scale(Vec3::ONE * GRID_CELL_SIZE * 2.);
                        let rot = match obj.dir.clone().unwrap_or("left".into()).as_str() {
                            "left" => Quat::from_rotation_z(std::f32::consts::PI * 3. / 2.),
                            "up" => Quat::from_rotation_z(std::f32::consts::PI),
                            "right" => Quat::from_rotation_z(std::f32::consts::PI / 2.),
                            "down" => Quat::IDENTITY,
                            _ => unimplemented!(),
                        };
                        p.spawn(SceneBundle {
                            scene: asset_server.load(format!("furniture/{}.glb#Scene0", obj.name)),
                            transform: Transform::default().with_rotation(rot) * base_xform,
                            ..default()
                        });
                    } else {
                        p.spawn(PbrBundle {
                            mesh: meshes.add(Cuboid::new(
                                collider_size,
                                collider_size,
                                0.1,
                            )),
                            material: obj_mat.clone(),
                            ..default()
                        });
                    }
                });
            }

            // Add keys and doors
            if let Some((x, y)) = level.key_pos {
                let pos = Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE;
                let mut child_builder = p.spawn((
                    Key,
                    TransformBundle::from_transform(
                        Transform::from_translation(pos)
                            .with_rotation(Quat::from_rotation_x(PI / 4.)),
                    ),
                    VisibilityBundle::default(),
                ));
                child_builder.with_children(|p| {
                    p.spawn(SceneBundle {
                        scene: asset_server.load("key.glb#Scene0"),
                        transform: Transform::default().with_scale(Vec3::ONE * GRID_CELL_SIZE),
                        ..default()
                    });
                });
            }
            if let Some((x, y)) = level.door_pos {
                let pos = Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE;
                let collider_size = GRID_CELL_SIZE;
                let angle = if y == level.size - 1 {
                    0.
                } else if x == level.size - 1 {
                    PI / 2.
                } else if y == 0 {
                    PI
                } else {
                    PI * 3. / 2.
                };
                let mut child_builder = p.spawn((
                    Door::default(),
                    Wall,
                    Collider::cuboid(collider_size / 2., collider_size / 2.),
                    TransformBundle::from_transform(
                        Transform::from_translation(pos).with_rotation(
                            Quat::from_rotation_x(PI / 2.) * Quat::from_rotation_y(angle),
                        ),
                    ),
                    VisibilityBundle::default(),
                ));
                child_builder.with_children(|p| {
                    p.spawn(SceneBundle {
                        scene: asset_server.load("furniture/wallDoorway.glb#Scene0"),
                        transform: Transform::default().with_scale(Vec3::ONE * GRID_CELL_SIZE),
                        ..default()
                    });
                    p.spawn((
                        DoorVisual,
                        SceneBundle {
                            scene: asset_server.load("furniture/doorway.glb#Scene0"),
                            transform: Transform::default().with_scale(Vec3::ONE * GRID_CELL_SIZE),
                            ..default()
                        },
                    ));
                });
            }
        });

    // Indicate we should start the game
    commands.insert_resource(ShouldRun);
}

/// Sent when the player wins or loses.
#[derive(Event)]
pub struct GameEndEvent {
    pub player_won: bool,
}

/// If the player has reached the door, win the game.
fn player_reached_door(
    mut ev_game_end: EventWriter<GameEndEvent>,
    level: Option<Res<LevelLayout>>,
    player_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
    mut commands: Commands,
) {
    if let Some(level) = level {
        if let Some(door_pos) = level.door_pos {
            if let Ok((player_e, player_xform)) = player_query.get_single() {
                let player_pos = player_xform.translation().xy() / GRID_CELL_SIZE;
                let player_pos = (
                    player_pos.x.round() as usize,
                    level.size - player_pos.y.round() as usize - 1,
                );
                if door_pos == player_pos {
                    commands.remove_resource::<ShouldRun>();
                    commands
                        .entity(player_e)
                        .insert(PickupEffect::from_color(Color::GREEN))
                        .remove::<PlayerAgent>();
                    ev_game_end.send(GameEndEvent { player_won: true });
                }
            }
        }
    }
}

/// If the pursuer can see the player and they are within 2 cells of each other, end the game.
fn pursuer_sees_player(
    mut ev_game_end: EventWriter<GameEndEvent>,
    player_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
    pursuer_query: Query<(&Observer, &GlobalTransform), With<PursuerAgent>>,
    mut commands: Commands,
) {
    if let Ok((player_e, player_xform)) = player_query.get_single() {
        if let Ok((pursuer_obs, pursuer_xform)) = pursuer_query.get_single() {
            let player_pos = player_xform.translation().xy();
            let pursuer_pos = pursuer_xform.translation().xy();
            if pursuer_obs.observing.contains(&player_e)
                && (player_pos - pursuer_pos).length_squared() < (2. * GRID_CELL_SIZE).powi(2)
            {
                commands.remove_resource::<ShouldRun>();
                commands
                    .entity(player_e)
                    .insert(PickupEffect::from_color(Color::RED))
                    .remove::<PlayerAgent>();
                ev_game_end.send(GameEndEvent { player_won: false });
            }
        }
    }
}

/// Resets the game.
#[derive(Event)]
pub struct ResetEvent {
    pub level: LevelLayout,
}

/// Resets the game when sent.
/// This is meant to be called during training.
fn reset_game(
    mut ev_reset: EventReader<ResetEvent>,
    screen_query: Query<Entity, With<GameScreen>>,
    mut commands: Commands,
) {
    for ev in ev_reset.read() {
        for e in screen_query.iter() {
            commands.entity(e).despawn_recursive();
        }
        commands.remove_resource::<LevelLayout>();
        commands.remove_resource::<ShouldRun>();
        commands.insert_resource(ev.level.clone());
    }
}
