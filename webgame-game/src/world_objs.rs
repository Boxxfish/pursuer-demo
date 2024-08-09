use std::{f32::consts::PI, time::Duration};

use crate::{
    agents::{get_entity, PlayerAgent},
    gridworld::GRID_CELL_SIZE,
    observer::Wall,
};
use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use rand::Rng;

/// Plugin for world objects (e.g. doors, noise sources).
pub struct WorldObjPlugin;

impl Plugin for WorldObjPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                handle_key_touch,
                update_noise_src,
                // visualize_visual_marker,
            ),
        );
    }
}

pub struct WorldObjPlayPlugin;

impl Plugin for WorldObjPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_particle_mesh).add_systems(
            Update,
            (
                key_idle_anim,
                update_pickup_effect,
                update_particles,
                visualize_noise_src,
            ),
        );
    }
}

/// A particle for effects.
/// This is affected by gravity, and despawns after the timer goes off.
#[derive(Component)]
struct EffectParticle {
    pub timer: Timer,
    pub vel: Vec3,
    pub acc: Vec3,
    pub start_size: f32,
}

fn update_particles(
    mut particle_query: Query<(Entity, &mut EffectParticle, &mut Transform)>,
    time: Res<Time>,
    mut commands: Commands,
) {
    for (e, mut particle, mut xform) in particle_query.iter_mut() {
        particle.vel = particle.vel + particle.acc * time.delta_seconds();
        xform.translation += particle.vel * time.delta_seconds();
        xform.scale = Vec3::splat(particle.start_size) * particle.timer.fraction_remaining();

        particle.timer.tick(time.delta());
        if particle.timer.finished() {
            commands.entity(e).despawn();
        }
    }
}

#[derive(Resource)]
pub struct QuadMesh(pub Handle<Mesh>);

#[derive(Resource)]
struct ShockwaveMaterial(Handle<StandardMaterial>);

fn init_particle_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let handle = meshes.add(Rectangle::new(1., 1.));
    commands.insert_resource(QuadMesh(handle));
    let shockwave_material = materials.add(StandardMaterial {
        base_color_texture: Some(asset_server.load("shockwave.png")),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.insert_resource(ShockwaveMaterial(shockwave_material));
}

/// Causes the entity to spin around, emit particles, then disappear.
#[derive(Component)]
pub struct PickupEffect {
    pub timer: Timer,
    pub spawned_particles: bool,
    pub color: Color,
}

impl PickupEffect {
    pub fn from_color(color: Color) -> Self {
        Self { color, ..default() }
    }
}

/// How long the effect lasts.
const PICKUP_EFFECT_TIME: f32 = 0.6;

impl Default for PickupEffect {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(PICKUP_EFFECT_TIME, TimerMode::Once),
            spawned_particles: false,
            color: Color::YELLOW,
        }
    }
}

fn update_pickup_effect(
    mut effect_query: Query<(Entity, &mut PickupEffect, &mut Transform, &GlobalTransform)>,
    time: Res<Time>,
    mut commands: Commands,
    particle_mesh: Res<QuadMesh>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (e, mut effect, mut xform, &global_xform) in effect_query.iter_mut() {
        if !effect.spawned_particles {
            let pos = global_xform.translation() + Vec3::Z * 10.;
            let mut rng = rand::thread_rng();
            let material = materials.add(StandardMaterial {
                base_color: effect.color,
                unlit: true,
                ..default()
            });
            for _ in 0..20 {
                let angle = rng.gen_range(0.0..(PI * 2.));
                let dir = Vec3::new(angle.sin(), angle.cos(), 0.);
                commands.spawn((
                    EffectParticle {
                        start_size: 4.,
                        timer: Timer::from_seconds(rng.gen_range(0.2..0.5), TimerMode::Once),
                        vel: dir * rng.gen_range(200.0..500.0),
                        acc: Vec3::ZERO,
                    },
                    MaterialMeshBundle {
                        mesh: particle_mesh.0.clone(),
                        material: material.clone(),
                        transform: Transform::from_translation(pos).with_scale(Vec3::splat(4.)),
                        ..default()
                    },
                ));
            }
            effect.spawned_particles = true;
        }

        effect.timer.tick(time.delta());
        if effect.timer.finished() {
            commands.entity(e).despawn_recursive();
        }

        let elapsed = effect.timer.elapsed_secs();
        xform.translation.z = 0.;
        xform.scale = Vec3::new(
            1. - elapsed / PICKUP_EFFECT_TIME,
            1. - elapsed / PICKUP_EFFECT_TIME,
            1.,
        );
        xform.rotation = Quat::from_rotation_y(PI * 4. * elapsed);
    }
}

/// A key for unlocking doors.
#[derive(Component)]
pub struct Key;

/// Adds an idle animation to keys.
pub fn key_idle_anim(mut key_query: Query<&mut Transform, With<Key>>, time: Res<Time>) {
    for mut key_xform in key_query.iter_mut() {
        key_xform.translation.z = (time.elapsed_seconds_wrapped() * 2.).cos() * 10. + 5.;
    }
}

/// How close the agent needs to be before it can pick up a key.
const PICKUP_DIST: f32 = GRID_CELL_SIZE / 2.;

/// Opens the door and destroys the key if the player touches it.
fn handle_key_touch(
    player_query: Query<&GlobalTransform, With<PlayerAgent>>,
    key_query: Query<(Entity, &GlobalTransform), With<Key>>,
    mut door_query: Query<(Entity, &mut Door)>,
    door_vis_query: Query<Entity, With<DoorVisual>>,
    mut commands: Commands,
    mut anim_query: Query<&mut AnimationPlayer>,
    child_query: Query<(Entity, Option<&Name>, Option<&Children>)>,
    asset_server: Res<AssetServer>,
) {
    for player_xform in player_query.iter() {
        let player_pos = player_xform.translation().xy();
        for (key_e, key_xform) in key_query.iter() {
            let obj_pos = key_xform.translation().xy();
            let dist_sq = (obj_pos - player_pos).length_squared();
            if dist_sq < PICKUP_DIST.powi(2) {
                commands
                    .entity(key_e)
                    .insert(PickupEffect::default())
                    .remove::<Key>();
                for (door_e, mut door) in door_query.iter_mut() {
                    door.open = !door.open;
                    commands.entity(door_e).remove::<(Wall, Collider)>();
                }
                for vis_e in door_vis_query.iter() {
                    let anim_e = get_entity(&vis_e, &["", "doorway(Clone)"], &child_query);
                    if let Some(anim_e) = anim_e {
                        let mut anim = anim_query.get_mut(anim_e).unwrap();
                        anim.play_with_transition(
                            asset_server.load("furniture/doorway.glb#Animation0"),
                            Duration::from_secs_f32(0.0),
                        );
                    }
                }
            }
        }
    }
}
/// A door that can be opened and closed.
#[derive(Component, Default)]
pub struct Door {
    pub open: bool,
}

/// A visual indicating the door iteself.
#[derive(Component)]
pub struct DoorVisual;

/// A source of noise that alerts observers within a radius.
#[derive(Component)]
pub struct NoiseSource {
    /// How far away to broadcast the noise.
    pub noise_radius: f32,
    /// How close an agent has to be to activate the noise source.
    pub active_radius: f32,
    pub activated_by_player: bool,
}

/// Broadcasts that an agent touched the noise source.
fn update_noise_src(
    agent_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
    mut noise_query: Query<(&GlobalTransform, &mut NoiseSource)>,
) {
    for (obj_xform, mut noise) in noise_query.iter_mut() {
        noise.activated_by_player = false;
        for (agent_e, agent_xform) in agent_query.iter() {
            let agent_pos = agent_xform.translation().xy();
            let obj_pos = obj_xform.translation().xy();
            let dist_sq = (obj_pos - agent_pos).length_squared();
            if dist_sq <= noise.active_radius.powi(2) {
                noise.activated_by_player = true;
            }
        }
    }
}

/// A shockwave visual effect.
#[derive(Component)]
struct ShockwaveEffect {
    pub timer: Timer,
    pub size: f32,
}

/// Visualizes a noise source.
fn visualize_noise_src(
    noise_query: Query<(&GlobalTransform, &NoiseSource)>,
    mut effect_query: Query<(
        Entity,
        &mut ShockwaveEffect,
        &mut Transform,
        &mut Handle<StandardMaterial>,
    )>,
    time: Res<Time>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    quad_mesh: Res<QuadMesh>,
    shockwave_material: Res<ShockwaveMaterial>,
) {
    for (obj_xform, noise) in noise_query.iter() {
        let obj_pos = obj_xform.translation().xy();
        for (effect_e, mut effect, mut xform, mat) in effect_query.iter_mut() {
            let pct = effect.timer.fraction();
            if let Some(mat) = materials.get_mut(mat.id()) {
                mat.base_color.set_a(1. - pct);
            }
            let scale = pct * effect.size * 2.;
            xform.scale = Vec3::new(scale, scale, 1.);
            effect.timer.tick(time.delta());
            if effect.timer.just_finished() {
                commands.entity(effect_e).despawn_recursive();
            }
        }
        if noise.activated_by_player {
            if effect_query.is_empty() {
                commands.spawn((
                    PbrBundle {
                        mesh: quad_mesh.0.clone(),
                        material: shockwave_material.0.clone(),
                        transform: Transform::from_translation(
                            obj_xform.translation().xy().extend(10.),
                        ),
                        ..default()
                    },
                    ShockwaveEffect {
                        timer: Timer::from_seconds(0.4, TimerMode::Once),
                        size: noise.noise_radius,
                    },
                ));
            }
        }
    }
}

/// A visual marker.
/// Observers record the last seen positions of these items.
#[derive(Component)]
pub struct VisualMarker;

/// Visualizes a visual marker.
#[allow(dead_code)]
fn visualize_visual_marker(
    mut gizmos: Gizmos,
    visual_query: Query<&GlobalTransform, With<VisualMarker>>,
) {
    for obj_xform in visual_query.iter() {
        let obj_pos = obj_xform.translation().xy();
        gizmos.rect(
            obj_pos.extend(GRID_CELL_SIZE),
            Quat::IDENTITY,
            Vec2::ONE * GRID_CELL_SIZE * 0.5,
            Color::RED,
        );
    }
}
