use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::{mesh::PrimitiveTopology, render_asset::RenderAssetUsages},
};
use bevy_rapier2d::{math::Real, prelude::*};
use ordered_float::OrderedFloat;

use crate::{
    agents::{move_agents, Agent, PursuerAgent},
    configs::IsPlayable,
    gridworld::GRID_CELL_SIZE,
    world_objs::VisualMarker,
};

/// Plugins for determining what agents can see.
pub struct ObserverPlugin;

impl Plugin for ObserverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_vm_data, add_vis_cones))
            .add_systems(
                PostUpdate,
                (
                    update_observers.after(bevy_rapier2d::plugin::PhysicsSet::Writeback),
                    draw_observer_areas
                        .after(update_observers)
                        .after(add_vis_cones),
                ),
            );
    }
}

/// Implements playable functionality for ObserverPlugin.
pub struct ObserverPlayPlugin;

impl Plugin for ObserverPlayPlugin {
    fn build(&self, app: &mut App) {}
}

/// Stores visual marker data for an observer
#[derive(Copy, Clone)]
pub struct VMSeenData {
    /// When it was last seen (time since startup).
    pub last_seen: f32,
    /// When it was last seen (time since last seen, if never seen before this is time since startup).
    pub last_seen_elapsed: f32,
    /// The position the agent currently thinks this is in (if never seen before this is the position it starts at).
    pub pos: Vec2,
    /// The last known position of this object (if never seen before this is the position it starts at).
    pub last_pos: Vec2,
    /// True if the marker was modified by this agent this frame.
    pub pushed_by_self: bool,
}

/// Indicates that this entity can observe observable entities.
#[derive(Default, Component)]
pub struct Observer {
    /// Entities the observer can see.
    pub observing: Vec<Entity>,
    /// Stores data on visual markers that it's seen.
    pub seen_markers: HashMap<Entity, VMSeenData>,
    /// Stores a list of triangles that make up the observer's field of vision.
    pub vis_mesh: Vec<[Vec2; 3]>,
}

/// Indicates that this entity can be observed.
#[derive(Component)]
pub struct Observable;

/// Causes debug info for this observer to be displayed.
#[derive(Component)]
pub struct DebugObserver;

/// Blocks the observer's field of view.
/// Currently, only supports entities with rect colliders.
#[derive(Component)]
pub struct Wall;

/// Updates observers with observable entities they can see.
fn update_observers(
    wall_query: Query<(Entity, &Transform, &Collider), With<Wall>>,
    mut observer_query: Query<(Entity, &mut Observer, &Transform, &Agent)>,
    observable_query: Query<(Entity, &Transform), With<Observable>>,
    rapier_ctx: Res<RapierContext>,
) {
    // Collect wall endpoints
    let mut all_endpoints = Vec::new();
    for (_, wall_xform, wall_c) in wall_query.iter() {
        let rect = wall_c.as_cuboid().unwrap();
        let half = rect.raw.half_extents.xy();
        let x_axis = wall_xform.right().xy();
        let y_axis = wall_xform.up().xy();
        let center = wall_xform.translation.xy();
        let endpoints = (0..4)
            .map(|i| (((i % 2) * 2 - 1) as f32, ((i / 2) * 2 - 1) as f32))
            .map(|(x_sign, y_sign)| center + x_sign * x_axis * half.x + y_sign * y_axis * half.y)
            .collect::<Vec<_>>();
        all_endpoints.extend_from_slice(&endpoints);
    }

    // Draw per agent visibility triangles
    let walls = wall_query.iter().map(|(e, _, _)| e).collect::<Vec<_>>();
    for (observer_e, mut observer, observer_xform, agent) in observer_query.iter_mut() {
        // Draw vision cone
        let fov = 60_f32.to_radians();
        let start = observer_xform.translation.xy();
        let cone_l = Mat2::from_angle(-fov / 2.) * agent.dir;
        let cone_r = Mat2::from_angle(fov / 2.) * agent.dir;

        // Add cone boundaries to endpoints
        let mut sorted_endpoints = all_endpoints.clone();
        sorted_endpoints.extend_from_slice(&[start + cone_l, start + cone_r]);

        // Sort endpoints by angle and remove any points not within the vision cone
        sorted_endpoints.retain_mut(|p| {
            let dir = (*p - start).normalize();
            dir.dot(agent.dir).acos() <= fov / 2. + 0.01
        });
        sorted_endpoints.sort_unstable_by_key(|p| {
            let dir = (*p - start).normalize();
            OrderedFloat(dir.x * -dir.y.signum() - dir.y.signum())
        });

        let first_idx = sorted_endpoints
            .iter()
            .position(|p| p.abs_diff_eq(start + cone_l, 0.1))
            .unwrap_or(0);

        // Sweep from `cone_l` to `cone_r`
        let mut all_tris = Vec::new();
        for i in 0..sorted_endpoints.len() {
            let i = (i + first_idx) % sorted_endpoints.len();
            let p = sorted_endpoints[i];
            let dir = (p - start).normalize();
            let mut tri = Vec::new();
            for mat in [Mat2::from_angle(-0.001), Mat2::from_angle(0.001)] {
                let dir = mat * dir;
                let result = rapier_ctx.cast_ray(
                    start,
                    dir,
                    Real::MAX,
                    false,
                    QueryFilter::new().predicate(&|e| walls.contains(&e)),
                );
                if let Some((_, dist)) = result {
                    tri.push(start + dir * dist);
                }
            }
            if tri.len() == 2 {
                all_tris.push(tri);
            }
        }

        // Generate new vision mesh
        let mut vis_mesh = Vec::new();
        if !all_tris.is_empty() {
            for i in 0..(all_tris.len() - 1) {
                let next_i = (i + 1) % all_tris.len();
                let tri = &all_tris[i];
                let next_tri = &all_tris[next_i];
                vis_mesh.push([start, tri[1], next_tri[0]]);
            }
        }
        observer.vis_mesh = vis_mesh;

        // Check which observable objects fall within the mesh
        let mut observing = Vec::new();
        for (observable_e, observable_xform) in observable_query.iter() {
            if observable_e == observer_e {
                continue;
            }

            let p = observable_xform.translation.xy();
            for tri in &observer.vis_mesh {
                let d1 = sign(p, tri[0], tri[1]);
                let d2 = sign(p, tri[1], tri[2]);
                let d3 = sign(p, tri[2], tri[0]);

                let has_neg = d1 < 0. || d2 < 0. || d3 < 0.;
                let has_pos = d1 > 0. || d2 > 0. || d3 > 0.;

                if !(has_neg && has_pos) {
                    observing.push(observable_e);
                    break;
                }
            }
        }
        observer.observing = observing;
    }
}

/// Updates observers' visual marker data.
pub fn update_vm_data(
    mut observer_query: Query<(&mut Observer, &GlobalTransform, Option<&PursuerAgent>)>,
    visual_query: Query<(Entity, &GlobalTransform), With<VisualMarker>>,
    time: Res<Time>,
    is_playable: Option<Res<IsPlayable>>,
) {
    for (mut observer, agent_xform, pursuer) in observer_query.iter_mut() {
        // During gameplay, only update vm data right before observations are updated
        if is_playable.is_some() {
            if let Some(pursuer) = pursuer {
                let mut timer = pursuer.obs_timer.clone();
                timer.tick(time.delta());
                if !timer.just_finished() {
                    continue;
                }
            }
        }
        for (v_e, xform) in visual_query.iter() {
            let pushed_by_self = (xform.translation().xy() - agent_xform.translation().xy())
                .length_squared()
                <= (GRID_CELL_SIZE * 0.8).powi(2);
            if observer.observing.contains(&v_e) || pushed_by_self {
                let mut last_seen = 0.;
                let mut last_pos = xform.translation().xy();
                if let Some(vm_data) = observer.seen_markers.get(&v_e) {
                    last_seen = vm_data.last_seen;
                    last_pos = vm_data.pos;
                }
                observer.seen_markers.insert(
                    v_e,
                    VMSeenData {
                        last_seen: time.elapsed_seconds_wrapped(),
                        last_seen_elapsed: time.elapsed_seconds_wrapped() - last_seen,
                        pos: xform.translation().xy(),
                        last_pos,
                        pushed_by_self,
                    },
                );
            }
        }
    }
}

/// Helper function for detecting if a point is in a triangle.
fn sign(p1: Vec2, p2: Vec2, p3: Vec2) -> f32 {
    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
}

/// Marker component for vision cone visual.
#[derive(Component)]
struct VisCone;

/// Adds a vision cone to all `DebugObserver`s.
fn add_vis_cones(
    observer_query: Query<(Entity, &GlobalTransform), Added<DebugObserver>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for (e, xform) in observer_query.iter() {
        commands.entity(e).with_children(|p| {
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(
                        Mesh::new(
                            PrimitiveTopology::TriangleList,
                            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
                        )
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new()),
                    ),
                    material: materials.add(StandardMaterial {
                        base_color: Color::WHITE.with_a(0.1),
                        unlit: true,
                        alpha_mode: AlphaMode::Add,
                        ..default()
                    }),
                    transform: Transform::from_matrix(xform.compute_matrix().inverse()),
                    ..default()
                },
                VisCone,
            ));
        });
    }
}

/// Forces cones to be regenerated on each frame.
#[derive(Resource)]
pub struct RegenerateCones;

/// Draws visible areas for observers.
fn draw_observer_areas(
    observer_query: Query<(Entity, &Observer, &Children, &GlobalTransform), With<DebugObserver>>,
    mut vis_cone_query: Query<
        (&Handle<Mesh>, &mut Transform, &Handle<StandardMaterial>),
        With<VisCone>,
    >,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    regen_cones: Option<Res<RegenerateCones>>,
) {
    for (obs_e, observer, children, xform) in observer_query.iter() {
        let mut vertices = Vec::new();
        for tri in &observer.vis_mesh {
            vertices.push([tri[0].x, tri[0].y, 2.]);
            vertices.push([tri[1].x, tri[1].y, 2.]);
            vertices.push([tri[2].x, tri[2].y, 2.]);
        }
        for child in children.iter() {
            if let Ok((mesh_handle, mut cone_xform, material)) = vis_cone_query.get_mut(*child) {
                if let Some(mesh) = meshes.get_mut(mesh_handle) {
                    *cone_xform = Transform::from_matrix(xform.compute_matrix().inverse());
                    *mesh = mesh
                        .clone()
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone());
                    if regen_cones.is_some() {
                        commands.entity(*child).despawn_recursive();
                        commands.entity(obs_e).with_children(|p| {
                            p.spawn((
                                PbrBundle {
                                    mesh: mesh_handle.clone(),
                                    material: material.clone(),
                                    transform: *cone_xform,
                                    ..default()
                                },
                                VisCone,
                            ));
                        });
                    }
                }
                break;
            }
        }
    }
}
