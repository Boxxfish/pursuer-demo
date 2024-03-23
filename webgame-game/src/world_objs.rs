use crate::{
    gridworld::{NextAction, GRID_CELL_SIZE},
    observer::Wall,
};
use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

/// Plugin for world objects (e.g. doors, noise sources).
pub struct WorldObjPlugin;

impl Plugin for WorldObjPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_door);
    }
}

/// Adds playable functionality for `WorldObjPlugin`.
pub struct WorldObjPlayPlugin;

impl Plugin for WorldObjPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, visualize_door);
    }
}

/// A door that can be opened and closed.
#[derive(Component, Default)]
pub struct Door {
    pub open: bool,
}

/// How close an object needs to be before the agent can toggle it.
const TOGGLE_DIST: f32 = GRID_CELL_SIZE * 1.5;

/// Opens and closes the door if the agent is not touching the door and it toggles nearby objects.
fn update_door(
    mut commands: Commands,
    agent_query: Query<(&GlobalTransform, &NextAction)>,
    mut door_query: Query<(Entity, &GlobalTransform, &mut Door)>,
) {
    for (agent_xform, action) in agent_query.iter() {
        let agent_pos = agent_xform.translation().xy();
        if action.toggle_objs {
            for (e, obj_xform, mut door) in door_query.iter_mut() {
                let obj_pos = obj_xform.translation().xy();
                let dist_sq = (obj_pos - agent_pos).length_squared();
                if dist_sq >= (GRID_CELL_SIZE / 2.).powi(2) && dist_sq < TOGGLE_DIST.powi(2) {
                    door.open = !door.open;
                    if door.open {
                        commands.entity(e).remove::<(Wall, Collider)>();
                    } else {
                        commands.entity(e).insert((
                            Wall,
                            Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                        ));
                    }
                }
            }
        }
    }
}

/// Updates the door visual.
fn visualize_door(
    mut commands: Commands,
    mut door_query: Query<(Entity, &Door, Option<&mut Sprite>), Changed<Door>>,
) {
    for (e, door, sprite) in door_query.iter_mut() {
        if door.open {
            sprite.unwrap().color.set_a(0.5);
            commands.entity(e).remove::<Wall>();
        } else if let Some(mut sprite) = sprite {
            sprite.color.set_a(1.);
            commands.entity(e).insert(Wall);
        } else {
            commands.entity(e).insert((
                Sprite {
                    color: Color::MAROON,
                    custom_size: Some(Vec2::ONE * GRID_CELL_SIZE),
                    ..default()
                },
                Handle::<Image>::default(),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
            commands.entity(e).insert(Wall);
        }
    }
}
