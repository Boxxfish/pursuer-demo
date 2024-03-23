//! Defines various configurations our game can be in.

use std::time::Duration;

use bevy::{
    asset::AssetMetaCheck, input::InputPlugin, prelude::*, scene::ScenePlugin,
    time::TimeUpdateStrategy,
};
use bevy_rapier2d::prelude::*;

use crate::{
    gridworld::{GridworldPlayPlugin, GridworldPlugin},
    net::NetPlugin,
    observer::{ObserverPlayPlugin, ObserverPlugin},
    world_objs::{WorldObjPlayPlugin, WorldObjPlugin},
};

/// Handles core functionality for our game (i.e. gameplay logic).
pub struct CoreGamePlugin;

impl Plugin for CoreGamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
            .add_plugins((NetPlugin, GridworldPlugin, ObserverPlugin, WorldObjPlugin));
    }
}

/// Adds functionality required to make the game playable (e.g. graphics and input handling).
pub struct PlayablePlugin;

impl Plugin for PlayablePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AssetMetaCheck::Never)
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Your Project (Game)".into(),
                    resolution: (640., 360.).into(),
                    ..default()
                }),
                ..default()
            }))
            .add_plugins(RapierDebugRenderPlugin::default())
            .add_plugins((GridworldPlayPlugin, ObserverPlayPlugin, WorldObjPlayPlugin));
    }
}

/// The configuration for published builds.
pub struct ReleaseCfgPlugin;

impl Plugin for ReleaseCfgPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((PlayablePlugin, CoreGamePlugin));
    }
}

/// The configuration for library builds (e.g. for machine learning).
pub struct LibCfgPlugin;

const FIXED_TS: f32 = 0.02;

impl Plugin for LibCfgPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            HierarchyPlugin,
            InputPlugin,
            AssetPlugin::default(),
            ScenePlugin,
            CoreGamePlugin,
        ))
        // Use constant timestep
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            FIXED_TS,
        )))
        .insert_resource(RapierConfiguration {
            timestep_mode: TimestepMode::Fixed {
                dt: FIXED_TS,
                substeps: 10,
            },
            ..default()
        });
    }
}
