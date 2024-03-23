use bevy::prelude::*;
use configs::ReleaseCfgPlugin;

mod net;
mod configs;
mod gridworld;
mod observer;
mod world_objs;

/// Main entry point for our game.
fn main() {
    App::new().add_plugins(ReleaseCfgPlugin).run();
}
