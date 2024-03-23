use std::collections::HashMap;

use bevy::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::{exceptions::PyValueError, prelude::*};
use webgame_game::{
    configs::LibCfgPlugin,
    gridworld::{Agent, LevelLayout, NextAction, PlayerAgent, PursuerAgent},
    observer::{Observable, Observer},
};

/// Describes an observable object.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ObservableObject {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub obj_type: String,
}

/// Represents a 2D vector.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct PyVec2 {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
}

impl From<Vec2> for PyVec2 {
    fn from(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

/// Contains the state of an agent for a single frame.
#[pyclass]
#[derive(Debug, Clone)]
pub struct AgentState {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub dir: PyVec2,
    #[pyo3(get)]
    pub observing: Vec<u64>,
}

/// Contains the state of the game for a single frame.
#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    #[pyo3(get)]
    pub player: AgentState,
    #[pyo3(get)]
    pub pursuer: AgentState,
    #[pyo3(get)]
    pub walls: Vec<bool>,
    #[pyo3(get)]
    pub level_size: usize,
    #[pyo3(get)]
    pub objects: HashMap<u64, ObservableObject>,
}

/// Indicates the kind of actions an agent can take.
#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum AgentAction {
    NoAction = 0,
    MoveUp = 1,
    MoveUpRight = 2,
    MoveRight = 3,
    MoveDownRight = 4,
    MoveDown = 5,
    MoveDownLeft = 6,
    MoveLeft = 7,
    MoveUpLeft = 8,
    ToggleObj = 9,
}

impl<'source> FromPyObject<'source> for AgentAction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index: u8 = ob.extract()?;
        Self::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid action"))
    }
}

/// Wraps our game in a gym-like interface.
#[pyclass]
pub struct GameWrapper {
    pub app: App,
}

#[pymethods]
impl GameWrapper {
    #[new]
    pub fn new() -> Self {
        let mut app = App::new();
        app.add_plugins(LibCfgPlugin);

        app.finish();
        app.cleanup();
        app.update();

        Self { app }
    }

    pub fn step(&mut self, action_player: AgentAction, action_pursuer: AgentAction) -> GameState {
        set_agent_action::<PlayerAgent>(&mut self.app.world, action_player);
        set_agent_action::<PursuerAgent>(&mut self.app.world, action_pursuer);

        self.app.update();

        self.get_state()
    }

    pub fn reset(&mut self) -> GameState {
        *self = Self::new();
        self.get_state()
    }
}

/// Queries the world for an agent with the provided component and sets the next action.
fn set_agent_action<T: Component>(world: &mut World, action: AgentAction) {
    let mut next_action = world
        .query_filtered::<&mut NextAction, With<T>>()
        .single_mut(world);
    next_action.dir = match action {
        AgentAction::MoveUp => Vec2::Y,
        AgentAction::MoveUpRight => (Vec2::Y + Vec2::X).normalize(),
        AgentAction::MoveRight => Vec2::X,
        AgentAction::MoveDownRight => (-Vec2::Y + Vec2::X).normalize(),
        AgentAction::MoveDown => -Vec2::Y,
        AgentAction::MoveDownLeft => (-Vec2::Y + -Vec2::X).normalize(),
        AgentAction::MoveLeft => -Vec2::X,
        AgentAction::MoveUpLeft => (Vec2::Y + -Vec2::X).normalize(),
        _ => Vec2::ZERO,
    };
    action.toggle_obj = action == AgentAction::ToggleObj;
}

/// Queries the world for an agent with the provided component and returns an `AgentState`.
fn get_agent_state<T: Component>(world: &mut World) -> AgentState {
    let (agent, xform, observer) = world
        .query_filtered::<(&Agent, &Transform, &Observer), With<T>>()
        .single(world);
    AgentState {
        pos: xform.translation.xy().into(),
        dir: agent.dir.into(),
        observing: observer.observing.iter().map(|e| e.to_bits()).collect(),
    }
}

impl GameWrapper {
    fn get_state(&mut self) -> GameState {
        let world = &mut self.app.world;
        let player = get_agent_state::<PlayerAgent>(world);
        let pursuer = get_agent_state::<PursuerAgent>(world);
        let mut observables =
            world.query_filtered::<(Entity, &Transform, Option<&Agent>), With<Observable>>();
        let mut objects = HashMap::new();
        for (e, xform, agent) in observables.iter(world) {
            if agent.is_some() {
                objects.insert(
                    e.to_bits(),
                    ObservableObject {
                        pos: xform.translation.xy().into(),
                        obj_type: "agent".into(),
                    },
                );
            }
        }
        let level = world.get_resource::<LevelLayout>().unwrap();
        GameState {
            player,
            pursuer,
            walls: level.walls.clone(),
            level_size: level.size,
            objects,
        }
    }
}

impl Default for GameWrapper {
    fn default() -> Self {
        Self::new()
    }
}

#[pymodule]
fn webgame_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameWrapper>()?;
    m.add_class::<ObservableObject>()?;
    m.add_class::<GameState>()?;
    m.add_class::<AgentState>()?;
    m.add_class::<PyVec2>()?;
    Ok(())
}
