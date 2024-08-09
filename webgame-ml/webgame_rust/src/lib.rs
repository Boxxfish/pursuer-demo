use std::collections::HashMap;

use bevy::{app::AppExit, prelude::*};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::{exceptions::PyValueError, prelude::*};
use webgame_game::{
    agents::{Agent, NextAction, PlayerAgent, PursuerAgent, UseGridPositions},
    configs::{LibCfgPlugin, VisualizerPlugin},
    gridworld::{LevelLayout, ResetEvent, GRID_CELL_SIZE},
    observations::fill_tri_half,
    observer::{Observable, Observer, RegenerateCones},
    screens::ScreenState,
    world_objs::NoiseSource,
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

/// Describes a noise source in the environment.
#[pyclass]
#[derive(Debug, Clone)]
pub struct NoiseSourceObject {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub active_radius: f32,
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

/// Stores data for visual markers.
#[pyclass]
#[derive(Debug, Clone)]
pub struct VMData {
    #[pyo3(get)]
    pub last_seen: f32,
    #[pyo3(get)]
    pub last_seen_elapsed: f32,
    #[pyo3(get)]
    pub last_pos: PyVec2,
    #[pyo3(get)]
    pub pushed_by_self: bool,
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
    #[pyo3(get)]
    pub listening: Vec<u64>,
    #[pyo3(get)]
    pub vm_data: HashMap<u64, VMData>,
    #[pyo3(get)]
    pub visible_cells: Vec<f32>,
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
    #[pyo3(get)]
    pub noise_sources: HashMap<u64, NoiseSourceObject>,
}

/// Indicates the kind of actions an agent can take.
#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive, PartialEq, Eq)]
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
    pub use_objs: bool,
    pub wall_prob: f64,
    pub visualize: bool,
    pub recording_id: Option<String>,
    pub grid_size: usize,
}

#[pymethods]
impl GameWrapper {
    #[new]
    pub fn new(
        use_objs: bool,
        wall_prob: f64,
        grid_size: usize,
        visualize: bool,
        recording_id: Option<String>,
    ) -> Self {
        let mut app = App::new();
        app.add_plugins(LibCfgPlugin);
        app.insert_state(ScreenState::Game);
        app.insert_resource(LevelLayout::random(
            grid_size,
            wall_prob,
            if use_objs { grid_size } else { 0 },
        ));
        app.insert_resource(UseGridPositions);

        if visualize {
            app.insert_resource(RegenerateCones);
            app.add_plugins(VisualizerPlugin {
                recording_id: recording_id.clone(),
            });
        }

        app.finish();
        app.cleanup();
        app.update();

        Self {
            app,
            visualize,
            recording_id,
            use_objs,
            wall_prob,
            grid_size,
        }
    }

    pub fn step(&mut self, action_player: AgentAction, action_pursuer: AgentAction) -> GameState {
        set_agent_action::<PlayerAgent>(&mut self.app.world, action_player);
        set_agent_action::<PursuerAgent>(&mut self.app.world, action_pursuer);

        self.app.update();

        self.get_state()
    }

    pub fn reset(&mut self) -> GameState {
        self.app.world.send_event(ResetEvent {
            level: LevelLayout::random(
                self.grid_size,
                self.wall_prob,
                if self.use_objs { self.grid_size } else { 0 },
            ),
        });
        self.app.update();
        self.app.update();
        self.get_state()
    }
}

/// Queries the world for an agent with the provided component and sets the next action.
fn set_agent_action<T: Component>(world: &mut World, action: AgentAction) {
    let mut next_action = world
        .query_filtered::<&mut NextAction, With<T>>()
        .single_mut(world);
    let dir = match action {
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
    next_action.dir = dir;
    next_action.toggle_objs = action == AgentAction::ToggleObj;
}

/// Queries the world for an agent with the provided component and returns an `AgentState`.
fn get_agent_state<T: Component>(world: &mut World, size: usize) -> AgentState {
    let (agent_e, agent, &xform, observer) = world
        .query_filtered::<(Entity, &Agent, &GlobalTransform, &Observer), With<T>>()
        .single(world);
    let vis_mesh = observer.vis_mesh.clone();
    let pos = xform.translation().xy().into();
    let dir = agent.dir.into();
    let observing = observer.observing.iter().map(|e| e.to_bits()).collect();
    let vm_data = observer
        .seen_markers
        .iter()
        .map(|(e, vm_data)| {
            (
                e.to_bits(),
                VMData {
                    last_seen: vm_data.last_seen,
                    last_seen_elapsed: vm_data.last_seen_elapsed,
                    last_pos: vm_data.last_pos.into(),
                    pushed_by_self: vm_data.pushed_by_self,
                },
            )
        })
        .collect();

    let listening = world
        .query::<(Entity, &GlobalTransform, &NoiseSource)>()
        .iter(world)
        .filter(|(_, noise_xform, noise_src)| {
            (xform.translation().xy() - noise_xform.translation().xy()).length_squared()
                <= noise_src.noise_radius.powi(2)
                && noise_src.activated_by_player
        })
        .map(|(e, _, _)| e.to_bits())
        .collect();

    // Compute intersection of agent visible area with grid
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
    }
}

impl GameWrapper {
    fn get_state(&mut self) -> GameState {
        let world = &mut self.app.world;
        let size = world.get_resource::<LevelLayout>().unwrap().size;
        let player = get_agent_state::<PlayerAgent>(world, size);
        let pursuer = get_agent_state::<PursuerAgent>(world, size);

        // Record all observable items
        let mut observables = world.query_filtered::<(
            Entity,
            &GlobalTransform,
            Option<&PlayerAgent>,
            Option<&PursuerAgent>,
        ), With<Observable>>();
        let mut objects = HashMap::new();
        for (e, xform, player, pursuer) in observables.iter(world) {
            if player.is_some() {
                objects.insert(
                    e.to_bits(),
                    ObservableObject {
                        pos: xform.translation().xy().into(),
                        obj_type: "player".into(),
                    },
                );
            } else if pursuer.is_some() {
                objects.insert(
                    e.to_bits(),
                    ObservableObject {
                        pos: xform.translation().xy().into(),
                        obj_type: "pursuer".into(),
                    },
                );
            } else {
                objects.insert(
                    e.to_bits(),
                    ObservableObject {
                        pos: xform.translation().xy().into(),
                        obj_type: "visual".into(),
                    },
                );
            }
        }

        // Record all noise sources
        let mut noise_srcs = world.query::<(Entity, &GlobalTransform, &NoiseSource)>();
        let mut noise_sources = HashMap::new();
        for (e, xform, noise_src) in noise_srcs.iter(world) {
            noise_sources.insert(
                e.to_bits(),
                NoiseSourceObject {
                    pos: xform.translation().xy().into(),
                    active_radius: noise_src.active_radius,
                },
            );
        }

        let level = world.get_resource::<LevelLayout>().unwrap();
        GameState {
            player,
            pursuer,
            walls: level.walls.clone(),
            level_size: level.size,
            objects,
            noise_sources,
        }
    }
}

impl Default for GameWrapper {
    fn default() -> Self {
        Self::new(false, 0.1, 8, false, None)
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
