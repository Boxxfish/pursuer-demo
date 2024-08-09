from typing import *

class ObservableObj:
    """
    Describes an observable object.
    """
    pos: PyVec2
    obj_type: str

class NoiseSourceObj:
    """
    Describes a noise source.
    """
    pos: PyVec2
    active_radius: float

class PyVec2:
    """
    Represents a 2D vector.
    """
    x: float
    y: float

class VMData:
    """
    Data on visual markers.
    """
    last_seen: float
    last_seen_elapsed: float
    last_pos: PyVec2
    pushed_by_self: bool

class AgentState:
    """
    Contains the state of an agent for a single frame.
    """
    pos: PyVec2
    dir: PyVec2
    observing: list[int]
    listening: list[int]
    vm_data: Mapping[int, VMData]
    visible_cells: list[float]

class GameState:
    """
    Contains the state of the game for a single frame.
    """
    player: AgentState
    pursuer: AgentState
    walls: list[bool]
    level_size: int
    objects: Mapping[int, ObservableObj]
    noise_sources: Mapping[int, NoiseSourceObj]

class GameWrapper:
    def __init__(self, use_objs: bool, wall_prob: float, grid_size: int, visualize: bool, recording_id: Optional[str]) -> None:
        """
        Args:
            use_objs: Whether the environment should add objects to the scene.
            wall_prob: Probability of each tile being a wall.
            grid_size: Size of the grid.
            visualize: If we should log visuals to Rerun.
            recording_id: Recording ID used by Rerun. Useful for syncing data between Python and Rust.
        """
        ...
    def step(
        self, action_player: int, action_pursuer: int
    ) -> GameState:
        """
        Runs one step of the game, and returns the next state of the game.
        """
        ...
    def reset(self) -> GameState: 
        """
        Resets the game, returning the next state of the game.
        """
        ...
