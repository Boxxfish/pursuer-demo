from typing import *
import gymnasium as gym
import pettingzoo  # type: ignore

# import rerun as rr  # type: ignore
from tqdm import tqdm
from webgame_rust import AgentState, GameWrapper, GameState
import numpy as np
import functools
import math

from webgame.common import pos_to_grid, process_obs
from webgame.filter import BayesFilter

# The maximum number of object vectors supported by the environment.
MAX_OBJS = 16
# The dimension of each object vector.
OBJ_DIM = 8

# The world space size of a grid cell
CELL_SIZE = 25


class GameEnv(pettingzoo.ParallelEnv):
    """
    An environment that wraps an instance of our game.

    Agents: pursuer, player

    Observation Space: A tuple, where the first item is a vector of the following form:

        0: This agent's x coordinate, normalized between 0 and 1
        1: This agent's y coordinate, normalized between 0 and 1
        2: This agent's direction vector's x coordinate, normalized
        3: This agent's direction vector's y coordinate, normalized

        , the second item is a 2D map showing where walls are, the third item is a list of items detected by the agent,
        and the fourth item is an attention mask for the previous item.

    Action Space: Discrete, check the `AgentAction` enum for a complete list.

    Args:
        visualize: If we should log visuals to Rerun.
    """

    def __init__(
        self,
        use_objs: bool = False,
        wall_prob: float = 0.1,
        max_timer: Optional[int] = None,
        visualize: bool = False,
        recording_id: Optional[str] = None,
        update_fn: Optional[
            Callable[
                [
                    Tuple[np.ndarray, np.ndarray, np.ndarray],
                    bool,
                    GameState,
                    AgentState,
                    int,
                    float,
                    bool,
                ],
                np.ndarray,
            ]
        ] = None,
        insert_visible_cells: bool = False,
        player_sees_visible_cells: bool = False,
        aux_rew_amount: float = 0.0,
        grid_size: int = 8,
        start_gt: bool = False,
    ):
        self.game = GameWrapper(use_objs, wall_prob, grid_size, visualize, recording_id)
        self.game_state: Optional[GameState] = None
        self.possible_agents = ["player", "pursuer"]
        self.agents = self.possible_agents[:]
        self.timer = 0
        self.max_timer = max_timer
        self.use_objs = use_objs
        self.update_fn = update_fn
        self.filters: Optional[Dict[str, BayesFilter]] = None
        self.insert_visible_cells = insert_visible_cells
        self.player_sees_visible_cells = player_sees_visible_cells
        self.aux_rew_amount = aux_rew_amount
        self.grid_size = grid_size
        self.start_gt = start_gt

    def step(self, actions: Mapping[str, int]) -> tuple[
        Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        Mapping[str, float],
        Mapping[str, bool],
        Mapping[str, bool],
        Mapping[str, None],
    ]:
        all_actions = []
        for agent in ["player", "pursuer"]:
            all_actions.append(actions[agent])
        self.game_state = self.game.step(all_actions[0], all_actions[1])
        assert self.game_state
        obs = self.game_state_to_obs(self.game_state)

        # Check if pursuer can see player
        player_e, _ = list(
            filter(lambda t: t[1].obj_type == "player", self.game_state.objects.items())
        )[0]
        seen_player = player_e in self.game_state.pursuer.observing

        self.timer += 1
        trunc = self.timer == self.max_timer

        player_pos = self.game_state.player.pos
        dx = (0.5 * CELL_SIZE + player_pos.x) - (
            self.game_state.level_size * CELL_SIZE / 2
        )
        dy = (0.5 * CELL_SIZE + player_pos.y) - (
            self.game_state.level_size * CELL_SIZE / 2
        )
        player_aux_rew = -math.sqrt(dx**2 + dy**2) / math.sqrt(
            ((self.game_state.level_size * CELL_SIZE / 2) ** 2) * 2
        )

        pursuer_pos = self.game_state.pursuer.pos
        dx = player_pos.x - pursuer_pos.x
        dy = player_pos.y - pursuer_pos.y
        pursuer_aux_rew = math.sqrt(dx**2 + dy**2) / math.sqrt(
            ((self.game_state.level_size * CELL_SIZE) ** 2) * 2
        )

        rewards = {
            "player": -float(seen_player) + player_aux_rew * self.aux_rew_amount,
            "pursuer": float(seen_player) + pursuer_aux_rew * self.aux_rew_amount,
        }
        dones = {
            "player": seen_player,
            "pursuer": seen_player,
        }
        truncs = {
            "player": trunc,
            "pursuer": trunc,
        }
        infos = {
            "player": None,
            "pursuer": None,
        }
        return (obs, rewards, dones, truncs, infos)

    def reset(self, *args) -> tuple[
        Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        Mapping[str, None],
    ]:
        self.game_state = self.game.reset()
        while not self.check_path():
            self.game_state = self.game.reset()
        assert self.game_state
        self.state = self.game.step(0, 0)

        self.timer = 0
        if self.update_fn:
            self.filters = {
                agent: BayesFilter(
                    self.game_state.level_size,
                    CELL_SIZE,
                    self.update_fn,
                    self.use_objs,
                    agent == "pursuer",
                )
                for agent in self.agents
            }
            if self.start_gt:
                self.filters["pursuer"].belief = np.zeros(
                    self.filters["pursuer"].belief.shape
                )
                play_pos = self.game_state.player.pos
                x, y = pos_to_grid(
                    play_pos.x, play_pos.y, self.game_state.level_size, CELL_SIZE
                )
                self.filters["pursuer"].belief[y, x] = 1
        obs = self.game_state_to_obs(self.game_state)
        infos = {
            "player": None,
            "pursuer": None,
        }
        return (obs, infos)

    def game_state_to_obs(
        self,
        game_state: GameState,
    ) -> Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Converts the game state to our expected observations.
        """
        return {
            "player": self.agent_state_to_obs(game_state.player, game_state, False),
            "pursuer": self.agent_state_to_obs(game_state.pursuer, game_state, True),
        }

    @functools.lru_cache(maxsize=None)
    def action_space(self, _agent: str) -> gym.Space:
        return gym.spaces.Discrete(9)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _: str) -> gym.Space:
        grid_channels = 2
        if self.player_sees_visible_cells:
            grid_channels = 3
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(0, 1, (7,)),
                gym.spaces.Box(0, 1, (grid_channels, self.grid_size, self.grid_size)),
                gym.spaces.Box(0, 1, (MAX_OBJS, OBJ_DIM)),
                gym.spaces.Box(0, 1, (MAX_OBJS,)),
            )
        )

    def agent_state_to_obs(
        self, agent_state: AgentState, game_state: GameState, is_pursuer: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates observations for an agent.
        """
        obs_vec = np.zeros([4], dtype=float)
        obs_vec[0] = (0.5 * CELL_SIZE + agent_state.pos.x) / (
            game_state.level_size * CELL_SIZE
        )
        obs_vec[1] = (0.5 * CELL_SIZE + agent_state.pos.y) / (
            game_state.level_size * CELL_SIZE
        )
        obs_vec[2] = agent_state.dir.x
        obs_vec[3] = agent_state.dir.y

        walls = np.array(game_state.walls, dtype=float).reshape(
            (game_state.level_size, game_state.level_size)
        )

        obs_vecs = np.zeros([MAX_OBJS, OBJ_DIM], dtype=float)
        for i, e in enumerate(agent_state.observing):
            if e in agent_state.vm_data:
                obs_obj = game_state.objects[e]
                obj_features = np.zeros([OBJ_DIM])
                obj_features[0] = 0.5 + obs_obj.pos.x / (
                    game_state.level_size * CELL_SIZE
                )
                obj_features[1] = 0.5 + obs_obj.pos.y / (
                    game_state.level_size * CELL_SIZE
                )
                obj_features[2] = 1
                vm_data = agent_state.vm_data[e]
                obj_features[5] = vm_data.last_seen_elapsed / 10.0
                obj_features[6] = obs_obj.pos.x - vm_data.last_pos.x
                obj_features[7] = obs_obj.pos.y - vm_data.last_pos.y
                obs_vecs[i] = obj_features
        for i, e in enumerate(agent_state.listening):
            obj_noise = game_state.noise_sources[e]
            obj_features = np.zeros([OBJ_DIM])
            obj_features[0] = 0.5 + obj_noise.pos.x / (
                game_state.level_size * CELL_SIZE
            )
            obj_features[1] = 0.5 + obj_noise.pos.y / (
                game_state.level_size * CELL_SIZE
            )
            obj_features[3] = 1
            obj_features[4] = obj_noise.active_radius
            obs_vecs[i + len(agent_state.observing)] = obj_features

        attn_mask = np.zeros([MAX_OBJS])
        attn_mask[len(agent_state.observing) + len(agent_state.listening) :] = 1

        agent_name = ["player", "pursuer"][int(is_pursuer)]

        if self.player_sees_visible_cells and not is_pursuer:
            loc_channel = np.zeros(walls.shape, dtype=float)
            pursuer_pos = game_state.pursuer.pos
            x, y = pos_to_grid(
                pursuer_pos.x, pursuer_pos.y, game_state.level_size, CELL_SIZE
            )
            loc_channel[y, x] = 1
            visible_cells = game_state.pursuer.visible_cells
            cells_channel = np.array(visible_cells).reshape(
                [game_state.level_size, game_state.level_size]
            )
            grid = np.stack([walls, loc_channel, cells_channel])
        else:
            extra_channel = np.zeros(walls.shape, dtype=float)
            assert not (self.filters is not None and self.insert_visible_cells)
            if self.filters:
                extra_channel = self.filters[agent_name].localize(
                    process_obs(
                        (
                            obs_vec,
                            np.stack([walls, extra_channel]),
                            obs_vec,
                            attn_mask,
                        )
                    ),
                    game_state,
                    agent_state,
                )
            if self.insert_visible_cells:
                visible_cells = [game_state.player, game_state.pursuer][
                    int(is_pursuer)
                ].visible_cells
                extra_channel = np.array(visible_cells).reshape(
                    [game_state.level_size, game_state.level_size]
                )
            if self.player_sees_visible_cells:
                grid = np.stack(
                    [walls, extra_channel, np.zeros(walls.shape, dtype=float)]
                )
            else:
                grid = np.stack([walls, extra_channel])

        return (obs_vec, grid, obs_vecs, attn_mask)

    def check_path(self) -> bool:
        assert self.game_state
        player_pos = self.game_state.player.pos
        start_pos = pos_to_grid(
            player_pos.x, player_pos.y, self.game_state.level_size, CELL_SIZE
        )
        pursuer_pos = self.game_state.pursuer.pos
        target_pos = pos_to_grid(
            pursuer_pos.x, pursuer_pos.y, self.game_state.level_size, CELL_SIZE
        )

        queue = [start_pos]
        parents: Dict[tuple[int, int], tuple[int, int]] = {}
        while len(queue) > 0:
            curr_pos = queue.pop(0)
            neighbors_delta = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ]
            finished = False
            for n_delta in neighbors_delta:
                neighbor = (curr_pos[0] + n_delta[0], curr_pos[1] + n_delta[1])

                def is_wall(pos: tuple[int, int]) -> bool:
                    assert self.game_state
                    if (
                        pos[0] < 0
                        or pos[0] >= self.game_state.level_size
                        or pos[1] < 0
                        or pos[1] >= self.game_state.level_size
                    ):
                        return True
                    return self.game_state.walls[
                        pos[1] * self.game_state.level_size + pos[0]
                    ]

                if not is_wall(neighbor) and neighbor not in parents.keys():
                    queue.append(neighbor)
                    parents[neighbor] = curr_pos
                    if neighbor == target_pos:
                        finished = True
                        break
            if finished:
                break

        return target_pos in parents.keys()


if __name__ == "__main__":
    env = GameEnv(visualize=False)
    env.reset()

    for _ in tqdm(range(1000)):
        env.step(
            {
                "player": env.action_space("player").sample(),
                "pursuer": env.action_space("pursuer").sample(),
            }
        )
