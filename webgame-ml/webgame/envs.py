from typing import *
import gymnasium as gym
import pettingzoo  # type: ignore

# import rerun as rr  # type: ignore
from tqdm import tqdm
from webgame_rust import AgentState, GameWrapper, GameState
import numpy as np
import functools

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
        4: 1 if the other agent is visible, 0 if not
        5: If the other agent is visible, the other agent's x coordinate divided by map size
        6: If the other agent is visible, the other agent's y coordinate divided by map size

        , the second item is a 2D map showing where walls are, the third item is a list of items detected by the agent,
        and the fourth item is an attention mask for the previous item.

    Action Space: Discrete, check the `AgentAction` enum for a complete list.

    Args:
        visualize: If we should log visuals to Rerun.
    """

    def __init__(
        self,
        use_objs: bool = False,
        visualize: bool = False,
        recording_id: Optional[str] = None,
    ):
        self.game = GameWrapper(use_objs, visualize, recording_id)
        self.game_state: Optional[GameState] = None
        self.possible_agents = ["player", "pursuer"]
        self.agents = self.possible_agents[:]

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
        rewards = {
            "player": 0.0,
            "pursuer": 0.0,
        }
        dones = {
            "player": False,
            "pursuer": False,
        }
        truncs = {
            "player": False,
            "pursuer": False,
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
        assert self.game_state
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
        return gym.spaces.Discrete(10)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _: str) -> gym.Space:
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(0, 1, (7,)),
                gym.spaces.Box(0, 1, (8, 8)),
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
        obs_vec = np.zeros([7], dtype=float)
        obs_vec[0] = 0.5 + agent_state.pos.x / (game_state.level_size * CELL_SIZE)
        obs_vec[1] = 0.5 + agent_state.pos.y / (game_state.level_size * CELL_SIZE)
        obs_vec[2] = agent_state.dir.x
        obs_vec[3] = agent_state.dir.y

        other_agent = ["pursuer", "player"][int(is_pursuer)]
        other_e, other_obs = list(
            filter(lambda t: t[1].obj_type == other_agent, game_state.objects.items())
        )[0]
        if other_e in agent_state.observing:
            obs_vec[4] = 1
            obs_vec[5] = 0.5 + other_obs.pos.x / (game_state.level_size * CELL_SIZE)
            obs_vec[6] = 0.5 + other_obs.pos.y / (game_state.level_size * CELL_SIZE)

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
            obj_features[0] = obj_noise.pos.x / (game_state.level_size * CELL_SIZE)
            obj_features[1] = obj_noise.pos.y / (game_state.level_size * CELL_SIZE)
            obj_features[3] = 1
            obj_features[4] = obj_noise.active_radius
            obs_vecs[i + len(agent_state.observing)] = obj_features

        attn_mask = np.zeros([MAX_OBJS])
        attn_mask[len(agent_state.observing) + len(agent_state.listening) :] = 1

        return (obs_vec, walls, obs_vecs, attn_mask)


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
