import math
from typing import *
import numpy as np
from scipy import signal  # type: ignore

import torch
from torch.distributions import Categorical
from safetensors.torch import load_model
from webgame_rust import AgentState, GameState

from webgame.common import convert_obs, explore_policy, pos_to_grid, process_obs

import gymnasium as gym
from torch import Tensor, nn
from webgame.models import MeasureModel, PolicyNet


class BayesFilter:
    """
    A discrete Bayes filter for localization.
    """

    def __init__(
        self,
        size: int,
        cell_size: float,
        update_fn: Callable[
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
        ],
        use_objs: bool,
        is_pursuer: bool,
        lkhd_min: float = 0.0,
    ):
        self.size = size
        self.cell_size = cell_size
        self.belief = np.ones([size, size]) / size**2
        self.update_fn = update_fn
        self.use_objs = use_objs
        self.is_pursuer = is_pursuer
        self.lkhd_min = lkhd_min

    def localize(
        self,
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        game_state: GameState,
        agent_state: AgentState,
    ) -> np.ndarray:
        """
        Given an agent's observations, returns the new location probabilities.
        """
        walls = np.array(game_state.walls).astype(float)
        walls = walls.reshape([game_state.level_size, game_state.level_size])
        self.belief = self.predict(self.belief, walls)
        lkhd = self.update_fn(
            obs,
            self.use_objs,
            game_state,
            agent_state,
            self.size,
            self.cell_size,
            self.is_pursuer,
        )
        lkhd = lkhd * (1 - self.lkhd_min) + self.lkhd_min
        self.belief = lkhd * self.belief
        if self.belief.sum() < 0.0001:
            print("Warning: Belief summed to zero. Resetting filter.")
            self.belief = np.ones(self.belief.shape)
        self.belief = self.belief / self.belief.sum()
        return self.belief

    def predict(self, belief: np.ndarray, walls: np.ndarray) -> np.ndarray:
        kernel = np.array([[0.25, 1, 0.25], [1, 1, 1], [0.25, 1, 0.25]])
        belief = signal.convolve2d(belief, kernel, mode="same")
        denom = signal.convolve2d(1.0 - walls, kernel, mode="same") + 0.001
        return belief / denom


def manual_update(
    obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    use_objs: bool,
    game_state: GameState,
    agent_state: AgentState,
    size: int,
    cell_size: float,
    is_pursuer: bool,
) -> np.ndarray:
    # Check whether agent can see the player
    other_agent = ["pursuer", "player"][int(is_pursuer)]
    other_e, other_obs = list(
        filter(lambda t: t[1].obj_type == other_agent, game_state.objects.items())
    )[0]
    player_vis_grid = None
    if other_e in agent_state.observing:
        player_vis_grid = pos_to_grid(other_obs.pos.x, other_obs.pos.y, size, cell_size)

    obs_grid = np.array(game_state.walls).reshape([size, size])
    grid_lkhd = 1 - obs_grid
    if player_vis_grid is not None:
        agent_lkhd = np.zeros([size, size])
        agent_lkhd[player_vis_grid[1], player_vis_grid[0]] = 1
    else:
        visible_cells = np.array(agent_state.visible_cells).reshape([size, size])
        # Cells within vision have 0% chance of agent being there
        agent_lkhd = 1.0 - visible_cells
        # All other cells are equally probable
        agent_lkhd = agent_lkhd / (size**2 - visible_cells.sum())
    lkhd = grid_lkhd * agent_lkhd
    agent_pos = agent_state.pos
    for y in range(size):
        for x in range(size):
            noise_lkhd = 1.0
            vis_lkhd = 1.0
            if player_vis_grid is None:
                # If any noise sources are triggered, make the likelihood a normal distribution centered on it
                pos = np.array([x, y], dtype=float) * cell_size
                for obj_id in agent_state.listening:
                    noise_obj = game_state.noise_sources[obj_id]
                    mean = np.array([noise_obj.pos.x, noise_obj.pos.y])
                    var = ((mean - np.array([agent_pos.x, agent_pos.y])) ** 2).sum()
                    dev = np.sqrt(var) / 4
                    val = np.exp(-((((pos - mean) ** 2).sum() / (2 * dev**2))))
                    noise_lkhd *= val

                # If any visual markers are moved, we can localize the player based on its start position, end position,
                # and how long it's been since the pursuer last looked at it
                max_speed = cell_size
                for obj_id in agent_state.observing:
                    if obj_id in agent_state.vm_data:
                        vm_data = agent_state.vm_data[obj_id]
                        obs_obj = game_state.objects[obj_id]
                        if vm_data.last_seen_elapsed > 1 and not vm_data.pushed_by_self:
                            last_pos = np.array(
                                [vm_data.last_pos.x, vm_data.last_pos.y]
                            )
                            curr_pos = np.array([obs_obj.pos.x, obs_obj.pos.y])
                            moved_amount = ((last_pos - curr_pos) ** 2).sum()
                            if moved_amount > 0.1:
                                curr_dist = ((pos - curr_pos) ** 2).sum()
                                max_dist = (max_speed * vm_data.last_seen_elapsed) ** 2
                                if curr_dist > max_dist:
                                    vis_lkhd = 0.0

            lkhd[y][x] *= noise_lkhd * vis_lkhd
    return lkhd


def model_update(
    model: nn.Module,
) -> Callable[
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
]:
    @torch.no_grad()
    def model_update_(
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        use_objs: bool,
        game_state: GameState,
        agent_state: AgentState,
        size: int,
        cell_size: float,
        is_pursuer: bool,
    ) -> np.ndarray:
        lkhd = (
            model(
                torch.from_numpy(obs[0]).float(),
                torch.from_numpy(obs[1]).float() if use_objs else None,
                torch.from_numpy(obs[2]).float() if use_objs else None,
            )
            .squeeze(0)
            .numpy()
        )
        return lkhd

    return model_update_


def gt_update(
    obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    use_objs: bool,
    game_state: GameState,
    agent_state: AgentState,
    size: int,
    cell_size: float,
    is_pursuer: bool,
) -> np.ndarray:
    player_pos = game_state.player.pos
    grid_pos = pos_to_grid(player_pos.x, player_pos.y, game_state.level_size, cell_size)
    lkhd = np.zeros([size, size])
    lkhd[grid_pos[1], grid_pos[0]] = 1
    kernel = np.array([[0.1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 0.1]])
    lkhd = signal.convolve2d(lkhd, kernel, mode="same")
    return lkhd


def replace_extra_channel(obs: Tuple[Tensor, Tensor, Tensor], channel: Tensor):
    """
    Replaces the extra channel.
    """
    zeroed = obs[0]
    zeroed[:, -1] = channel
    return (zeroed.numpy(), obs[1].numpy(), obs[2].numpy())


if __name__ == "__main__":
    from webgame.envs import GameEnv, CELL_SIZE, MAX_OBJS, OBJ_DIM
    import rerun as rr  # type: ignore
    import random
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pursuer-chkpt", type=str, default=None)
    parser.add_argument("--player-chkpt", type=str, default=None)
    parser.add_argument("--use-pos", action="store_true")
    parser.add_argument("--use-objs", action="store_true")
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--wall-prob", type=float, default=0.1)
    parser.add_argument("--lkhd-min", type=float, default=0.0)
    parser.add_argument("--insert-visible-cells", default=False, action="store_true")
    parser.add_argument("--start-gt", default=False, action="store_true")
    parser.add_argument("--use-pathfinding", default=False, action="store_true")
    parser.add_argument(
        "--player-sees-visible-cells", default=False, action="store_true"
    )
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    def heuristic_policy(
        agent: str, env: GameEnv, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> int:
        if random.random() < 0.1:
            action = env.action_space(agent).sample()
        else:
            assert env.game_state
            action = explore_policy(env.game_state, agent == "pursuer")
        return action

    def pathfinding_policy(
        agent: str, env: GameEnv, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> int:
        if args.player_sees_visible_cells:
            idx = -2
        else:
            idx = -1
        probs = obs[0].squeeze(0)[idx]
        max_idx = probs.flatten().argmax()
        assert env.game_state
        y = int(max_idx) // env.game_state.level_size
        x = int(max_idx) % env.game_state.level_size
        start_pos = (x, y)
        pursuer_pos = env.game_state.pursuer.pos
        target_pos = pos_to_grid(
            pursuer_pos.x, pursuer_pos.y, env.game_state.level_size, CELL_SIZE
        )

        queue = [start_pos]
        parents: Dict[tuple[int, int], tuple[int, int]] = {}
        while len(queue) > 0:
            curr_pos = queue.pop(0)
            neighbors_delta = [
                (1, 0),
                (1, -1),
                (1, 1),
                (-1, 0),
                (-1, -1),
                (-1, 1),
                (0, 1),
                (-1, 1),
                (1, 1),
                (0, -1),
                (-1, -1),
                (1, -1),
            ]
            finished = False
            for n_delta in neighbors_delta:
                neighbor = (curr_pos[0] + n_delta[0], curr_pos[1] + n_delta[1])

                def is_wall(pos: tuple[int, int]) -> bool:
                    assert env.game_state
                    if (
                        pos[0] < 0
                        or pos[0] >= env.game_state.level_size
                        or pos[1] < 0
                        or pos[1] >= env.game_state.level_size
                    ):
                        return True
                    return env.game_state.walls[
                        pos[1] * env.game_state.level_size + pos[0]
                    ]

                can_enter = not is_wall(neighbor)
                if n_delta[0] != 0 and n_delta[1] != 0:
                    n1 = (curr_pos[0] + n_delta[0], curr_pos[1])
                    n2 = (curr_pos[0], curr_pos[1] + n_delta[1])
                    can_enter = can_enter and (not is_wall(n1) or not is_wall(n2))
                if can_enter and neighbor not in parents.keys():
                    queue.append(neighbor)
                    parents[neighbor] = curr_pos
                    if neighbor == target_pos:
                        finished = True
                        break
            if finished:
                break

        if target_pos not in parents.keys():
            return env.action_space(agent).sample()

        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        actual_dir = np.array(parents[target_pos]) - np.array(target_pos)
        actual_dir = actual_dir / math.sqrt(float((actual_dir**2).sum()))
        best_action = 0
        best_score = -1
        for i, dir_ in enumerate(dirs):
            action = i + 1
            norm_dir = np.array(dir_)
            norm_dir = norm_dir / math.sqrt(float((norm_dir**2).sum()))
            score = norm_dir @ actual_dir.T
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def model_policy(
        chkpt_path: str,
        action_count: int,
        use_pos: bool,
        use_objs: bool,
    ) -> Callable[[str, GameEnv, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
        channels = 6
        if args.player_sees_visible_cells:
            channels = 7
        p_net = PolicyNet(
            channels,
            args.grid_size,
            action_count,
            use_pos,
            (MAX_OBJS, OBJ_DIM) if use_objs else None,
        )
        load_model(p_net, chkpt_path)

        def policy(
            agent: str,
            env: GameEnv,
            obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> int:
            action_probs = p_net(*obs).squeeze(0)
            action = Categorical(logits=action_probs).sample().item()
            return action

        return policy

    recording_id = "filter_test-" + str(random.randint(0, 10000))
    rr.init(application_id="Pursuer", recording_id=recording_id)
    rr.connect()

    env = GameEnv(
        wall_prob=args.wall_prob,
        use_objs=args.use_objs,
        visualize=True,
        recording_id=recording_id,
        player_sees_visible_cells=args.player_sees_visible_cells,
        update_fn=manual_update,
        grid_size=args.grid_size,
        start_gt=args.start_gt,
    )
    obs_ = env.reset()[0]
    obs = {agent: convert_obs(obs_[agent], True) for agent in env.agents}

    action_space = env.action_space("pursuer")  # Same for both agents
    assert isinstance(action_space, gym.spaces.Discrete)
    assert env.game_state is not None

    # Set up filter
    if args.checkpoint:
        model = MeasureModel(9, env.game_state.level_size, args.use_pos)
        load_model(model, args.checkpoint)
        update_fn = model_update(model)
    elif args.use_gt:
        update_fn = gt_update
    else:
        update_fn = manual_update
    b_filter = BayesFilter(
        env.game_state.level_size,
        CELL_SIZE,
        update_fn,
        args.use_objs,
        True,
        args.lkhd_min,
    )
    if args.start_gt:
        b_filter.belief = np.zeros(b_filter.belief.shape)
        play_pos = env.game_state.player.pos
        x, y = pos_to_grid(play_pos.x, play_pos.y, env.game_state.level_size, CELL_SIZE)
        b_filter.belief[y, x] = 1

    # Set up policies
    policies = {}
    chkpts = {"pursuer": args.pursuer_chkpt, "player": args.player_chkpt}
    for agent in env.agents:
        if chkpts[agent]:
            policies[agent] = model_policy(
                chkpts[agent], int(action_space.n), args.use_pos, args.use_objs
            )
        elif args.use_pathfinding and agent == "pursuer":
            policies[agent] = pathfinding_policy
        else:
            policies[agent] = heuristic_policy

    for _ in range(100):
        actions = {}
        for agent in env.agents:
            action = policies[agent](agent, env, obs[agent])
            actions[agent] = action
        obs_, rew, done, trunc, info = env.step(actions)
        # if done["pursuer"]:
        #     obs_ = env.reset()[0]
        #     b_filter = BayesFilter(
        #         env.game_state.level_size,
        #         CELL_SIZE,
        #         update_fn,
        #         args.use_objs,
        #         True,
        #         args.lkhd_min,
        #     )
        #     if args.start_gt:
        #         b_filter.belief = np.zeros(b_filter.belief.shape)
        #         play_pos = env.game_state.player.pos
        #         x, y = pos_to_grid(
        #             play_pos.x, play_pos.y, env.game_state.level_size, CELL_SIZE
        #         )
        #         b_filter.belief[y, x] = 1
        obs = {agent: convert_obs(obs_[agent], True) for agent in env.agents}

        game_state = env.game_state
        assert game_state is not None
        agent_state = game_state.pursuer
        extra_channel = torch.zeros(
            [game_state.level_size, game_state.level_size], dtype=torch.float
        )
        if args.insert_visible_cells:
            visible_cells = agent_state.visible_cells
            extra_channel = torch.tensor(visible_cells, dtype=torch.float).reshape(
                [game_state.level_size, game_state.level_size]
            )
        filter_obs = replace_extra_channel(obs["pursuer"], extra_channel)
        lkhd = update_fn(
            filter_obs,
            False,
            game_state,
            agent_state,
            game_state.level_size,
            CELL_SIZE,
            True,
        )
        manual_lkhd = manual_update(
            filter_obs,
            False,
            game_state,
            agent_state,
            game_state.level_size,
            CELL_SIZE,
            True,
        )
        probs = b_filter.localize(filter_obs, game_state, agent_state)
        rr.log("filter/belief", rr.Tensor(probs), timeless=False)
        rr.log("filter/measurement_likelihood", rr.Tensor(lkhd), timeless=False)
        rr.log(
            "filter/manual_measurement_likelihood",
            rr.Tensor(manual_lkhd),
            timeless=False,
        )
        rr.log("filter/filter_obs", rr.Tensor(filter_obs[0]), timeless=False)
        rr.log("agent/pursuer_obs", rr.Tensor(obs["pursuer"][0]), timeless=False)
        rr.log("agent/player_obs", rr.Tensor(obs["player"][0]), timeless=False)
