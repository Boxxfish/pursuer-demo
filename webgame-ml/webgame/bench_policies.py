import math
from typing import *
import numpy as np
from scipy import signal  # type: ignore

import torch
from torch.distributions import Categorical
from safetensors.torch import load_model
from tqdm import tqdm
from webgame_rust import AgentState, GameState

from webgame.common import convert_obs, explore_policy, pos_to_grid, process_obs

import gymnasium as gym
from torch import Tensor, nn
from webgame.filter import BayesFilter, manual_update, replace_extra_channel
from webgame.models import MeasureModel, PolicyNet
from webgame.envs import GameEnv, CELL_SIZE, MAX_OBJS, OBJ_DIM
import random
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--pursuer-chkpt", type=str, default=None)
    parser.add_argument("--use-pos", action="store_true")
    parser.add_argument("--use-objs", action="store_true")
    parser.add_argument("--wall-prob", type=float, default=0.1)
    parser.add_argument("--lkhd-min", type=float, default=0.0)
    parser.add_argument("--insert-visible-cells", default=False, action="store_true")
    parser.add_argument("--start-gt", default=False, action="store_true")
    parser.add_argument("--use-pathfinding", default=False, action="store_true")
    parser.add_argument(
        "--player-sees-visible-cells", default=False, action="store_true"
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-iters", type=int, default=1000)
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

    env = GameEnv(
        wall_prob=args.wall_prob,
        use_objs=args.use_objs,
        player_sees_visible_cells=args.player_sees_visible_cells,
        update_fn=manual_update,
        grid_size=args.grid_size,
        start_gt=args.start_gt,
        max_timer=100,
    )
    obs_ = env.reset()[0]
    obs = {agent: convert_obs(obs_[agent], True) for agent in env.agents}

    action_space = env.action_space("pursuer")  # Same for both agents
    assert isinstance(action_space, gym.spaces.Discrete)
    assert env.game_state is not None

    # Set up filter
    update_fn = manual_update

    # Set up policies
    policies = {}
    chkpts = {"pursuer": args.pursuer_chkpt, "player": None}
    for agent in env.agents:
        if chkpts[agent]:
            policies[agent] = model_policy(
                chkpts[agent], int(action_space.n), args.use_pos, args.use_objs
            )
        elif args.use_pathfinding and agent == "pursuer":
            policies[agent] = pathfinding_policy
        else:
            policies[agent] = heuristic_policy

    found_count = 0
    for _ in tqdm(range(args.num_iters)):
        actions = {}
        for agent in env.agents:
            action = policies[agent](agent, env, obs[agent])
            actions[agent] = action
        obs_, rew, done, trunc, info = env.step(actions)
        if done["pursuer"] or trunc["pursuer"]:
            if done["pursuer"]:
                found_count += 1
            obs_ = env.reset()[0]
        obs = {agent: convert_obs(obs_[agent], True) for agent in env.agents}
    print("Found pct:", found_count / args.num_iters)
