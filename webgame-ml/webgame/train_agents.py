from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
import os
from pathlib import Path
from typing import *

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from torch.distributions import Categorical
from tqdm import tqdm
from safetensors.torch import save_model, load_model

from webgame.algorithms.parallel_vec_wrapper import ParallelVecWrapper
from webgame.algorithms.ppo import train_ppo
from webgame.algorithms.rollout_buffer import RolloutBuffer
from webgame.common import convert_obs, explore_policy, process_obs
from webgame.conf import entity
from webgame.envs import MAX_OBJS, OBJ_DIM, GameEnv
from webgame.filter import gt_update, manual_update, model_update
from webgame.models import Backbone, MeasureModel, PolicyNet

_: Any

torch.tensor([1], device="cuda")


@dataclass
class Config:
    out_dir: str = "./runs"  # Output directory.
    num_envs: int = (
        64  # Number of environments to step through at once during sampling.
    )
    train_steps: int = (
        32  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
    )
    iterations: int = 1000  # Number of sample/train iterations.
    train_iters: int = 2  # Number of passes over the samples collected.
    train_batch_size: int = 512  # Minibatch size while training models.
    discount: float = 0.98  # Discount factor applied to rewards.
    lambda_: float = 0.95  # Lambda for GAE.
    epsilon: float = 0.2  # Epsilon for importance sample clipping.
    max_eval_steps: int = 500  # Max number of steps to take during each eval run.
    eval_steps: int = 32  # Number of eval runs to average over.
    v_lr: float = 0.01  # Learning rate of the value net.
    p_lr: float = 0.001  # Learning rate of the policy net.
    use_objs: bool = False  # Whether we should use objects in the simulation.
    use_pos: bool = False  # Whether we use a position encoding.
    max_timer: int = 100  # Maximum length of an episode.
    save_every: int = 10  # How many iterations to wait before saving.
    eval_every: int = 2  # How many iterations before evaluating.
    wall_prob: float = 0.1  # Probability of a cell containing a wall.
    entropy_coeff: float = 0.001  # Entropy bonus applied.
    gradient_clip: float = 0.1  # Gradient clipping for networks.
    gradient_steps: int = (
        1  # Number of gradient steps, effectively increases the batch size.
    )
    update_fn: str = (
        "gt"  # The filter's update function. Valid choices: manual, model, gt
    )
    update_chkpt: str = ""  # Checkpoint to use for filter.
    player_sees_visible_cells: bool = (
        False  # Whether the player should have ground truth information on the pursuer.
    )
    checkpoint_player: str = ""  # Player checkpoint to continue from.
    checkpoint_pursuer: str = ""  # Pursuer checkpoint to continue from.
    aux_rew_amount: float = 0.0
    grid_size: int = 8
    start_gt: bool = False
    device: str = "cuda"  # Device to use during training.


class ValueNet(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        proj_dim = 64
        self.backbone = Backbone(channels, proj_dim, size, use_pos, objs_shape)
        self.net1 = nn.Sequential(
            nn.Conv2d(proj_dim, 64, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        features = self.backbone(grid, objs, objs_attn_mask)
        values = self.net1(features)
        values = values.amax(-1).amax(-1)
        values = self.net2(values)
        return values


class AgentData:
    def __init__(
        self,
        channels: int,
        grid_size: int,
        max_objs: int,
        obj_dim: int,
        cfg: Config,
        act_count: int,
    ):
        self.v_net = ValueNet(
            channels * 2,
            grid_size,
            cfg.use_pos,
            (max_objs * 2, obj_dim) if cfg.use_objs else None,
        )
        self.p_net = PolicyNet(
            channels,
            grid_size,
            act_count,
            cfg.use_pos,
            (max_objs, obj_dim) if cfg.use_objs else None,
        )
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=cfg.v_lr)
        self.p_opt = torch.optim.Adam(self.p_net.parameters(), lr=cfg.p_lr)
        self.buffer = RolloutBuffer(
            [
                (torch.Size((channels, grid_size, grid_size)), torch.float),
                (torch.Size((max_objs, obj_dim)), torch.float),
                (torch.Size((max_objs,)), torch.bool),
            ],
            torch.Size((1,)),
            torch.Size((act_count,)),
            torch.int,
            cfg.num_envs,
            cfg.train_steps,
        )


if __name__ == "__main__":
    cfg = Config()
    parser = ArgumentParser()
    for k, v in cfg.__dict__.items():
        if isinstance(v, bool):
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v, action="store_true"
            )
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    args = parser.parse_args()
    cfg = Config(**args.__dict__)
    device = torch.device(cfg.device)

    assert cfg.update_fn in ["manual", "model", "gt"]
    if cfg.update_chkpt:
        assert cfg.update_fn == "model"
    if cfg.update_fn == "model":
        m_model = MeasureModel(9, 8, cfg.use_pos)
        load_model(m_model, cfg.update_chkpt)
        update_fn = model_update(m_model)
    elif cfg.update_fn == "manual":
        update_fn = manual_update
    else:
        update_fn = gt_update

    wandb_cfg = {"experiment": "agents"}
    wandb_cfg.update(cfg.__dict__)
    wandb.init(
        project="pursuer",
        entity=entity,
        config=wandb_cfg,
    )

    assert wandb.run is not None
    for _ in range(100):
        if wandb.run.name != "":
            break
    if wandb.run.name != "":
        out_id = wandb.run.name
    else:
        out_id = "testing"

    out_dir = Path(cfg.out_dir)
    try:
        os.mkdir(out_dir / out_id)
    except OSError as e:
        print(e)
    chkpt_path = out_dir / out_id / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)

    env = ParallelVecWrapper(
        [
            lambda: GameEnv(
                cfg.use_objs,
                cfg.wall_prob,
                grid_size=cfg.grid_size,
                max_timer=cfg.max_timer,
                player_sees_visible_cells=cfg.player_sees_visible_cells,
                aux_rew_amount=cfg.aux_rew_amount,
                update_fn=update_fn,
                start_gt=cfg.start_gt,
            )
            for _ in range(cfg.num_envs)
        ]
    )
    test_env = GameEnv(
        cfg.use_objs,
        cfg.wall_prob,
        grid_size=cfg.grid_size,
        max_timer=cfg.max_timer,
        player_sees_visible_cells=cfg.player_sees_visible_cells,
        update_fn=update_fn,
        start_gt=cfg.start_gt,
    )

    # Initialize policy and value networks
    channels = 6
    if cfg.player_sees_visible_cells:
        channels = 7
    grid_size = cfg.grid_size
    max_objs = MAX_OBJS
    obj_dim = OBJ_DIM
    act_space = env.action_space(env.agents[0])
    assert isinstance(act_space, gym.spaces.Discrete)
    agents = {
        agent: AgentData(channels, grid_size, max_objs, obj_dim, cfg, int(act_space.n))
        for agent in env.agents
    }
    if cfg.checkpoint_pursuer != "":
        load_model(agents["pursuer"].p_net, cfg.checkpoint_pursuer)
    if cfg.checkpoint_player != "":
        load_model(agents["player"].p_net, cfg.checkpoint_player)

    obs_: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
        env.reset()[0]
    )
    obs = {agent: convert_obs(obs_[agent]) for agent in env.agents}
    for step in tqdm(range(cfg.iterations), position=0):
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in tqdm(range(cfg.train_steps), position=1):
                all_action_probs = {}
                all_actions = {}
                for agent in env.agents:
                    action_probs = agents[agent].p_net(*obs[agent])
                    actions = Categorical(logits=action_probs).sample().numpy()
                    all_action_probs[agent] = action_probs
                    all_actions[agent] = actions
                obs_, rewards, dones, truncs, _ = env.step(all_actions)
                for agent in env.agents:
                    agents[agent].buffer.insert_step(
                        list(obs[agent]),
                        torch.from_numpy(all_actions[agent]).unsqueeze(-1),
                        all_action_probs[agent],
                        rewards[agent],
                        dones[agent],
                        truncs[agent],
                    )
                obs = {agent: convert_obs(obs_[agent]) for agent in env.agents}
            for agent in env.agents:
                agents[agent].buffer.insert_final_step(list(obs[agent]))

        # Train
        log_dict = {}
        for agent in env.agents:
            other_agent = {
                "player": "pursuer",
                "pursuer": "player",
            }[agent]
            total_p_loss, total_v_loss = train_ppo(
                agents[agent].p_net,
                agents[agent].v_net,
                agents[agent].p_opt,
                agents[agent].v_opt,
                agents[agent].buffer,
                agents[other_agent].buffer,
                device,
                cfg.train_iters,
                cfg.train_batch_size,
                cfg.discount,
                cfg.lambda_,
                cfg.epsilon,
                entropy_coeff=cfg.entropy_coeff,
                gradient_clip=cfg.gradient_clip,
                gradient_steps=cfg.gradient_steps,
            )
            agents[agent].buffer.clear()
            log_dict[f"{agent}_avg_v_loss"] = total_v_loss / cfg.train_iters
            log_dict[f"{agent}_avg_p_loss"] = total_p_loss / cfg.train_iters

        # Evaluate agents
        if step % cfg.eval_every == 0:
            with torch.no_grad():
                for use_explore in [False, True]:
                    reward_total = {agent: 0.0 for agent in env.agents}
                    entropy_total = {agent: 0.0 for agent in env.agents}
                    for _ in range(cfg.eval_steps):
                        avg_entropy = {agent: 0.0 for agent in env.agents}
                        steps_taken = 0
                        obs_ = test_env.reset()[0]
                        eval_obs = {
                            agent: convert_obs(obs_[agent], True)
                            for agent in env.agents
                        }
                        for _ in range(cfg.max_eval_steps):
                            all_actions = {}
                            all_distrs = {}
                            for agent in env.agents:
                                distr = Categorical(
                                    logits=agents[agent]
                                    .p_net(*eval_obs[agent])
                                    .squeeze()
                                )
                                all_distrs[agent] = distr
                                action = distr.sample().item()
                                all_actions[agent] = action
                            if use_explore:
                                assert test_env.game_state
                                all_actions["player"] = explore_policy(
                                    test_env.game_state, False
                                )
                            obs_, reward, eval_done, _, _ = test_env.step(all_actions)
                            eval_obs = {
                                agent: convert_obs(obs_[agent], True)
                                for agent in env.agents
                            }
                            steps_taken += 1
                            for agent in env.agents:
                                reward_total[agent] += reward[agent]
                                avg_entropy[agent] += all_distrs[agent].entropy()
                            if eval_done:
                                break
                        for agent in env.agents:
                            avg_entropy[agent] /= steps_taken
                            entropy_total[agent] += avg_entropy[agent]
                    prefix = "baseline_" if use_explore else ""
                    for agent in env.agents:
                        log_dict.update(
                            {
                                f"{prefix}{agent}_avg_eval_episode_return": reward_total[
                                    agent
                                ]
                                / cfg.eval_steps,
                                f"{prefix}{agent}_avg_eval_entropy": entropy_total[
                                    agent
                                ]
                                / cfg.eval_steps,
                            }
                        )

        wandb.log(log_dict)

        if step % cfg.save_every == 0:
            for agent in env.agents:
                save_model(
                    agents[agent].p_net,
                    str(chkpt_path / f"{agent}-p_net-{step}.safetensors"),
                )
                save_model(
                    agents[agent].v_net,
                    str(chkpt_path / f"{agent}-v_net-{step}.safetensors"),
                )
