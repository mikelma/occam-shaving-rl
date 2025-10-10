# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from muon import SingleDeviceMuonWithAuxAdam

WeightInitializations = Literal["cleanrl", "he",
                                "xavier", "xavier_normal", "kaiming", "kaiming_normal"]


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "baselines"
    """the wandb's project name"""
    wandb_entity: str = "occam-shaving-rl"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    hidden_size: int = 64
    """Hidden dimension of actor and critic networks"""
    muon: bool = False
    """whether to use muon for the hidden layers or not"""
    shared_network: bool = False
    """whether to use shard networks or not"""
    grad_norm: bool = True
    """whether to do gradient normalization"""
    symlog: bool = False
    """whether to use symlog for reward transform (clipping) or not"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    weight_initialization: WeightInitializations = field(default="cleanrl")
    """the type of weight initialization used to initialize the weights of the network"""


def make_env(env_id, idx, capture_video, run_name, gamma, symlog):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # deal with dm_control's Dict observation space
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if symlog:
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.sign(reward) * np.log1p(np.abs(reward)))  # symlog transform
        else:
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init_xavier(layer, gain=1.0, bias_const=0.0, **kwargs):
    torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_xavier_normal(layer, gain=1.0, bias_const=0.0, **kwargs):
    torch.nn.init.xavier_normal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_kaiming(layer, mode="fan_in", nonlinearity="tanh", bias_const=0.0, **kwargs):
    torch.nn.init.kaiming_uniform_(
        layer.weight, mode=mode, nonlinearity=nonlinearity)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_kaiming_normal(layer, mode="fan_in", nonlinearity="tanh", bias_const=0.0, **kwargs):
    torch.nn.init.kaiming_normal_(
        layer.weight, mode=mode, nonlinearity=nonlinearity)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_he(layer, mode="fan_in", bias_const=0.0, **kwargs):
    torch.nn.init.kaiming_uniform_(
        layer.weight, mode=mode, nonlinearity="tanh")
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_cleanrl(layer, std=np.sqrt(2), bias_const=0.0, **kwargs):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module, ABC):
    @abstractmethod
    def get_value(self, x):
        pass

    @abstractmethod
    def get_action_and_value(self, x, action=None):
        pass


class Agent(BaseAgent):

    def __init__(self, envs, hdim, layer_init=layer_init_cleanrl):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), hdim)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hdim, hdim)),
            nn.Tanh(),
            layer_init(nn.Linear(hdim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), hdim)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hdim, hdim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hdim, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class SharedAgent(BaseAgent):

    def __init__(self, envs, hdim, layer_init=layer_init_cleanrl):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), hdim)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hdim, hdim)),
            nn.Tanh(),
        )
        self.critic_head = layer_init(nn.Linear(hdim, 1), std=1.0)
        self.actor_head_mean = layer_init(
            nn.Linear(hdim, np.prod(envs.single_action_space.shape)), std=0.01
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        h = self.backbone(x)
        return self.critic_head(h)

    def get_action_and_value(self, x, action=None):
        h = self.backbone(x)
        action_mean = self.actor_head_mean(h)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        values = self.get_value(x)
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            values,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video,
                     run_name, args.gamma, args.symlog)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    layer_inits: dict[WeightInitializations, Callable] = {
        "cleanrl": layer_init_cleanrl,
        "he": layer_init_he,
        "xavier": layer_init_xavier,
        "xavier_normal": layer_init_xavier_normal,
        "kaiming": layer_init_kaiming,
        "kaiming_normal": layer_init_kaiming_normal,
    }

    # weight initialization
    layer_init = layer_inits[args.weight_initialization]

    if args.shared_network:
        agent = SharedAgent(envs, args.hidden_size,
                            layer_init=layer_init).to(device)
    else:
        agent = Agent(envs, args.hidden_size, layer_init=layer_init).to(device)

    if args.muon:
        hidden_weights, other_params = [], []
        for param in agent.parameters():
            if len(param.shape) >= 2 and param.shape[0] == args.hidden_size \
                    and param.shape[1] == args.hidden_size:
                hidden_weights.append(param)
            else:
                other_params.append(param)

        param_groups = [
            dict(params=hidden_weights,
                 use_muon=True,
                 lr=0.02,
                 # weight_decay=0.01
                 ),
            dict(
                params=other_params,
                use_muon=False,
                lr=args.learning_rate,
                # betas=(0.9, 0.95),
                # weight_decay=0.01,
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        num_hparams = sum(p.numel() for p in hidden_weights)
        num_other = sum(p.numel() for p in other_params)
        print(
            f"[*] Using Muon! ({round(100*num_hparams/(num_other+num_hparams), 3)}% of params)")
    else:
        optimizer = optim.Adam(
            agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            if args.muon:
                for i, opt in enumerate(optimizer.param_groups):
                    frac = 1.0 - (iteration - 1.0) / args.num_iterations
                    lrnow = frac * param_groups[i]["lr"]
                    optimizer.param_groups[0]["lr"] = lrnow

            else:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues *
                    nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.grad_norm:
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() -
                              start_time)), global_step
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
