import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal, glorot_uniform
from typing import Sequence, NamedTuple, Callable
from flax.training.train_state import TrainState
import distrax
import tyro
import copy
import os
from dataclasses import dataclass
import msgpack

from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


@dataclass
class Args:
    id: int = 0
    """Experiment's ID (and seed)"""

    confs: str = "brax_confs.bin"
    """Meta configuration file path"""


def make_initializers(specification):
    inits = {}
    for layer, method_lst in specification.items():
        method = method_lst[0]
        if method == "orthogonal":
            inits[layer] = orthogonal(method_lst[1])
        elif method == "glorot_u":
            inits[layer] = glorot_uniform()
        else:
            raise NotImplementedError(f"Weight initializer '{method}' not implemented")
    return inits


class SplitActorCritic(nn.Module):
    initializers: dict[str, Callable]
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 256
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        if self.layer_norm:
            actor_mean = nn.LayerNorm()(actor_mean)

        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        if self.layer_norm:
            actor_mean = nn.LayerNorm()(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=self.initializers["actor"],
            bias_init=constant(0.0),
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        if self.layer_norm:
            critic = nn.LayerNorm()(critic)

        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        if self.layer_norm:
            critic = nn.LayerNorm()(critic)

        critic = nn.Dense(
            1, kernel_init=self.initializers["critic"], bias_init=constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class CombinedActorCritic(nn.Module):
    initializers: dict[str, Callable]
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 256
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        shared_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(x)
        shared_mean = activation(shared_mean)

        if self.layer_norm:
            shared_mean = nn.LayerNorm()(shared_mean)

        shared_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializers["shared"],
            bias_init=constant(0.0),
        )(shared_mean)
        shared_mean = activation(shared_mean)

        if self.layer_norm:
            shared_mean = nn.LayerNorm()(shared_mean)

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=self.initializers["actor"],
            bias_init=constant(0.0),
        )(shared_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            1, kernel_init=self.initializers["critic"], bias_init=constant(0.0)
        )(shared_mean)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # If you want randomized resetting
    # env = RandomizedAutoResetWrapper(env)

    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if config["SPLIT_AC"]:
            network = SplitActorCritic(
                config["INITIALIZERS"],
                env.action_space(env_params).shape[0],
                activation=config["ACTIVATION"],
                hidden_dim=config["HIDDEN_DIM"],
                layer_norm=config["LAYER_NORM"],
            )
        else:
            network = CombinedActorCritic(
                config["INITIALIZERS"],
                env.action_space(env_params).shape[0],
                activation=config["ACTIVATION"],
                hidden_dim=config["HIDDEN_DIM"],
                layer_norm=config["LAYER_NORM"],
            )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["USE_MUON"]:
            optimizer = optax.contrib.muon
        else:
            optimizer = optax.adam
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optimizer(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optimizer(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_VALUE_EPS"], config["CLIP_VALUE_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        if config["GAE_NORMALIZATION"]:
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], (
                    "batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    args = tyro.cli(Args)

    f = open(args.confs, "rb")
    bin = f.read()
    f.close()
    configs = msgpack.unpackb(bin, raw=False)

    config_readable = configs[args.id]

    print(config_readable)

    config = copy.deepcopy(config_readable)
    config["INITIALIZERS"] = make_initializers(config_readable["INITIALIZERS"])

    rng = jax.random.PRNGKey(args.id)
    keys = jax.random.split(rng, num=config["NUM_PARALLEL_RUNS"])
    train_jit_vmap = jax.jit(jax.vmap(make_train(config)))
    out = jax.block_until_ready(train_jit_vmap(keys))

    # dims: (num_parallel_seeds, num_updates, num_steps_in_batch, num_parallel_envs)
    ep_rets = out["metrics"]["returned_episode_returns"]
    ep_lens = out["metrics"]["returned_episode_lengths"]

    ep_rets = ep_rets.mean(-1)  # average results from parallel envs.
    ep_lens = ep_lens.mean(-1)
    ep_rets = ep_rets.reshape(
        ep_rets.shape[0], -1
    )  # flatten array to shape: (num_seeds, num_updates*num_steps)
    ep_lens = ep_lens.reshape(ep_lens.shape[0], -1)

    # global_steps = jnp.arange(0, config["TOTAL_TIMESTEPS"], step=config["NUM_ENVS"])
    global_steps = config["NUM_ENVS"] * jnp.arange(1, ep_lens.shape[1] + 1)

    runs_dir = f"{config['LOG_DIR']}/brax_{args.id}"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    with open(f"{runs_dir}/brax_{args.id}.csv", "w") as f:
        # write header
        f.write("seed,step,episodic_return,episode_len\n")
        for run_id in range(config["NUM_PARALLEL_RUNS"]):
            # the rest of rows
            for i in range(global_steps.shape[0]):
                f.write(
                    f"{run_id},{global_steps[i]},{ep_rets[run_id][i]},{ep_lens[run_id][i]}\n"
                )
