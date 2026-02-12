import jax
import jax.numpy as jnp
from brax.envs.base import State, Wrapper

class RandomizedAutoResetWrapper(Wrapper):
    """
    Automatically resets Brax envs that are done. and mask the goal observation.
    Force resample every step.
    """
    def reset(self, rng: jnp.ndarray) -> State:
        rng, _rng = jax.random.split(rng)
        state = self.env.reset(_rng)
        state.info['rng'] = rng
        state.info['pipeline_state'] = state.pipeline_state
        state.info['obs'] = state.obs
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
            state = state.replace(done=jnp.zeros_like(state.done))
            state = self.env.step(state, action)
            maybe_reset = self.reset(state.info['rng'])
            def where_done(x, y):
                done = state.done
                if done.shape:
                    done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1)) # type: ignore
                return jnp.where(done, x, y)
            pipeline_state = jax.tree_util.tree_map(
            where_done, maybe_reset.info['pipeline_state'], state.pipeline_state        )
            obs = where_done(maybe_reset.info['obs'], state.obs)
            return state.replace(pipeline_state=pipeline_state, obs=obs)
