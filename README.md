# Occam shaving RL 🪒


## The benchmarks

For continuous control:

- Walker2D
- Ant
- Humanoid

For discrete environments (from ALE's evaluation environments):

- Asterix 
- Seaquest
- Space Invaders

We use 1M timesteps for all continuous control environments and 10M for Atari games. -Finally, we run 30 seeds per PPO configuration in each environment


## Getting started

Just install [uv](https://docs.astral.sh/uv/) and run the following commands in the repo's directory.
```bash
uv sync
uv run <name-of-the-script>
```


## Plotting

```bash
uv run plot.py data/baseline.csv "Baseline" --output docs/img/baseline.svg
```

```bash
# First order changes
uv run plot.py data/baseline.csv "Baseline" data/full_batch.csv "Full-batch" data/grad-norm-False.csv "No gradient normalization" data/muon.csv "Muon optimizer" data/no_anneal_lr.csv "No LR scheduler" data/norm-adv-False.csv "No advantage normalization" data/policy_narrow_std.csv "Narrow policy std init" data/remove-value-clip.csv "No value clipping" data/reward_scaling_symlog.csv "Symlog reward scaling" data/shared_network.csv "Shared (backbone) network" --output docs/img/first-order.svg

# Combos
uv run plot.py data/baseline.csv "Baseline" data/combo1.csv "Combo 1" data/combo3.csv "Combo 3" data/combo4.csv "Combo 4" data/combo5.csv "Combo 5" data/combo6.csv "Combo 6" data/combo8.csv "Combo 8" --output docs/img/combos.svg

# Ant runs
uv run plot.py data/ant_baseline.csv "Baseline" data/ant_combo8.csv "Combo 8" data/ant_muon.csv "Muon" data/ant_cocombowowombo.csv "Cocombo Wowombo" --output docs/img/ant.svg --useglobalstep

# Humanoid runs
uv run plot.py data/humanoid_baseline.csv "Baseline" data/humanoid_muon.csv "Muon" data/humanoid_cocombo.csv "Cocombo" --output docs/img/humanoid.svg --useglobalstep
```
