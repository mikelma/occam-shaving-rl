# Occam shaving RL 🪒


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
uv run plot.py data/baseline.csv "Baseline" data/full_batch.csv "Full-batch" data/grad-norm-False.csv "No gradient normalization" data/muon.csv "Muon optimizer" data/no_anneal_lr.csv "No LR scheduler" data/norm-adv-False.csv "No advantage normalization" data/policy_narrow_std.csv "Narrow policy std init" data/remove-value-clip.csv "No value clipping" data/reward_scaling_symlog.csv "Symlog reward scaling" data/shared_network.csv "Shared (backbone) network" --output docs/img/first-order.svg
```