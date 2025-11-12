from flax.linen.initializers import glorot_uniform, orthogonal
import tyro
import itertools
from config import META_CONFIG
import pprint


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


def num_configurations(config):
    num = 1
    for param, values in config.items():
        if isinstance(values, list):
            num *= len(values)
    return num


def generate_all_configs(conf):
    const_items = {k: v for k, v in conf.items() if not isinstance(v, list)}
    var_items = [(k, v) for k, v in conf.items() if isinstance(v, list)]

    var_keys = [k for k, _ in var_items]
    var_values = [v for _, v in var_items]

    combos = itertools.product(*var_values)

    configs = [{**const_items, **dict(zip(var_keys, combo))} for combo in combos]

    return configs


def generate_config(cfg_key: str = "hopper", id: int = 0, verbose: bool = True):
    cfgs = META_CONFIG[cfg_key]
    total_num = num_configurations(cfgs)

    if verbose:
        print(f"Generating all {total_num} configurations...")

    all_configs = generate_all_configs(cfgs)

    cfg = all_configs[id]
    
    if verbose:
        print()
        pprint.pprint(cfg)

    return cfg

if __name__ == "__main__":
    tyro.cli(generate_config)
