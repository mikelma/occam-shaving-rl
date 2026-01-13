import tyro
import itertools
from config import META_CONFIG
import pprint
import msgpack


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


def configs_to_bin(cfg_key: str = "hopper", out_path: str = "configs.bin"):
    cfgs = META_CONFIG[cfg_key]
    total_num = num_configurations(cfgs)

    print(f"Generating all {total_num} configurations...")

    all_configs = generate_all_configs(cfgs)

    input("> Press any key to continue...")

    bin_data = msgpack.packb(all_configs, use_bin_type=True)
    with open(out_path, "wb") as binary_file:
        binary_file.write(bin_data)


if __name__ == "__main__":
    # tyro.cli(generate_config)
    tyro.cli(configs_to_bin)
