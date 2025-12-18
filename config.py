import numpy as np


TF = [True, False]


META_CONFIG = {
    "brax": {
        # constants
        "ENV_NAME": ["walker2d", "ant", "humanoid"],
        "DEBUG": False,
        "LOG_DIR": "logs/",
        "NUM_PARALLEL_RUNS": 30,
        "VF_COEF": 0.5,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1e6,
        "USE_MUON": False,
        "GAMMA": 0.99,
        "SPLIT_AC": False,
        "HIDDEN_DIM": 64,
        "NUM_ENVS": 2048,
        "ACTIVATION": "tanh",
        # variables
        "LR": [3e-3, 3e-4, 3e-5],
        "ENT_COEF": [0.0, 0.01, 0.1],
        "UPDATE_EPOCHS": [1, 4],
        "NUM_MINIBATCHES": [16, 32, 64],
        "GAE_LAMBDA": [0.9, 0.95],
        "CLIP_EPS": [0.2, 1e6],  # on / off
        "CLIP_VALUE_EPS": [0.2, 1e6],  # on / off
        "MAX_GRAD_NORM": [0.5, 1e6],  # on / off
        "ANNEAL_LR": TF,
        "NORMALIZE_ENV": TF,
        "GAE_NORMALIZATION": TF,
        "LAYER_NORM": TF,
        "INITIALIZERS": [
            {
                "shared": ["orthogonal", np.sqrt(2)],
                "actor": ["orthogonal", 0.01],
                "critic": ["orthogonal", 1],
            },
            {
                "shared": ["orthogonal", np.sqrt(2)],
                "actor": ["orthogonal", 1],
                "critic": ["orthogonal", 1],
            },
            # {
            #     "shared": ["glorot_u"],
            #     "actor": ["glorot_u"],
            #     "critic": ["glorot_u"],
            # },
        ],
    },
    "atari": {
        # constants
        "env_id": ["SpaceInvaders-v5", "Asterix-v5", "Seaquest-v5"],
        "log_dir": "logs/",
        "seed": list(range(10)),
        "vf_coef": 0.5,
        "num_steps": 128,
        "total_timesteps": int(10_000_000),
        # "use_muon": False,
        "gamma": 0.99,
        # "SPLIT_AC": False,
        "hidden_dim": 512,
        "num_envs": 8,
        # "activation": "relu",  # NOTE use relu always
        # variables
        "learning_rate": [2.5e-3, 2.5e-4, 2.5e-5],
        "ent_coef": [0.0, 0.01, 0.1],
        "update_epochs": [1, 4],
        "num_minibatches": [4, 8],
        "gae_lambda": [0.9, 0.95],
        "clip_coef": [0.1, 1e6],
        # "clip_vloss": [0.2, 1e6],  # TODO the script does not implement this
        "max_grad_norm": [0.5, 1e6],  # on / off
        "anneal_lr": TF,
        # "NORMALIZE_ENV": TF, # just applying all the Atari tricks (see: https://envpool.readthedocs.io/en/latest/env/atari.html#env-wrappers)
        "norm_adv": TF,  # NOTE same as GAE normalization in brax
        "layer_norm": TF,
        # "INITIALIZERS": [
        #     {
        #         "shared": ["orthogonal", np.sqrt(2)],
        #         "actor": ["orthogonal", 0.01],
        #         "critic": ["orthogonal", 1],
        #     },
        #     {
        #         "shared": ["orthogonal", np.sqrt(2)],
        #         "actor": ["orthogonal", 1],
        #         "critic": ["orthogonal", 1],
        #     },
        #     # {
        #     #     "shared": ["glorot_u"],
        #     #     "actor": ["glorot_u"],
        #     #     "critic": ["glorot_u"],
        #     # },
        # ],
    },
}
