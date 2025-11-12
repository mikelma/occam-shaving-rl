import numpy as np


TF = [True, False]


META_CONFIG = {
    "hopper": {
        # constants
        "ENV_NAME": "hopper",
        "DEBUG": False,
        "NUM_PARALLEL_RUNS": 30,
        "VF_COEF": 0.5,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "USE_MUON": False,

        # variables
        "LR": [1e-4, 3e-4, 1e-5],
        "NUM_ENVS": [512, 1024, 2048],
        "UPDATE_EPOCHS": [1, 4],
        "NUM_MINIBATCHES": [8, 16, 32, 64],
        "GAMMA": [0.9, 0.95, 0.99],
        "GAE_LAMBDA": [0.95, 1],  # on / off
        "CLIP_EPS": [0.2, 1e6],  # on / off
        "CLIP_VALUE_EPS": [0.2, 1e6],  # on / off
        "ENT_COEF": [0.0, 0.1],  # on / of
        "MAX_GRAD_NORM": [0.5, 1e6],  # on / off
        "ACTIVATION": ["tanh", "relu"],
        "ANNEAL_LR": TF,
        "NORMALIZE_ENV": TF,
        "GAE_NORMALIZATION": TF,
        "SPLIT_AC": TF,
        "HIDDEN_DIM": [128, 256, 512],
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
    }
}
