import numpy as np


TF = [True, False]


META_CONFIG = {
    "hopper": {
        # constants
        "ENV_NAME": "hopper",
        "DEBUG": True,
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
    }
}
