"""Small training inputs shared by integration tests."""


def tiny_training_params():
    return {
        "experiment": {
            "on_policy_collected_frames_per_batch": 8,
            "on_policy_n_envs_per_worker": 2,
            "on_policy_n_minibatch_iters": 1,
            "on_policy_minibatch_size": 4,
        },
        "model": {"num_cells": [8]},
    }
