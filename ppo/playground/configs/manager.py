import json
import importlib
import gym
import os
import sys
from gym.wrappers import Monitor
# from playground.utils.misc import plot_from_monitor_results


def load_policy_class(policy_name):
    print("policy_name", policy_name)
    mod = importlib.import_module("playground.policies")
    policy_class = getattr(mod, policy_name)
    return policy_class


def load_wrapper_class(wrapper_name):
    mod = importlib.import_module("playground.utils.wrappers")
    wrapper_class = getattr(mod, wrapper_name)
    return wrapper_class


def apply_wrappers(env, list_of_wrappers):
    for name, params in list_of_wrappers:
        wrapper_class = load_wrapper_class(name)
        env = wrapper_class(env, **params)
    return env


class ConfigManager:
    def __init__(self, env_name, policy_name, policy_params=None, train_params=None,
                 wrappers=None):
        self.env_name = env_name
        self.policy_name = policy_name
        self.policy_params = policy_params or {}
        self.train_params = train_params or {}
        self.wrappers = wrappers or []

        self.env = gym.make(self.env_name)
        self.env = apply_wrappers(self.env, self.wrappers)

    def to_json(self):
        return dict(
            env_name=self.env_name,
            wrappers=self.wrappers,
            policy_name=self.policy_name,
            policy_params=self.policy_params,
            train_params=self.train_params,
        )

    @classmethod
    def load(cls, file_path):
        assert os.path.exists(file_path)
        return cls(**json.load(open(file_path)))

    def save(self, file_path):
        with open(file_path, 'w') as fin:
            json.dump(self.to_json(), fin, indent=4, sort_keys=True)

    def start_training(self, model_name):
        self.env.reset()
        print ("model_name", model_name)
        env = Monitor(self.env, '/tmp/' + model_name, force=True)
        policy = load_policy_class(self.policy_name)(
            env, model_name, training=True, **self.policy_params)

        print("\n==================================================")
        print("Loaded gym.env:", self.env_name)
        print("Wrappers:", self.wrappers)
        print("Loaded policy:", policy.__class__)
        print("Policy params:", self.policy_params)
        print("Train params:", self.train_params)
        print("==================================================\n")


        policy.build()
        train_config = policy.TrainConfig(**self.train_params)
        policy.train(train_config)

        env.close()
        # plot_from_monitor_results('/tmp/' + model_name, window=50)
        print("Training completed:", model_name)
