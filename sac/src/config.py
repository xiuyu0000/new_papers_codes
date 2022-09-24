# Copyright 2022
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
SAC config.
"""

import mindspore
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from src.sac import SACActor, SACLearner, SACPolicy

env_params = {'name': 'HalfCheetah-v2'}
eval_env_params = {'name': 'HalfCheetah-v2'}

policy_params = {
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 256,
    'hidden_size2': 256,
}

learner_params = {
    'gamma': 0.99,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'epsilon': 0.2,
    'critic_lr': 3e-4,
    'actor_lr': 3e-4,
    'alpha_lr': 3e-4,
    'reward_scale_factor': 0.1,
    'critic_loss_weight': 0.5,
    'actor_loss_weight': 1.0,
    'alpha_loss_weight': 1.0,
    'init_alpha': 0.,
    'target_entropy': -3.0,
    'update_factor': 0.005,
    'update_interval': 1,
}

trainer_params = {
    'duration': 1000,
    'batch_size': 256,
    'ckpt_path': './ckpt',
    'num_eval_episode': 30,
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': SACActor,
        'policies': ['init_policy', 'collect_policy', 'eval_policy'],
    },
    'learner': {
        'number': 1,
        'type': SACLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net1', 'critic_net2', 'target_critic_net1', 'target_critic_net2']
    },
    'policy_and_network': {
        'type': SACPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'number': 1,
        'type': UniformReplayBuffer,
        'capacity': 1000000,
        'data_shape': [(17,), (6,), (1,), (17,)],
        'data_type': [
            mindspore.float32, mindspore.float32, mindspore.float32, mindspore.float32
        ],
        'sample_size': 256
    }
}
