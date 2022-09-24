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
MAPPO config.
"""

from example.mappo.src.mappo import MAPPOAgent, MAPPOActor, MAPPOLearner, MAPPOPolicy
from example.mappo.src.mappo_replaybuffer import MAPPOReplayBuffer
from mindspore_rl.environment.mpe_environment import MPEMultiEnvironment
import mindspore as ms

env_params = {'name': 'simple_spread', 'proc_num': 32, 'num': 128}
eval_env_params = {'name': 'simple_spread', 'proc_num': 1, 'num': 1}

policy_params = {
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size': 64,
}

learner_params = {
    'learning_rate': 0.0007,
    'gamma': 0.99,
    'td_lambda': 0.95,
    'iter_time': 10,
}

trainer_params = {
    'duration': 25,
    'eval_interval': 20,
    'metrics': False,
    'ckpt_path': './ckpt',
}

NUM_AGENT = 3
algorithm_config = {

    'agent': {
        'number': NUM_AGENT,
        'type': MAPPOAgent,
    },

    'actor': {
        'number': 1,
        'type': MAPPOActor,
        'policies': ['collect_policy'],
        'networks': ['critic_net'],
    },
    'learner': {
        'number': 1,
        'type': MAPPOLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net']
    },
    'policy_and_network': {
        'type': MAPPOPolicy,
        'params': policy_params
    },

    'replay_buffer': {
        "multi_type_replaybuffer": True,
        'local_replaybuffer': {
            'number': NUM_AGENT,
            'type': MAPPOReplayBuffer,
            'capacity': 26,
            'data_shape': [(128, NUM_AGENT * 6), (128, 1, 64), (128, 1, 64),
                           (128, 1), (128, 1), (128, 1), (128, 1), (128, 1)],
            'data_type': [ms.float32, ms.float32, ms.float32,
                          ms.float32, ms.int32, ms.float32, ms.float32, ms.float32],
        },
        'global_replaybuffer': {
            'number': 1,
            'type': MAPPOReplayBuffer,
            'capacity': 26,
            'data_shape': [(128, NUM_AGENT * NUM_AGENT * 6)],
            'data_type': [ms.float32],
        }

    },

    'collect_environment': {
        'type': MPEMultiEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': MPEMultiEnvironment,
        'params': eval_env_params
    },
}
