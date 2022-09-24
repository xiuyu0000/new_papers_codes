# Copyright 2021
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
DQN eval example.
"""
import argparse
from src.dqn_trainer import DQNTrainer
from src import config
from mindspore_rl.core import Session
from mindspore import context
from mindspore import dtype as mstype

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DQN')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the dqn example(Default: Auto).')
parser.add_argument('--ckpt_path', type=str, default=None, help='The ckpt file in eval.')
args = parser.parse_args()

def dqn_eval():
    '''DQN evaluation entry'''
    if args.device_target != 'Auto':
        context.set_context(device_target=args.device_target)

    compute_type = mstype.float16 if context.get_context('device_target') in ['Ascend'] else mstype.float32
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type

    context.set_context(mode=context.GRAPH_MODE)
    config.trainer_params.update({'ckpt_path': args.ckpt_path})
    dqn_session = Session(config.algorithm_config)
    dqn_session.run(class_type=DQNTrainer, is_train=False, params=config.trainer_params)

if __name__ == "__main__":
    dqn_eval()
