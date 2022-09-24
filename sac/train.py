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
SAC training example.
"""

#pylint: disable=C0413
import argparse
from src import config
from src.sac_trainer import SACTrainer
from mindspore import context
from mindspore import dtype as mstype
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback

parser = argparse.ArgumentParser(description='MindSpore Reinforcement SAC')
parser.add_argument('--episode', type=int, default=1000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the sac example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    '''SAC train entry.'''
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    device = context.get_context('device_target')
    if device in ['CPU', 'GPU']:
        context.set_context(enable_graph_kernel=True)

    compute_type = mstype.float32 if options.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
    if compute_type == mstype.float16 and device in ['CPU']:
        raise ValueError("Fp16 mode is supported by Ascend and GPU backend.")

    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    sac_session = Session(config.algorithm_config)
    loss_cb = LossCallback()
    ckpt_cb = CheckpointCallback(100, config.trainer_params['ckpt_path'])
    eval_cb = EvaluateCallback(30)
    cbs = [loss_cb, ckpt_cb, eval_cb]
    sac_session.run(class_type=SACTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)

if __name__ == "__main__":
    train()
