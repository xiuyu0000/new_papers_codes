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
"""QMIX Train"""

import argparse
from src import config
from src.qmix_trainer import QMIXTrainer
from mindspore import context
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, Callback

parser = argparse.ArgumentParser(description='MindSpore Reinforcement QMIX')
parser.add_argument('--episode', type=int, default=25000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the qmix example(Default: Auto).')
options, _ = parser.parse_known_args()


class StepInfoCollectCallback(Callback):
    """Step info call back for collect, provides detail information getting from sc2"""

    def __init__(self, interval):
        self.interval = interval
        self.won_counter = []
        self.dead_allies_counter = []
        self.dead_enemies_counter = []

    def episode_end(self, params):
        """Step info stats during training"""
        battle_won, dead_allies, dead_enemies = params.others
        self.won_counter.append(battle_won.asnumpy()[0])
        self.dead_allies_counter.append(dead_allies.asnumpy()[0])
        self.dead_enemies_counter.append(dead_enemies.asnumpy()[0])
        if (params.cur_episode + 1) % self.interval == 0:
            win_rate = sum(self.won_counter) / self.interval
            avg_dead_allies = sum(self.dead_allies_counter) / self.interval
            avg_dead_enemies = sum(self.dead_enemies_counter) / self.interval
            self.won_counter = []
            self.dead_allies_counter = []
            self.dead_enemies_counter = []

            print("---------------------------------------------------")
            print("The average statical results of these {} episodes during training is ".format(self.interval),
                  flush=True)
            print("Win Rate: {:.3}".format(win_rate), flush=True)
            print("Average Dead Allies: {:.3}".format(avg_dead_allies), flush=True)
            print("Average Dead Enemies: {:.3}".format(avg_dead_enemies), flush=True)
            print("---------------------------------------------------")


class StepInfoEvalCallback(Callback):
    """Step info call back for evaluation, provides detail information getting from sc2"""

    def __init__(self, eval_rate, times):
        super(StepInfoEvalCallback, self).__init__()
        if not isinstance(eval_rate, int) or eval_rate < 0:
            raise ValueError("The arg of 'evaluation_frequency' must be int and >= 0, but get ", eval_rate)
        self._eval_rate = eval_rate
        self.won_counter = []
        self.dead_allies_counter = []
        self.dead_enemies_counter = []
        self.times = times

    def begin(self, params):
        """Store the eval rate in the begin of training, run once."""
        params.eval_rate = self._eval_rate

    def episode_end(self, params):
        """Run evaluate in the end of episode, and print the rewards."""
        if self._eval_rate != 0 and params.cur_episode > 0 and \
                params.cur_episode % self._eval_rate == 0:
            # Call the `evaluate` function provided by user.
            for _ in range(self.times):
                battle_won, dead_allies, dead_enemies = params.evaluate()
                self.won_counter.append(battle_won.asnumpy()[0])
                self.dead_allies_counter.append(dead_allies.asnumpy()[0])
                self.dead_enemies_counter.append(dead_enemies.asnumpy()[0])

            win_rate = sum(self.won_counter) / self.times
            avg_dead_allies = sum(self.dead_allies_counter) / self.times
            avg_dead_enemies = sum(self.dead_enemies_counter) / self.times
            self.won_counter = []
            self.dead_allies_counter = []
            self.dead_enemies_counter = []

            print("---------------------------------------------------")
            print("The average statical results of these {} episodes during evaluation is ".format(self.times),
                  flush=True)
            print("Win Rate: {:.3}".format(win_rate), flush=True)
            print("Average Dead Allies: {:.3}".format(avg_dead_allies), flush=True)
            print("Average Dead Enemies: {:.3}".format(avg_dead_enemies), flush=True)
            print("---------------------------------------------------")


def train(episode=options.episode):
    """start to train qmix algorithm"""
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)

    context.set_context(mode=context.GRAPH_MODE)
    qmix_session = Session(config.algorithm_config)
    loss_cb = LossCallback()
    step_info_train_cb = StepInfoCollectCallback(100)
    step_info_eval_cb = StepInfoEvalCallback(200, 20)
    ckpt_cb = CheckpointCallback(50, config.trainer_params['ckpt_path'])
    cbs = [step_info_train_cb, step_info_eval_cb, loss_cb, ckpt_cb]
    qmix_session.run(class_type=QMIXTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)


if __name__ == "__main__":
    train()
