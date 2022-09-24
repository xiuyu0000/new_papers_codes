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
"""A3C Trainer"""
import collections
import statistics
import tqdm

from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer
import mindspore
import mindspore.nn as nn
from mindspore.nn.reinforcement._tensors_queue import TensorsQueue
from mindspore.ops import operations as ops
from mindspore import ms_function


class A3CTrainer(Trainer):
    '''A3CTrainer'''
    def __init__(self, msrl):
        nn.Cell.__init__(self, auto_prefix=False)
        self.reduce_sum = ops.ReduceSum()
        self.actor_nums = msrl.actors.__len__()
        self.weight_copy = msrl.learner.weight_copy
        shapes = []
        for i in self.weight_copy:
            shapes.append(i.shape)
        self.grads_queue = TensorsQueue(dtype=mindspore.float32, shapes=shapes, size=10, name="grads_q")
        self.zero = mindspore.Tensor(0, mindspore.float32)
        super(A3CTrainer, self).__init__(msrl)

    #pylint: disable=W0613
    def train(self, episodes, callbacks=None, ckpt_path=None):
        '''Train A3C'''
        running_reward = 0
        episode_reward: collections.deque = collections.deque(maxlen=100)
        with tqdm.trange(episodes) as t:
            for i in t:
                loss, reward = self.train_one_episode()
                episode_reward.append(reward.asnumpy().tolist())
                running_reward = statistics.mean(episode_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=reward.asnumpy(), loss=loss.asnumpy(), running_reward=running_reward)
                if running_reward > 195 and i >= 100:
                    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}.')
                    break
                if i == episodes - 1:
                    print(f'\nFailed to solved this problem after running {episodes} episodes.')

    @ms_function
    def train_one_episode(self):
        '''Train one episode'''
        a3c_loss = self.zero
        episode_reward = self.zero
        for i in range(self.actor_nums):
            rewards_i, grads_i, loss_i = self.msrl.actors[i].act(trainer.COLLECT, actor_id=0,
                                                                 weight_copy=self.weight_copy)
            self.grads_queue.put(grads_i)
            a3c_loss += loss_i
            episode_reward += rewards_i
            self.msrl.agent_learn(self.grads_queue.pop()) # side effect
        return a3c_loss / self.actor_nums, episode_reward / self.actor_nums

    def evaluate(self):
        '''Default evaluate'''
        return

    def trainable_variables(self):
        '''Default trainable variables'''
        return
