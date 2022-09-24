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
"""SAC"""
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import SoftUpdate
from src.tanh_normal import TanhMultivariateNormalDiag


class SACPolicy():
    """
    This is SACPolicy class. You should define your networks (SACActorNet and SACCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (SACLossCell here) which calculates the loss between policy and your ground truth value.
    """
    class SACActorNet(nn.Cell):
        """
        SACActorNet is the actor network of SAC algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution
        """
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size, compute_type=mindspore.float32):
            super(SACPolicy.SACActorNet, self).__init__()
            self.fc1 = nn.Dense(input_size,
                                hidden_size1,
                                weight_init='XavierUniform',
                                activation='relu').to_float(compute_type)
            self.fc2 = nn.Dense(hidden_size1,
                                hidden_size2,
                                weight_init='XavierUniform',
                                activation='relu').to_float(compute_type)
            self.fc3 = nn.Dense(hidden_size2,
                                output_size * 2,
                                weight_init='XavierUniform').to_float(compute_type)
            self.split = P.Split(axis=-1, output_num=2)
            self.exp = P.Exp()
            self.max = P.Maximum()

        def construct(self, x):
            """calculate miu and sigma"""
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            means, stds = self.split(x)
            # Clip stds to improve numeric stable
            stds = self.max(stds, -5.0)
            stds = self.exp(stds)
            return means, stds

    class SACCriticNet(nn.Cell):
        """
        SACCriticNet is the critic network of SAC algorithm. It takes a set of states as input
        and outputs the value of input state
        """
        def __init__(self, obs_size, action_size, hidden_size1, hidden_size2,
                     output_size, compute_type=mindspore.float32):
            super(SACPolicy.SACCriticNet, self).__init__()
            self.concat = P.Concat(axis=1)
            self.fc1 = nn.Dense(obs_size + action_size,
                                hidden_size1,
                                weight_init='XavierUniform',
                                activation='relu').to_float(compute_type)
            self.fc2 = nn.Dense(hidden_size1,
                                hidden_size2,
                                weight_init='XavierUniform',
                                activation='relu').to_float(compute_type)
            self.fc3 = nn.Dense(hidden_size2, output_size, weight_init='XavierUniform').to_float(compute_type)

        def construct(self, obs, action):
            """predict value"""
            x = self.concat((obs, action))
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    class RandomPolicy(nn.Cell):
        def __init__(self, action_space_dim):
            super(SACPolicy.RandomPolicy, self).__init__()
            self.uniform = P.UniformReal()
            self.shape = (action_space_dim,)

        def construct(self):
            return self.uniform(self.shape) * 2 - 1

    class CollectPolicy(nn.Cell):
        """Collect Policy"""
        def __init__(self, actor_net):
            super(SACPolicy.CollectPolicy, self).__init__()
            self.actor_net = actor_net
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)

        def construct(self, obs):
            means, stds = self.actor_net(obs)
            actions = self.dist.sample((), means, stds)
            return actions


    class EvalPolicy(nn.Cell):
        """Eval Policy"""
        def __init__(self, actor_net):
            super(SACPolicy.EvalPolicy, self).__init__()
            self.actor_net = actor_net

        def construct(self, obs):
            means, _ = self.actor_net(obs)
            return means

    def __init__(self, params):
        self.actor_net = self.SACActorNet(params['state_space_dim'],
                                          params['hidden_size1'],
                                          params['hidden_size2'],
                                          params['action_space_dim'],
                                          params['compute_type'])
        self.critic_net1 = self.SACCriticNet(params['state_space_dim'],
                                             params['action_space_dim'],
                                             params['hidden_size1'],
                                             params['hidden_size2'],
                                             1,
                                             params['compute_type'])
        self.critic_net2 = self.SACCriticNet(params['state_space_dim'],
                                             params['action_space_dim'],
                                             params['hidden_size1'],
                                             params['hidden_size2'],
                                             1,
                                             params['compute_type'])
        self.target_critic_net1 = self.SACCriticNet(params['state_space_dim'],
                                                    params['action_space_dim'],
                                                    params['hidden_size1'],
                                                    params['hidden_size2'],
                                                    1,
                                                    params['compute_type'])
        self.target_critic_net2 = self.SACCriticNet(params['state_space_dim'],
                                                    params['action_space_dim'],
                                                    params['hidden_size1'],
                                                    params['hidden_size2'],
                                                    1,
                                                    params['compute_type'])

        self.init_policy = self.RandomPolicy(params['action_space_dim'])
        self.collect_policy = self.CollectPolicy(self.actor_net)
        self.eval_policy = self.EvalPolicy(self.actor_net)


class SACActor(Actor):
    """
    This is an actor class of SAC algorithm, which is used to interact with environment, and
    generate/insert experience (data)
    """

    def __init__(self, params=None):
        super(SACActor, self).__init__()
        self._params_config = params
        self._environment = params['collect_environment']
        self._eval_env = params['eval_environment']
        self.init_policy = params['init_policy']
        self.collect_policy = params['collect_policy']
        self.eval_policy = params['eval_policy']
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=0)

    def act(self, phase, params):
        """collect experience and insert to replay buffer (used during training)"""
        if phase == 1:
            action = self.init_policy()
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 2:
            params = self.expand_dims(params, 0)
            action = self.collect_policy(params)
            action = self.squeeze(action)
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 3:
            params = self.expand_dims(params, 0)
            action = self.eval_policy(params)
            new_state, reward, _ = self._eval_env.step(action)
            return reward, new_state
        self.print("Phase is incorrect")
        return 0


class SACLearner(Learner):
    """This is the learner class of SAC algorithm, which is used to update the policy net"""

    class CriticLossCell(nn.Cell):
        """CriticLossCell"""
        def __init__(self, gamma, alpha, critic_loss_weight, reward_scale_factor, actor_net, target_critic_net1,
                     target_critic_net2, critic_net1, critic_net2):
            super(SACLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.alpha = alpha
            self.reward_scale_factor = reward_scale_factor
            self.critic_loss_weight = critic_loss_weight
            self.actor_net = actor_net
            self.target_critic_net1 = target_critic_net1
            self.target_critic_net2 = target_critic_net2
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)
            self.min = P.Minimum()
            self.exp = P.Exp()
            self.mse = nn.MSELoss(reduction='none')

        def construct(self, next_state, reward, state, action):
            """Calculate critic loss"""
            next_means, next_stds = self.actor_net(next_state)
            next_action, next_log_prob = self.dist.sample_and_log_prob((), next_means, next_stds)

            target_q_value1 = self.target_critic_net1(next_state, next_action).squeeze(axis=-1)
            target_q_value2 = self.target_critic_net2(next_state, next_action).squeeze(axis=-1)
            target_q_value = self.min(target_q_value1, target_q_value2) - self.exp(self.alpha) * next_log_prob
            td_target = self.reward_scale_factor * reward + self.gamma * target_q_value

            pred_td_target1 = self.critic_net1(state, action).squeeze(axis=-1)
            pred_td_target2 = self.critic_net2(state, action).squeeze(axis=-1)
            critic_loss1 = self.mse(td_target, pred_td_target1)
            critic_loss2 = self.mse(td_target, pred_td_target2)
            critic_loss = (critic_loss1 + critic_loss2).mean()
            return critic_loss * self.critic_loss_weight


    class ActorLossCell(nn.Cell):
        """ActorLossCell"""
        def __init__(self, alpha, actor_loss_weight, actor_net, critic_net1, critic_net2):
            super(SACLearner.ActorLossCell, self).__init__(auto_prefix=False)
            self.alpha = alpha
            self.actor_net = actor_net
            self.actor_loss_weight = actor_loss_weight
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)
            self.min = P.Minimum()
            self.exp = P.Exp()

        def construct(self, state):
            means, stds = self.actor_net(state)
            action, log_prob = self.dist.sample_and_log_prob((), means, stds)

            target_q_value1 = self.critic_net1(state, action)
            target_q_value2 = self.critic_net2(state, action)
            target_q_value = self.min(target_q_value1, target_q_value2).squeeze(axis=-1)
            actor_loss = (self.exp(self.alpha) * log_prob - target_q_value).mean()
            return actor_loss * self.actor_loss_weight

    class AlphaLossCell(nn.Cell):
        """AlphaLossCell"""
        def __init__(self, alpha, target_entropy, alpha_loss_weight, actor_net):
            super(SACLearner.AlphaLossCell, self).__init__(auto_prefix=False)
            self.alpha = alpha
            self.target_entropy = target_entropy
            self.alpha_loss_weight = alpha_loss_weight
            self.actor_net = actor_net
            self.dist = TanhMultivariateNormalDiag(reduce_axis=-1)

        def construct(self, state_list):
            means, stds = self.actor_net(state_list)
            _, log_prob = self.dist.sample_and_log_prob((), means, stds)
            entropy_diff = -log_prob - self.target_entropy
            alpha_loss = self.alpha * entropy_diff
            alpha_loss = alpha_loss.mean()
            return alpha_loss * self.alpha_loss_weight


    def __init__(self, params):
        super(SACLearner, self).__init__()
        self._params_config = params
        gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        actor_net = params['actor_net']
        critic_net1 = params['critic_net1']
        critic_net2 = params['critic_net2']
        target_critic_net1 = params['target_critic_net1']
        target_critic_net2 = params['target_critic_net2']

        init_alpha = params['init_alpha']
        alpha = Parameter(Tensor([init_alpha,], mindspore.float32), name='alpha', requires_grad=True)

        critic_loss_net = SACLearner.CriticLossCell(gamma,
                                                    alpha,
                                                    params['critic_loss_weight'],
                                                    params['reward_scale_factor'],
                                                    actor_net,
                                                    target_critic_net1,
                                                    target_critic_net2,
                                                    critic_net1,
                                                    critic_net2)
        actor_loss_net = SACLearner.ActorLossCell(alpha,
                                                  params['actor_loss_weight'],
                                                  actor_net,
                                                  critic_net1,
                                                  critic_net2)
        alpha_loss_net = SACLearner.AlphaLossCell(alpha,
                                                  params['target_entropy'],
                                                  params['alpha_loss_weight'],
                                                  actor_net)

        critic_trainable_params = critic_net1.trainable_params() + critic_net2.trainable_params()
        critic_optim = nn.Adam(critic_trainable_params, learning_rate=params['critic_lr'])
        actor_optim = nn.Adam(actor_net.trainable_params(), learning_rate=params['actor_lr'])
        alpha_optim = nn.Adam([alpha], learning_rate=params['alpha_lr'])

        self.critic_train = nn.TrainOneStepCell(critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(actor_loss_net, actor_optim)
        self.alpha_train = nn.TrainOneStepCell(alpha_loss_net, alpha_optim)

        factor, interval = params['update_factor'], params['update_interval']
        params = critic_net1.trainable_params() + critic_net2.trainable_params()
        target_params = target_critic_net1.trainable_params() + target_critic_net2.trainable_params()
        self.soft_updater = SoftUpdate(factor, interval, params, target_params)

    def learn(self, experience):
        """learn"""
        state, action, reward, next_state = experience
        reward = reward.squeeze(axis=-1)

        critic_loss = self.critic_train(next_state, reward, state, action)
        actor_loss = self.actor_train(state)
        alpha_loss = self.alpha_train(state)
        self.soft_updater()
        loss = critic_loss + actor_loss + alpha_loss
        return loss
