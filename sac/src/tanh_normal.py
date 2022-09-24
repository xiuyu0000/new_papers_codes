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
"""TanhMultivariateNormalDiag"""
import numpy as np
import mindspore
import mindspore.nn.probability.distribution as msd
import mindspore.nn.probability.bijector as msb
from mindspore.ops import operations as P
from mindspore import Tensor


class TanhBijector(msb.Bijector):
    """Tanh Bijector"""
    def __init__(self, reduce_axis=None, name='Tanh'):
        """
        Constructor of Tanh Bijector.
        """
        param = dict(locals())
        super(TanhBijector, self).__init__(
            is_constant_jacobian=False,
            is_injective=True,
            name=name, dtype=None,
            param=param)

        self.reduce_axis = reduce_axis
        self.tanh = P.Tanh()
        self.softplus = P.Softplus()
        self.log2 = Tensor([np.log(2.0)], mindspore.float32)

    def forward_log_jacobian(self, x):
        log_jac = 2.0 * (self.log2 - x - self.softplus(-2.0 * x))
        if self.reduce_axis is not None:
            log_jac = log_jac.sum(axis=self.reduce_axis)
        return log_jac

    def _forward(self, x):
        return self.tanh(x)


class MultivariateNormalDiag(msd.Normal):
    """MultivariateNormalDiag distribute"""
    def __init__(self,
                 loc=None,
                 scale=None,
                 reduce_axis=None,
                 seed=None,
                 dtype=mindspore.float32,
                 name="MultivariateNormalDiag"):
        super(MultivariateNormalDiag, self).__init__(loc, scale, seed, dtype, name)
        self.reduce_axis = reduce_axis

    def _log_prob(self, value, mean=None, sd=None):
        log_prob = super()._log_prob(value, mean=mean, sd=sd)
        if self.reduce_axis is not None:
            log_prob = log_prob.sum(axis=self.reduce_axis)
        return log_prob


class TanhMultivariateNormalDiag(msd.TransformedDistribution):
    """MultivariateNormalDiag with Tanh Bijector"""
    def __init__(self,
                 loc=None,
                 scale=None,
                 reduce_axis=None,
                 seed=0,
                 dtype=mindspore.float32,
                 name="TanhMultivariateNormalDiag"):
        distribution = MultivariateNormalDiag(loc=loc, scale=scale, reduce_axis=reduce_axis, seed=seed, dtype=dtype)
        super(TanhMultivariateNormalDiag, self).__init__(distribution=distribution,
                                                         bijector=TanhBijector(reduce_axis=reduce_axis),
                                                         seed=seed,
                                                         name=name)

    def sample_and_log_prob(self, shape, means, stds):
        '''
        Combine sample() and log_prob() to improve numeric stable:
        x' = atanh(tanh(x).clip()) will result error results when x is in the saturation ragion.
        '''
        x = self.distribution.sample(shape, means, stds)
        y = self.bijector.forward(x)

        unadjust_prob = self.distribution.log_prob(x, means, stds)
        log_jacobian = self.bijector.forward_log_jacobian(x)
        log_prob = unadjust_prob - log_jacobian
        return y, log_prob

    def _sample(self, *args, **kwargs):
        org_sample = self.distribution.sample(*args, **kwargs)
        return self.bijector.forward(org_sample)
