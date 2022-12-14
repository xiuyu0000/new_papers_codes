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

from mindspore import nn


class CMREvalNet(nn.Cell):
    """
    Encapsulation class of CMR evaluation
    """
    def __init__(self, network):
        super(CMREvalNet, self).__init__()
        self.network = network

    def construct(self, images):
        """
        :param images: shape = (B, 3, 224, 224)
        """
        return self.network(images)
