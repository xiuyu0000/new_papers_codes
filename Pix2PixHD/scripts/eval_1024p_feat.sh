#!/bin/bash
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

# first precompute and cluster all features
python encode_features.py --name label2city_1024p_feat --batch_size 1 --no_flip True \
                          --serial_batches True --instance_feat True --continue_train True \
                          --netG local --ngf 32 --resize_or_crop none;
# use instance-wise features
python eval.py --name label2city_1024p_feat --phase test --batch_size 1 --no_flip True \
               --serial_batches True --instance_feat True \
               --netG local --ngf 32 --resize_or_crop none
