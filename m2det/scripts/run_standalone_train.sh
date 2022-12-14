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
# ============================================================================


if [[ $# -ne 3 ]]; then
    echo "Usage: bash ./scripts/run_standalone_train.sh [DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET_PATH]"
exit 1
fi

if [ ! -d "logs" ]; then
        mkdir logs
fi

nohup python -u train.py --device_id=$1 --pretrained_backbone $2 --dataset_path $3 > ./logs/train.log 2>&1 &
echo "Training started on device $1 ! PID: $!"
echo $! > ./logs/train.pid
