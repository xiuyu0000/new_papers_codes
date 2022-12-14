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

if [ $# != 2 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash ./run_eval.sh [DATASET] [BATCH_SIZE]"
  echo "for example: bash ./run_eval.sh up-3d 16"
  echo "=============================================================================================================="
  exit 1
fi

DATASET=$1
BATCH_SIZE=$2

cd ..

python eval.py  \
  --dataset=$DATASET \
  --batch_size=$BATCH_SIZE > eval.log 2>&1 &
