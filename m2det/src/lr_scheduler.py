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
"""Learning rate scheduler."""

from collections import Counter

import numpy as np


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    if lr_epochs[-1] != max_epoch:
        lr_epochs = lr_epochs[:-1]
        lr_epochs.append(max_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    if isinstance(gamma, list):
        gamma_per_milestone = [1]
        for reduction in gamma:
            gamma_per_milestone.append(reduction * gamma_per_milestone[-1])
    current_milestone = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        elif isinstance(gamma, list):
            if milestones_steps_counter[i] == 1:
                current_milestone += milestones_steps_counter[i]
                lr = base_lr * gamma_per_milestone[current_milestone]
        else:
            lr = lr * gamma**milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def get_lr(cfg, steps_per_epoch):
    """generate learning rate."""
    lr = warmup_step_lr(cfg['lr'],
                        cfg['lr_epochs'],
                        steps_per_epoch,
                        cfg['warmup'],
                        cfg['total_epochs'],
                        gamma=cfg['gamma'],
                        )
    return lr
