import numpy as np
import math


def cyclical_step_decay(initial_lr, cycle_step=30, min_lr=1e-8, max_epochs=3000,
                        rate_decay=0.8, policy='exp_range', multiplier=False):
    """
    implementation of CLR
    :param initial_lr:
    :param cycle_step:
    :param min_lr:
    :param max_epochs:
    :param rate_decay:
    :param policy:
    :param multiplier:
    :return:
    """
    if policy not in ['exp_range', 'triangular', 'triangular_2']:
        raise ValueError('Not supported decay policy.')

    def _rate_sch(epoch):
        current_iter = np.floor(1 + epoch / (cycle_step * 2))
        x = np.abs(epoch / cycle_step - 2 * current_iter + 1)
        if policy == 'exp_range':
            max_lr = min_lr + initial_lr * math.pow(1.0 - epoch / max_epochs, rate_decay)
            lr = min_lr + (max_lr - min_lr) * np.maximum(0, x)

        elif policy == 'triangular':
            lr = min_lr + (initial_lr - min_lr) * np.maximum(0, x)

        else:
            lr = min_lr + (initial_lr - min_lr) * np.maximum(0, x / math.pow(2, current_iter))

        lr = max(lr, min_lr)

        return lr if not multiplier else lr / initial_lr

    return _rate_sch
