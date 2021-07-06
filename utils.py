from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
import tensorflow as tf
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


class DiceFocalLoss(Loss):
    def __init__(self, alpha=10.0, focal_alpha=0.8, focal_gamma=2.0, dice_smooth=1.0, *args, **kwargs):
        self.alpha = alpha
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        super(DiceFocalLoss, self).__init__(*args, **kwargs)

    def focal_loss(self, y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        focal_loss = K.mean(self.focal_alpha * K.pow((1 - bce_exp), self.focal_gamma) * bce)
        return focal_loss

    def dice_loss(self, y_true, y_pred):
        smooth = K.epsilon()
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return 1 - (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    def call(self, y_true, y_pred):
        return self.alpha * self.focal_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

    def get_config(self):
        base_config = super(DiceFocalLoss, self).get_config()
        base_config.update({
            'alpha': self.alpha,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'dice_smooth': self.dice_smooth
        })

        return base_config




