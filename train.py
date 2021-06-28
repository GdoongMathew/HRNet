import tensorflow as tf
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from hrnet import HRNet_W18 as HRNet
from load_data.load_airbus_ship_data import load_csv_data
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from utils import cyclical_step_decay
import tensorflow_addons as tfad

train_shape = (608, 608, 3)
(train_ds, val_ds), info = load_csv_data(r'E:\Data\airbus-ship-detection',
                                         with_info=True,
                                         batch_size=2,
                                         train_shape=train_shape[:2])
steps_per_epochs = info['train_images'] // info['batch_size']
lr = 1e-5
epochs = 20000

model = HRNet(input_shape=train_shape,
              num_classes=1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tfad.losses.SigmoidFocalCrossEntropy(),
    metrics='acc'
)

model.summary(line_length=150)

callbacks = [
    LearningRateScheduler(cyclical_step_decay(lr, cycle_step=200,
                                              min_lr=1e-10,
                                              max_epochs=epochs)),
    TensorBoard(log_dir='./logs', profile_batch=(3, 8)),
    ModelCheckpoint('weight/HRNet.h5', save_best_only=True, monitor='val_acc')
]

model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epochs,
    callbacks=[callbacks]
)
