from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os

AUTO = tf.data.AUTOTUNE


def decode_data(img_folder, df, train_shape):

    def _decode_str(name):
        name = name.decode('utf-8')
        mask_str = df.loc[df.ImageId == name]['EncodedPixels']
        img_path = os.path.join(img_folder, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = np.zeros(img.shape[0] * img.shape[1], dtype=np.uint8)
        if isinstance(mask_str, str):
            mask_str = [mask_str]

        if not isinstance(mask_str, float):
            for _m in mask_str:
                s = _m.split()
                for i in range(len(s) // 2):
                    start = int(s[2 * i]) - 1
                    length = int(s[2 * i + 1])
                    mask[start:start + length] = 1
        mask = mask.reshape(img.shape[:2]).T
        img_shape = img.shape
        return img, mask, img_shape[0], img_shape[1]

    def _decode_data(tf_data):
        img, mask, w, h = tf.numpy_function(_decode_str, [tf_data, ], [tf.uint8, tf.uint8, tf.int32, tf.int32])
        img = tf.reshape(img, (w, h, 3))
        mask = tf.reshape(mask, (w, h, 1))

        if train_shape is not None:
            img = tf.image.resize(img, train_shape, method='bilinear')
            img = tf.cast(img, tf.uint8)
            mask = tf.image.resize(mask, train_shape, method='nearest')

        return img, mask

    return _decode_data


def augmentation():

    aug = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,
                                                                     width_factor=0.2,
                                                                     fill_mode='constant',
                                                                     fill_value=0.),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(1.0,
                                                                  fill_mode='constant',
                                                                  fill_value=0.),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.4,
                                                              width_factor=0.4,
                                                              fill_mode='constant',
                                                              fill_value=0.),
    ])

    def _aug_fn(img, mask):
        img_chn = img.shape[-1]
        tmp = tf.concat([img, mask], axis=-1)
        tmp = aug(tmp, training=True)
        img = tmp[:, :, :, :img_chn]
        mask = tmp[:, :, :, img_chn:]
        return img, mask
    return _aug_fn


def load_csv_data(root_dir,
                  remove_empty=True,
                  batch_size=64,
                  train_shape=None,
                  with_info=False):
    csv_path = os.path.join(root_dir, 'train_ship_segmentations_v2.csv')
    data_df = pd.read_csv(csv_path)

    train_path = os.path.join(root_dir, 'train_v2')

    data_df = data_df.dropna() if remove_empty else data_df.fillna(value='')
    train_names, val_names = train_test_split(data_df.ImageId.drop_duplicates().to_list(), test_size=0.05)

    train_df = data_df[data_df.ImageId.isin(train_names)]
    val_df = data_df[data_df.ImageId.isin(val_names)]

    train_ds = (tf.data.Dataset.from_tensor_slices(train_names)
                .shuffle(len(train_names))
                .repeat()
                .map(decode_data(train_path, train_df, train_shape), num_parallel_calls=AUTO)
                .batch(batch_size, drop_remainder=True)
                .map(augmentation(), num_parallel_calls=AUTO)
                .prefetch(AUTO)
                )

    val_ds = (tf.data.Dataset.from_tensor_slices(val_names)
              .map(decode_data(train_path, val_df, train_shape), num_parallel_calls=AUTO)
              .batch(batch_size, drop_remainder=True)
              .prefetch(AUTO)
              )

    if not with_info:
        return train_ds, val_ds
    else:
        info = {
            'train_images': len(train_names),
            'validation_images': len(val_names),
            'batch_size': batch_size,
        }
        return (train_ds, val_ds), info
