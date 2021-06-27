import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Input

from tensorflow.keras import Model


def exchange_unit(filters,
                  down_sample=False,
                  activation='relu',
                  data_format='channels_last',
                  name='exchangeunit',
                  up_sample_method='bilinear',
                  fusion_method='add',
                  ):
    """
    Exchange unit, input filters will be down/up sampling and merge into different resolutions. In the third stage,
    the highest resolution filters will be down sample twice, the first down sampled filters will be fused with the middle
    resolution filters, and the second down sampled filters will be fused with the lowest ones.

    HRF -> HRF-m -> HRF-l, MRF -> MRF-h, LRF -> LRF-m -> LRF-h
                               -> MRF-l
    Fusion process:
    new_HRF = Fuse([HRF, MRF-h, LRF-h]),
    new_MRF = Fuse([MRF, HRF-m, LRF-m]),
    new_LRF = Fuse([LRF, HRF-l, MRF-l])

    This implementation is a bit different with the original description in the paper.
    Original description:
    HRF -> HRF-m-1, HRF -> HRF-m-2 -> HRF-l
    MRF -> MRF-l, MRF -> MRF-h
    LRF -> LRF-m-1, LRF -> LRF-m-2 -> LRF-h

    Fusion process:
    new_HRF = Fuse([HRF, MRF-h, LRF-h]),
    new_MRF = Fuse([MRF, LRF-m-1, HRF-m-1]),
    new_LRF = Fuse([LRF, LRF-m-1, HRF-l])

    :param filters:
    :param down_sample: output a new lower resolution filters with doubling the number of channels
    :param activation: activation method
    :param data_format: channels_last as default
    :param name: name of the layers
    :param up_sample_method: upsampling method, can be bilinear, nearest, or conv2d_transpose
    :param fusion_method: add, or concat
    :return:
    """
    assert isinstance(filters, (type, list))
    assert up_sample_method in ['nearest', 'bilinear', 'conv2d_transpose']
    assert fusion_method in ['add', 'concat']
    bn_axis = 3 if data_format == 'channels_last' else 1
    w_axis = 2 if data_format == 'channels_last' else 3
    num_outputs = len(filters) if not down_sample else len(filters) + 1

    filters = sorted(filters, key=lambda x: x.shape[w_axis], reverse=True)

    def _down_sample(_filters, _name):
        _c_num = _filters.shape[bn_axis]
        _filters = Conv2D(_c_num * 2, 3,
                          strides=2,
                          padding='same',
                          data_format=data_format,
                          name=f'{_name}_conv2d')(_filters)
        _filters = BatchNormalization(axis=bn_axis, name=f'{_name}_bn')(_filters)
        return _filters

    def _up_sample(_filters, _name):
        _c_num = _filters.shape[bn_axis]
        _up_fn = UpSampling2D(interpolation=up_sample_method,
                              data_format=data_format,
                              name=f'{_name}_upsample2d') if up_sample_method in ['nearest', 'bilinear'] else \
            Conv2DTranspose(_c_num // 2, 3,
                            strides=2,
                            padding='same',
                            data_format=data_format,
                            name=f'{_name}_conv2d_transpose')

        _filters = Conv2D(_c_num // 2, 1,
                          padding='same',
                          data_format=data_format,
                          name=f'{_name}_conv2d')(_filters)
        _filters = _up_fn(_filters)
        _filters = BatchNormalization(axis=bn_axis, name=f'{_name}_bn')(_filters)
        return _filters

    output_filters = []
    for i, _filter in enumerate(filters):
        tmp_filters = [None] * num_outputs
        tmp_filters[i] = _filter
        down_times = num_outputs - 1 - i
        up_times = i

        up_filters = down_filters = _filter
        for j in range(down_times):
            _name = f'{name}_downsample_{i}_{j}'
            down_filters = _down_sample(down_filters, _name)
            tmp_filters[j + i + 1] = down_filters

        for j in range(up_times):
            _name = f'{name}_upsample_{i}_{j}'
            up_filters = _up_sample(up_filters, _name)
            tmp_filters[i - 1 - j] = up_filters

        # tmp_filters should be like [high_resolution_filters, middle_resolution_filter, low_resolution_filters]
        output_filters.append(tmp_filters)

    def fusion_layer(_filters, layer_name='fusion'):
        if fusion_method == 'add':
            fused_filter = Add(name=f'{layer_name}_add')(_filters)
        else:
            _c_num = _filters[0].shape[bn_axis]
            fused_filter = Concatenate(axis=bn_axis, name=f'{layer_name}_concat')(_filters)
            fused_filter = Conv2D(_c_num, 1,
                                  padding='same',
                                  data_format=data_format,
                                  name=f'{layer_name}_conv2d')(fused_filter)
            fused_filter = BatchNormalization(axis=bn_axis, name=f'{layer_name}_bn')(fused_filter)
        fused_filter = Activation(activation=activation, name=f'{layer_name}_{activation}')(fused_filter)
        return fused_filter

    if len(filters) != 1:
        final_filters = []
        # fusing filters with the same size.
        for i, same_filters in enumerate(zip(*output_filters)):
            fusion_name = f'{name}_fusion_{i}'
            same_filter = fusion_layer(same_filters, layer_name=fusion_name)
            final_filters.append(same_filter)
    else:
        final_filters = output_filters[0]
    assert len(final_filters) == num_outputs
    return final_filters


def basic_block(filters,
                activation='relu',
                data_format='channels_last',
                units=4,
                name='basic_block'):
    bn_axis = 3 if data_format == 'channels_last' else 1
    channel_num = filters.shape[bn_axis]

    for i in range(units):
        residual = filters
        filters = Conv2D(channel_num, 3,
                         padding='same',
                         data_format=data_format,
                         name=f'{name}_conv2d_{i}_1')(filters)

        filters = BatchNormalization(axis=bn_axis, name=f'{name}_bn_{i}_1')(filters)
        filters = Activation(activation=activation,
                             name=f'{name}_{activation}_{i}_1')(filters)

        filters = Conv2D(channel_num, 3,
                         padding='same',
                         data_format=data_format,
                         name=f'{name}_conv2d_{i}_2')(filters)
        filters = BatchNormalization(axis=bn_axis, name=f'{name}_bn_{i}_2')(filters)

        filters = Add(name=f'{name}_add_{i}')([residual, filters])
        filters = Activation(activation=activation, name=f'{name}_{activation}_{i}_2')(filters)
    return filters


def bottleneck(filters,
               add_residual=False,
               activation='relu',
               data_format='channels_last',
               name='bottleneck'):
    bn_axis = 3 if data_format == 'channels_last' else 1
    residual = filters if add_residual else None
    filters = Conv2D(64, 1,
                     padding='same',
                     data_format=data_format,
                     name=f'{name}_conv2d_1')(filters)

    filters = BatchNormalization(axis=bn_axis, name=f'{name}_bn_1')(filters)
    filters = Activation(activation=activation, name=f'{name}_{activation}_1')(filters)

    filters = Conv2D(64, 3,
                     padding='same',
                     data_format=data_format,
                     name=f'{name}_conv2d_2')(filters)

    filters = BatchNormalization(axis=bn_axis, name=f'{name}_bn_2')(filters)
    filters = Activation(activation=activation, name=f'{name}_{activation}_2')(filters)

    filters = Conv2D(256, 1,
                     padding='same',
                     data_format=data_format,
                     name=f'{name}_conv2d_3')(filters)
    filters = BatchNormalization(axis=bn_axis, name=f'{name}_bn_3')(filters)
    if add_residual:
        residual = Conv2D(256, 1,
                          padding='same',
                          data_format=data_format,
                          name=f'{name}_residual_conv2d')(residual)

        residual = BatchNormalization(axis=bn_axis, name=f'{name}_residual_bn')(residual)
        filters = Add(name=f'{name}_residual_add')([filters, residual])

    filters = Activation(activation=activation, name=f'{name}_{activation}_3')(filters)
    return filters


def hr_net(input_shape=(512, 512, 3),
           num_classes=20,
           activation='relu',
           data_format='channels_last',
           initial_channels=32,
           up_sample_method='bilinear',
           fusion_method='add',
           name='HRNet'):

    assert up_sample_method in ['nearest', 'bilinear', 'conv2d_transpose']
    assert fusion_method in ['add', 'concat']

    ini_inputs = Input(shape=input_shape)
    bn_axis = 3 if data_format == 'channels_last' else 1

    # Stage 1
    # stem
    x = Conv2D(64, 3,
               padding='same',
               strides=2,
               data_format=data_format)(ini_inputs)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation=activation)(x)

    for i in range(4):
        bottle_name = f'bottleneck_{i}'
        x = bottleneck(x, add_residual=i == 0, name=bottle_name)

    x = Conv2D(initial_channels, 3, padding='same', data_format=data_format)(x)
    r1_x, r2_x = exchange_unit([x],
                               down_sample=True,
                               activation=activation,
                               data_format=data_format,
                               fusion_method=fusion_method,
                               name='stage1_exchangeunit_1')

    # Stage 2
    # 1 resolution subnetwork
    r1_x = basic_block(r1_x, activation=activation, data_format=data_format, name='stage2_1r_basic_block_1')

    # 2 resolution subnetwork
    r2_x = basic_block(r2_x, activation=activation, data_format=data_format, name='stage2_2r_basic_block_1')

    r1_x, r2_x, r3_x = exchange_unit([r1_x, r2_x],
                                     down_sample=True,
                                     activation=activation,
                                     data_format=data_format,
                                     fusion_method=fusion_method,
                                     name='stage2_exchangeunit_2')

    # Stage 3
    num_iter = 4
    for i in range(num_iter):
        r1_x = basic_block(r1_x, activation=activation, data_format=data_format, name=f'stage3_1r_basic_block_{i + 1}')
        r2_x = basic_block(r2_x, activation=activation, data_format=data_format, name=f'stage3_2r_basic_block_{i + 1}')
        r3_x = basic_block(r3_x, activation=activation, data_format=data_format, name=f'stage3_3r_basic_block_{i + 1}')

        if i != num_iter - 1:
            r1_x, r2_x, r3_x = exchange_unit([r1_x, r2_x, r3_x],
                                             down_sample=False,
                                             activation=activation,
                                             data_format=data_format,
                                             fusion_method=fusion_method,
                                             name=f'stage3_exchangeunit_{i + 1}')

        else:
            r1_x, r2_x, r3_x, r4_x = exchange_unit([r1_x, r2_x, r3_x],
                                                   down_sample=True,
                                                   activation=activation,
                                                   data_format=data_format,
                                                   fusion_method=fusion_method,
                                                   name=f'stage3_exchangeunit_{i + 1}')

    # Stage 4
    num_iter = 2
    for i in range(num_iter):
        r1_x = basic_block(r1_x, activation=activation, data_format=data_format, name=f'stage4_1r_basic_block_{i + 1}')
        r2_x = basic_block(r2_x, activation=activation, data_format=data_format, name=f'stage4_2r_basic_block_{i + 1}')
        r3_x = basic_block(r3_x, activation=activation, data_format=data_format, name=f'stage4_3r_basic_block_{i + 1}')
        r4_x = basic_block(r4_x, activation=activation, data_format=data_format, name=f'stage4_rr_basic_block_{i + 1}')

        r1_x, r2_x, r3_x, r4_x = exchange_unit([r1_x, r2_x, r3_x, r4_x],
                                               down_sample=False,
                                               activation=activation,
                                               data_format=data_format,
                                               fusion_method=fusion_method,
                                               name=f'stage4_exchangeunit_{i + 1}')

    # Stage 4 final output
    r2_x = UpSampling2D(size=(2, 2),
                        interpolation='bilinear',
                        data_format=data_format,
                        name=f'stage4_2r_output_upsample2d')(r2_x)
    r3_x = UpSampling2D(size=(4, 4),
                        interpolation='bilinear',
                        data_format=data_format,
                        name=f'stage4_3r_output_upsample2d')(r3_x)
    r4_x = UpSampling2D(size=(8, 8),
                        interpolation='bilinear',
                        data_format=data_format,
                        name=f'stage4_4r_output_upsample2d')(r4_x)

    x = Concatenate(axis=bn_axis)([r1_x, r2_x, r3_x, r4_x])
    n_channels = x.shape[bn_axis]

    if up_sample_method is not 'conv2d_transpose':
        x = UpSampling2D(interpolation=up_sample_method, data_format=data_format)(x)
        x = Conv2D(n_channels, 3, padding='same', data_format=data_format)(x)
    else:
        x = Conv2DTranspose(n_channels, 3, strides=2, padding='same', data_format=data_format)(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation=activation)(x)

    x = Conv2D(num_classes, 1, padding='same')(x)

    if num_classes > 1:
        x = Activation('softmax', dtype='float32')(x)
    else:
        x = Activation('sigmoid', dtype='float32')(x)

    model = Model(inputs=ini_inputs, outputs=x, name=name)
    return model
