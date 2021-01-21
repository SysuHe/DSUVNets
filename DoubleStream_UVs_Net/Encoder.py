import os
from resnet_v2 import resnet_v2_50, resnet_v2_50_nosam
from xceptions import xception_41, xception_41_nosam


# the alias mapping of the pool-like(not always be pooling layers, might caused by strides as well) layers
scope_table = {
    "resnet_v2_50": {"pool1": "conv1", "pool2": "block1_unit3_act1", "pool3": "block2_unit4_act1", "pool4": "act_encoder"},
    "resnet_v2_50_nosam": {"pool1": "ns_conv1", "pool2": "ns_block1_unit3_act1", "pool3": "ns_block2_unit4_act1", "pool4": "ns_act_encoder"},

    "xception_41": {"pool1": "entry_block2_sepconv3_act", "pool2": "entry_block3_sepconv3_act", "pool3": "entry_block4_sepconv3_act", "pool4": "middle_block16_sepconv3_bn"},
    "xception_41_nosam": {"pool1": "ns_entry_block2_sepconv3_act", "pool2": "ns_entry_block3_sepconv3_act", "pool3": "ns_entry_block4_sepconv3_act", "pool4": "ns_middle_block16_sepconv3_bn",
                          "pool5": "ns_exit_match_dim_act"},
}


def build_encoder(input_shape,
                  encoder_name,
                  encoder_weights=None,
                  weight_decay=1e-4,
                  kernel_initializer="he_normal",
                  bn_epsilon=1e-3,
                  bn_momentum=0.99):
    """ the main api to build a encoder.
    :param input_shape: tuple, i.e., (height, width. channel).
    :param encoder_name: string, name of the encoder, refer to 'scope_table' above.
    :param encoder_weights: string, path of the weight, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder_name = encoder_name.lower()

    if encoder_name=="resnet_v2_50":
        encoder = resnet_v2_50(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_50_nosam":
        encoder = resnet_v2_50_nosam(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer,bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    elif encoder_name == "xception_41":
        encoder = xception_41(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name == "xception_41_nosam":
        encoder = xception_41_nosam(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    else:
        raise ValueError("Invalid encoder name")

    if encoder_weights is not None and os.path.exists(encoder_weights):
        encoder.load_weights(encoder_weights)

    return encoder