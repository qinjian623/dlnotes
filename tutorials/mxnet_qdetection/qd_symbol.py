import mxnet as mx


def conv_block(block_name, bottom, filters,
               kernel_size=(3, 3), pad=(1, 1), stride=1):
    conv = mx.symbol.Convolution(data=bottom,
                                 kernel=kernel_size,
                                 pad=pad,
                                 num_filter=filters,
                                 name=block_name+"_conv")
    bn = mx.symbol.BatchNorm(data=conv,
                             name=block_name+"_bn")
    relu = mx.symbol.LeakyReLU(data=bn,
                               name=block_name+"_leakyrelu")
    return relu


def pooling_layer(block_name, bottom):
    return mx.symbol.Pooling(data=bottom,
                             pool_type="max",
                             kernel=(2, 2),
                             stride=(2, 2),
                             name=block_name+"_pooling")


def upsample_layer(block_name, bottom, upsample_ratio):
    return mx.symbol.UpSampling(data=bottom,
                                scale=upsample_ratio,
                                sample_type="bilinear",
                                # num_filter=32,
                                num_args=2,
                                name=block_name+"_upsampling")


def dual_conv_block(block_name, bottom, filters, with_pool=True):
    c0 = conv_block(block_name+"_c0", bottom, filters[0])
    c1 = conv_block(block_name+"_c1", c0, filters[1])
    if with_pool:
        p = pooling_layer(block_name+"_p", c1)
        return p, c1
    else:
        return c1, c1


def build_network():
    data = mx.symbol.Variable('data')
    # 1/2
    c0 = conv_block('C0', data, 16,
                    kernel_size=(5, 5),
                    pad=(2, 2),
                    stride=2)
    # 1/4
    p0 = pooling_layer('p0', c0)
    # 1/8
    dc0, f1_4 = dual_conv_block("dc0", p0, (32, 32))
    # 1/16
    dc1, f1_8 = dual_conv_block("dc1", dc0, (64, 64))
    # 1/32
    dc2, f1_16 = dual_conv_block("dc2", dc1, (128, 128))
    # 1/64
    dc3, f1_32 = dual_conv_block("dc3", dc2, (256, 256))
    dc4, _ = dual_conv_block("dc4", dc3, (512, 512), with_pool=False)
    # Adaptive
    ada_64 = conv_block("ada_1/64", dc4, 16)
    ada_32, _ = dual_conv_block("ada_1/32",
                                f1_32, (32, 16),
                                with_pool=False)
    ada_16, _ = dual_conv_block("ada_1/16",
                                f1_16, (32, 16),
                                with_pool=False)
    ada_8, _ = dual_conv_block("ada_1/8",
                               f1_8, (32, 16),
                               with_pool=False)
    ada_4, _ = dual_conv_block("ada_1/4",
                               f1_4, (32, 16),
                               with_pool=False)
    # Upsample
    u_16 = upsample_layer("x16", ada_64, 16)
    u_8 = upsample_layer("x8", ada_32, 8)
    u_4 = upsample_layer("x4", ada_16, 4)
    u_2 = upsample_layer("x2", ada_8, 2)
    # Fusion
    fusion = mx.symbol.Concat(u_16, u_8, u_4, u_2, ada_4)
    sep, _ = dual_conv_block("xxx", fusion, (16, 16), with_pool=False)

    # Cls branch
    cls_b0 = conv_block('cls_branch0', sep, 16)
    cls = mx.symbol.Convolution(data=cls_b0,
                                kernel=(3, 3),
                                pad=(1, 1),
                                num_filter=3,
                                name="cls_conv")
    softmax = mx.symbol.SoftmaxOutput(data=cls,
                                      name='softmax')
    # Reg branch
    reg_b0 = conv_block('reg_branch0', sep, 32)
    reg = mx.symbol.Convolution(data=reg_b0,
                                kernel=(3, 3),
                                pad=(1, 1),
                                num_filter=4,
                                name="reg_conv")
    reg_loss = mx.symbol.LinearRegressionOutput(data=reg, name="reg_loss")
    # Group loss
    group_loss = mx.symbol.Group([softmax, reg_loss])
