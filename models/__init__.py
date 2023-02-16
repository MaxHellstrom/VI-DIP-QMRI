from .skip import skip


def get_net(
    input_depth,
    NET_TYPE,
    pad,
    upsample_mode,
    n_channels=3,
    act_fun="ReLU",
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    downsample_mode="stride",
    bayes=False,
    dropout_mode_down="2d",
    dropout_p_down=0.5,
    dropout_mode_up="2d",
    dropout_p_up=0.5,
    dropout_mode_skip="None",
    dropout_p_skip=0.5,
    dropout_mode_output="None",
    dropout_p_output=0.5,
    need_output_act=False,
):

    if NET_TYPE == "skip":
        net = skip(
            num_input_channels=input_depth,
            num_output_channels=n_channels,
            num_channels_down=[skip_n33d] * num_scales
            if isinstance(skip_n33d, int)
            else skip_n33d,
            num_channels_up=[skip_n33u] * num_scales
            if isinstance(skip_n33u, int)
            else skip_n33u,
            num_channels_skip=[skip_n11] * num_scales
            if isinstance(skip_n11, int)
            else skip_n11,
            need_sigmoid=False,
            need_bias=True,
            pad=pad,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,
            act_fun=act_fun,
            bayes=bayes,
            dropout_mode_down=dropout_mode_down,
            dropout_p_down=dropout_p_down,
            dropout_mode_up=dropout_mode_up,
            dropout_p_up=dropout_p_up,
            dropout_mode_skip=dropout_mode_skip,
            dropout_p_skip=dropout_p_skip,
            dropout_mode_output=dropout_mode_output,
            dropout_p_output=dropout_p_output,
            need_output_act=need_output_act,
        )

    else:
        assert False

    return net
