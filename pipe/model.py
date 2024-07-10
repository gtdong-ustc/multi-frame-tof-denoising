import sys
sys.path.insert(0, './network/')

##################
## Method
from network.dear_kpn_no_rgb_DeepToF import dear_kpn_no_rgb_DeepToF
from network.dear_kpn_no_rgb import dear_kpn_no_rgb
from network.sample_pyramid_add_kpn import sample_pyramid_add_kpn
from network.pyramid_corr_multi_frame_denoising import pyramid_corr_mask_multi_frame_denoising
##################

NETWORK_NAME = {
    'dear_kpn_no_rgb_DeepToF': dear_kpn_no_rgb_DeepToF,
    'dear_kpn_no_rgb': dear_kpn_no_rgb,
    'sample_pyramid_add_kpn': sample_pyramid_add_kpn,
    'pyramid_corr_multi_frame_denoising': pyramid_corr_mask_multi_frame_denoising,
}

ALL_NETWORKS = dict(NETWORK_NAME)


def get_network(name, x, flg, regular, batch_size, range):
    """
    this function is used to selected the network
    :param name: network name
    :param x: network input, such as depth, amplitude, raw measurement, 
    :param flg: Indicates whether the code is in training mode
    :param regular: Regularization parameter(not be used)
    :param batch_size: 
    :param range: kernel Deformable range(used in deformable kernel network)
    :return: 
    """
    if name not in NETWORK_NAME.keys():
        print('Unrecognized network, pick one among: {}'.format(ALL_NETWORKS.keys()))
        raise Exception('Unknown network selected')
    selected_network = ALL_NETWORKS[name]
    return selected_network(x, flg, regular, batch_size, range)
