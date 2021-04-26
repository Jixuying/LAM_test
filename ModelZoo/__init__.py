import os
import torch

MODEL_DIR = '/content/LAM_test/ModelZoo/models/my_model'
# MODEL_DIR = '/home/disk1/jixuying/pycharm/毕设/my_complex_ChannelNet/LAM_Demo/ModelZoo/models/my_model'

NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'SRCNNpara_1218_rx_1sir_5_real_channelnet',
    'SRCNNpara_1218_rx_1sir_5_complex_channelnet',
    'SRCNNpara_1218_rx_1sir_n10_real_channelnet',
    'SRCNNpara_1218_rx_1sir_n10_complex_channelnet',
    'DNCNNpara_1218_rx_1sir_5_real_channelnet',
    'DNCNNpara_1218_rx_1sir_5_complex_channelnet',
    'NewNetpara_1218_rx_1sir_5_real_channelnet',
    'NewNetpara_1218_rx_1sir_5_complex_channelnet'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'SRCNNpara_1218_rx_1sir_5_real_channelnet': {
        'Base': 'SRCNNpara_1218_rx_1sir_5_real_channelnet.pth',
    },
    'SRCNNpara_1218_rx_1sir_5_complex_channelnet': {
        'Base': 'SRCNNpara_1218_rx_1sir_5_complex_channelnet.pth',
    },
    'SRCNNpara_1218_rx_1sir_n10_real_channelnet': {
        'Base': 'SRCNNpara_1218_rx_1sir_n10_real_channelnet.pth',
    },
    'SRCNNpara_1218_rx_1sir_n10_complex_channelnet': {
        'Base': 'SRCNNpara_1218_rx_1sir_n10_complex_channelnet.pth',
    },
    'DNCNNpara_1218_rx_1sir_5_real_channelnet': {
        'Base': 'DNCNNpara_1218_rx_1sir_5_real_channelnet.pth',
    },
    'DNCNNpara_1218_rx_1sir_5_complex_channelnet': {
        'Base': 'DNCNNpara_1218_rx_1sir_5_complex_channelnet.pth',
    },
    'NewNetpara_1218_rx_1sir_5_real_channelnet': {
        'Base': 'NewNetpara_1218_rx_1sir_5_real_channelnet.pth',
    },
    'NewNetpara_1218_rx_1sir_5_complex_channelnet': {
        'Base': 'NewNetpara_1218_rx_1sir_5_complex_channelnet.pth',
    }
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print('Getting SR Network'+str(model_name))
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

        elif model_name == 'SRCNNpara_1218_rx_1sir_5_real_channelnet':
            from .NN.SRCNN_real import SRCNN_Net
            net = SRCNN_Net()

        elif model_name == 'SRCNNpara_1218_rx_1sir_5_complex_channelnet':
            from .NN.SRCNN_complex import SRCNN_ComplexNet
            net = SRCNN_ComplexNet()
            
        elif model_name == 'SRCNNpara_1218_rx_1sir_n10_real_channelnet':
            from .NN.SRCNN_real import SRCNN_Net
            net = SRCNN_Net()

        elif model_name == 'SRCNNpara_1218_rx_1sir_n10_complex_channelnet':
            from .NN.SRCNN_complex import SRCNN_ComplexNet
            net = SRCNN_ComplexNet()
            
        elif model_name == 'DNCNNpara_1218_rx_1sir_5_real_channelnet':
            from .NN.DNCNN_real import DNCNN_Net
            net = DNCNN_Net()

        elif model_name == 'DNCNNpara_1218_rx_1sir_5_complex_channelnet':
            from .NN.DNCNN_complex import DNCNN_ComplexNet
            net = DNCNN_ComplexNet()
            
        elif model_name == 'DNCNNpara_1218_rx_1sir_n10_real_channelnet':
            from .NN.DNCNN_real import DNCNN_Net
            net = DNCNN_Net()

        elif model_name == 'DNCNNpara_1218_rx_1sir_n10_complex_channelnet':
            from .NN.DNCNN_complex import DNCNN_ComplexNet
            net = DNCNN_ComplexNet()
            
        elif model_name == 'NewNetpara_1218_rx_1sir_5_real_channelnet':
            from .NN.NewNet_real import NewNet_Net
            net = NewNet_Net(growthRate=12, depth=11, reduction=0.5,
                                               bottleneck=False, nClasses=10)

        elif model_name == 'DNCNNpara_1218_rx_1sir_5_complex_channelnet':
            from .NN.NewNet_complex import NewNet_ComplexNet
            net = NewNet_ComplexNet(growthRate=6, depth=11, reduction=0.5,
                               bottleneck=False, nClasses=10)

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print('Loading model'+str(state_dict_path)+' for '+str(model_name)+' network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict)
    return net




