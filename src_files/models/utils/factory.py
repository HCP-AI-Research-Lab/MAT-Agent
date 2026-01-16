import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args,load_head=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = 'tresnet_l' # args.model_name.lower()
    print("Using tresnet_l and ignore arg from command!!!!")

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    args.use_ml_decoder = True
    args.num_of_groups = -1
    args.decoder_embedding = 768
    args.zsl = 0
    print("使用ML Decoder和默认参数(num_of_groups, decoder_embedding, zsl), 忽略命令行参数！！！！")
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    ####################################################################################
    # loading pretrain model
    model_path = '' # args.model_path
    print("Use tresnet_l.pth as pretrain model")
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            # request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict())}
            model.load_state_dict(filtered_dict, strict=False)

    return model
