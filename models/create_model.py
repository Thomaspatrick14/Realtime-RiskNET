import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnext_3d as resnext

class MaskNet(nn.Module):
    def __init__(self, modality, args, backbone):
        super(MaskNet, self).__init__()
        if modality == 'flow':
            self.layers = [0, 1]
            self.mode = 'flow'
        elif modality == 'depth':
            self.layers = [2]
            self.mode = 'depth'
        elif modality == 'rgb':
            self.layers = [3, 4, 5]
            self.mode = 'rgb'
        else:
            raise ValueError("Modality should be flow, depth or rgb but is ", modality)
        
        print("\nLoading SingleNet model with as only input {} data".format(modality))
        
        self.out_features = 4096    #Last layer size of ResNext and the number of input to the FC layer

        if args.batch_norm:
            self.modality_layer = nn.Sequential(
                nn.Conv3d(len(self.layers), args.conv1_out, kernel_size=3, padding=(1, 1, 1), bias=True),
                nn.BatchNorm3d(args.conv1_out)
            )
        else:
            self.modality_layer = nn.Conv3d(len(self.layers), args.conv1_out, kernel_size=3, padding=(1, 1, 1), bias=True)

        self.relevance_layer = nn.Conv3d(1, args.conv1_out, kernel_size=3, padding=(1, 1, 1), bias=True)

        if backbone == 'ResNext101':
            self.model_flow = resnext.resnext101(input_depth=args.conv1_out)

        elif backbone == 'ResNext50':
            self.model_flow = resnext.resnext50(input_depth=args.conv1_out)

        elif backbone == 'ResNext24':
            self.model_flow = resnext.resnext24(input_depth=args.conv1_out)

        elif backbone == 'ResNext18':
            self.model_flow = resnext.resnext18(input_depth=args.conv1_out)

        else:
            raise ValueError(f"Backbone {backbone} is not implemented")
        print(f"Loaded the {backbone} backbone\n\n")

        self.prediction_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.out_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, flow, depth, rgb, mask):
        if self.mode == 'flow':
            modality = flow
        elif self.mode == 'depth':
            modality = depth
        else:
            modality = rgb

        modality = self.modality_layer(modality)
        relevance = self.relevance_layer(mask)
        x = modality * relevance
        x, c = self.model_flow(x)
        self.out_features = c
        x = self.prediction_layer(x)
        return x

class OnlyMaskNet(nn.Module):
    def __init__(self, backbone):
        super(OnlyMaskNet, self).__init__()
        print("\nLoading OnlyMaskNet model with only masks as input data")

        self.out_features = 4096    #Last layer size of ResNext and the number of input to the FC layer

        if backbone == 'ResNext101':
            self.model_flow = resnext.resnext101(input_depth=1)

        elif backbone == 'ResNext50':
            self.model_flow = resnext.resnext50(input_depth=1)

        elif backbone == 'ResNext24':
            self.model_flow = resnext.resnext24(input_depth=1)

        elif backbone == 'ResNext18':
            self.model_flow = resnext.resnext18(input_depth=1)

        else:
            raise ValueError(f"Backbone {backbone} is not implemented")
        print(f"Loaded the {backbone} backbone\n\n")

        self.prediction_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.out_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, mask):
        x = self.model_flow(mask)
        x = self.prediction_layer(x)
        return x
    

def get_model(args):
    """
    Returns a model based on the selected arguments.
    If CUDA is available, it will load the model onto the GPU.

    Args:
        args: Namespace with the arguments to be used in the model selection. The required arguments are model_inputs,
              n_backbones, backbone.
              model_inputs: must be a list of the inputs to be used by the model (for instance ['flow', 'rgb']).
              n_backbones: must be an int. Contains the number of backbones to be used. Can never be lower than the
                           length of model_inputs (as two backbones on one modality makes no sense)
              backbone: the backbone type which the model should use. Needs to be one of:
                        MobileNetV2, ResNext101, MobileNet2D, ResNext2D, MobileNet-LSTM, ResNext-LSTM.

    Returns:

    """
    model_inputs = args.input
    n_backbones = args.n_backbones
    backbone = args.backbone
    # input_modalities = len(model_inputs)
    # single backbone 3D convolution models
    if 'mask' in model_inputs:
        modalities = model_inputs.copy()
        modalities.remove('mask')
        if len(modalities) == 1 and n_backbones == 1 and backbone in ['ResNext101',  'ResNext50', 'ResNext24', 'ResNext18']:
            model = MaskNet(modalities[0], args, backbone=backbone)
        else:
            print('-'*79)
            print(f"!!!!!!!!!!WARNING!!!!!!!!!!\n"
                  f"Model will only train/infer on masks!!\n"
                  f"!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print('-' * 79)
            model = OnlyMaskNet(backbone=backbone)
    else:
        raise NotImplementedError("This configuration can't be loaded.")

    if torch.cuda.is_available():
        print("CUDA available: loading the RiskNet model on the GPU")
        model = model.cuda()
    return model