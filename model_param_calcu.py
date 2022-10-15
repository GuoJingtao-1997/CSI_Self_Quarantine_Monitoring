'''
Author: Guo Jingtao
Date: 2022-06-03 15:55:57
LastEditTime: 2022-06-17 16:41:37
LastEditors: Guo Jingtao
Description: 
FilePath: /undefined/Users/guojingtao/Documents/CSI_DataProcess/HumanActivityCode/model_calcu.py

'''

import torch

from thop import profile, clever_format

# from torchprofile import profile_macs

from fvcore.nn import FlopCountAnalysis, parameter_count_table

from model import EfficientNet

from torchstat import stat

from ghostmodel import ghostnet

from ghostnet_model import BranchyGhostnet

from time_classifier import *

if __name__ == "__main__":

    # model = ConvBiLSTMClassifier(
    #     encoder_features=234, dropout=0.2
    # )  # MACs 569.166M, Params 3.653M 3652744 569.166M
    # model = AttentionBiLSTMClassifier(
    #     emb_features=234
    # )  # MACs 210.243M, Params 700.808K 700808
    # model = LSTMClassifier(
    #     encoder_features=234, dropout=0.2
    # )  # MACs 105.122M, Params 350.408K 350408
    # model = TUNet()  # MACs 234.361M, Params 5.215M 5215176
    # model = FCNBaseline(in_channels=234)
    # model = MLSTM_FCN(c_in=234)
    # model = GRU_FCN(234)
    # model = BiLSTMClassifier(encoder_features=234)  # MACs 303.941M, Params 1.436M 1435528
    # model = gMLP(c_in=234, seq_len=300)

    # ensemble_num = 5
    # InceptionTime = {}
    # for i in range(ensemble_num):
    # model = InceptionModel(
    #     c_in=234
    # )  # MACs = 704.646M, Params = 2.353M 2503720 MACs 704.646M MACs single: MACs 140.929M Params 470.664K
    #     InceptionTime[i] = model

    # model_input = torch.randn(1, 1, 234, 300)
    # macs = params = 0
    # for i in range(ensemble_num):
    #     mac, param = profile(InceptionTime[i], (model_input,), verbose=False)
    #     macs += mac
    #     params += param
    #     # params += sum(p.numel() for p in InceptionTime[i].parameters())
    # macs, params = clever_format([macs, params], "%.3f")

    # print("MACs", macs)
    # print("Params", params)

    model = BranchyGhostnet(
        branch_classes=3, final_classes=5, width=1.3
    )  # MACs 370.920122 M, Params 7.1M 7143996
    # model = EfficientNet.from_name(
    #     'efficientnet-b2', image_size=(234, 300)
    # )  # MACs 1014.851144 M, Params 7.7M 7712266
    model_input = torch.randn(1, 3, 234, 300)
    macs, params = profile(model, (model_input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs", macs)
    print("Params", params)
    # macs = profile_macs(model, model_input)
    # print(macs / 1e6)
    # flops = FlopCountAnalysis(model, model_input)
    # print(flops.total() / 1e6)
    # print("Params", parameter_count_table(model))
    # stat(model, (3, 234, 300))
    # total_num = sum(p.numel() for p in model.parameters())
    # print(params)
