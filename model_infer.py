import argparse
import os
import time
from matplotlib.pyplot import draw
import onnx
import onnxruntime
from onnxsim import simplify
from tqdm import tqdm
import torch
import torch.onnx
from torch import nn
import numpy as np

from ghostnet_model import BranchyGhostnet
from load import get_load_data
from utils import LabelSmoothingLossCanonical


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Inference settings
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--val_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')

    parser.add_argument('--use_cuda', action="store_true", help='use cuda')

    parser.add_argument('--use_benchmark', action="store_true", help='use benchmark')

    args = parser.parse_args()
    return args

def to_onnx(model, input_size, onnx_path, device, test_in=None, batch_size=1):
    if test_in is None:
        x = torch.randn(batch_size, *input_size).to(device)
    else:
        x = test_in

    src_model = torch.jit.script(model)
    print("PRINT PYTORCH MODEL SCRIPT")
    print(src_model.graph, "\n")
    ex_out = src_model(x)
    #ex_out = model(x)

    torch.onnx.export(
        model,
        x,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        example_outputs=ex_out,
        input_names=['input'],
        output_names=['output']
    )

    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    assert check, "Simplified onnx model can not be validated"

    onnx.save(onnx_model_simp, onnx_path)
    return onnx_path

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    model_folder = os.path.join(os.getcwd(), f'branchynet_human_act_1.3_se')

    if args.use_cuda:
        device = torch.device('cuda')
        print(
            f'Using GPU: {torch.cuda.get_device_name()}, id: {torch.cuda.current_device()}')
        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            print('Using cudnn.benchmark.')
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')


    print("############setup_dataloader (START)#############")
    [
        train_dataloader,
        val_dataloader,
        train_data_num,
        val_data_num,
        each_act_num,
    ] = get_load_data(args.train_batch_size, args.val_batch_size)
    person_classes = ["empty", "one", "two"]
    act_classes = ["sit", "stand", "walk", "sitdown", "standup"]
    print("############setup_dataloader (END)##############")

    #model.set_fast_infer_mode()

   
    # batch_size = 1
    # input_size = [3, 224, 224]
    # onnx_path = os.path.join(os.getcwd(), f'onnx')

    # if not os.path.exists(onnx_path):
    #     os.makedirs(onnx_path)

    # onnx_file = os.path.join(onnx_path, 'BranchyGhostnet.onnx')

    # print("SAVE MODEL TO ONNX: ", onnx_path)
    # onnx_model_simp = to_onnx(model, input_size, onnx_file, device, batch_size=batch_size)
    # print("Simplify onnx model")
    
    # ort_session = onnxruntime.InferenceSession(onnx_model_simp)
    # val_iter = iter(val_dataloader)

    # torch_early_exit_time = []
    # torch_normal_exit_time = []

    # onnx_early_exit_time = []
    # onnx_normal_exit_time = []
    # i = 0
    # for _ in range(2300):
    #     img, label = val_iter.next()
    #     img, label = img.to(device), label.to(device)
    #     human_label, act_label = label[:, :3], label[:, 3:]
    #     #torch.cuda.synchronize()
    #     start = time.time()
    #     human_label_hat, act_label_hat = model(img)
    #     #torch.cuda.synchronize()
    #     end = time.time()
    #     _, human_idx = torch.max(nn.functional.softmax(human_label_hat, dim=-1), dim=-1)
    #     _, act_idx = torch.max(nn.functional.softmax(act_label_hat, dim=-1), dim=-1)
    #     if human_idx.item() == 1:
    #         torch_normal_exit_time.append(end - start)
    #         #print(f"true result: person {person_classes[torch.nonzero(human_label.squeeze()).item()]} act {act_classes[torch.nonzero(act_label.squeeze()).item()]}")
    #     else:
    #         torch_early_exit_time.append(end - start)
    #         #print(f"true result: person {person_classes[torch.nonzero(human_label.squeeze()).item()]}")
    #     #print(f"torch predict result: person {person_classes[human_idx.item()]} act {act_classes[act_idx.item()]}")
    #     #torch.cuda.synchronize()
    #     start = time.time()
    #     ort_input = {ort_session.get_inputs()[0].name: to_numpy(img)}
    #     ort_human_output, ort_act_output = ort_session.run(None, ort_input)
    #     #torch.cuda.synchronize()
    #     end = time.time()
    #     human_onnx_idx = np.argmax(softmax(ort_human_output), axis=-1)
    #     act_onnx_idx = np.argmax(softmax(ort_act_output), axis=-1)
    #     #print(f"onnx predict result: person {person_classes[human_onnx_idx.item()]} act {act_classes[act_onnx_idx.item()]}")
    #     if human_onnx_idx.item() == 1:
    #         onnx_normal_exit_time.append(end - start)
    #         #print(f"true result: person {person_classes[torch.nonzero(human_label.squeeze()).item()]} act {act_classes[torch.nonzero(act_label.squeeze()).item()]}")
    #     else:
    #         i += 1
    #         onnx_early_exit_time.append(end - start)
    #     #print(f"onnx Inference time: {end - start:.3f}")
    #     np.testing.assert_allclose(to_numpy(human_label_hat.squeeze()), ort_human_output[0], rtol=1e-3, atol=1e-5)
    # print(f"torch early exit time {sum(torch_early_exit_time) / len(torch_early_exit_time):3f} torch normal exit time {sum(torch_normal_exit_time) / len(torch_normal_exit_time):.3f}")
    # print(f"onnx early exit time {sum(onnx_early_exit_time) / len(onnx_early_exit_time):3f} num {i} onnx normal exit time {sum(onnx_normal_exit_time) / len(onnx_normal_exit_time):.3f}")
    #plot_confuse_matrix(y, y_hat, range(len(person_classes)), result_folder)
