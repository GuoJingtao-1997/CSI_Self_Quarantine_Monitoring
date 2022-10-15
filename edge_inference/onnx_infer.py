from asyncio import subprocess
import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.onnx
import paho.mqtt.client as paho  # mqtt library
import telepot
import cv2
import numpy as np
import onnxruntime
import pandas as pd
import csiread
import matplotlib.pyplot as plt

from ghostnet_model import BranchyGhostnet
from load import get_load_data
from utils import (
    Accumulator,
    Evaluations,
    pred_label_cpu,
    softmax,
)


ACCESS_TOKEN = 'Thingsboard Token of the device'
broker = "demo.thingsboard.io"  # host name
port = 1883  # data listening port
nullsubcarriers = np.array(
    [
        x + 128
        for x in [-128, -127, -126, -125, -124, -123, -1, 0, 1, 123, 124, 125, 126, 127]
    ]
)
pilotsubcarriers = np.array([x + 128 for x in [-103, -75, -39, -11, 11, 39, 75, 103]])
no_subcarriers = 256 - len(nullsubcarriers) - len(pilotsubcarriers)
csipacket = xlim = 300


def on_publish(client, userdata, result):  # create function for callback
    print("data published to thingsboard \n")
    pass


client1 = paho.Client("control1")  # create client object
client1.on_publish = on_publish  # assign function to callback
client1.username_pw_set(ACCESS_TOKEN)  # access token from thingsboard device
client1.connect(broker, port, keepalive=60)  # establish connection
bot = telepot.Bot('the telebot token')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Inference settings

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=1,
        metavar='N',
        help='input batch size for testing (default: 64)',
    )

    parser.add_argument('--use_benchmark', action="store_true", help='use benchmark')

    args = parser.parse_args()
    return args


def test(model, test_dataloader, person_classes, act_classes, device):
    """test phase"""
    metric = Accumulator([[]], 4)
    # metric_loss = Accumulator([0.0], 2)
    # test_size = len(test_dataloader) * 60
    model.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(
            test_dataloader, total=len(test_dataloader), desc='test'
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            human_labels, act_labels = labels[:, :3], labels[:, 3:]
            # imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
            human_labels_hat, act_labels_hat = model(imgs)
            # branch_loss = criterion(human_labels_hat, torch.max(human_labels, dim=-1)[1])
            # final_loss = criterion(act_labels_hat, torch.max(act_labels, dim=-1)[1])
            metric.add(
                torch.max(human_labels, dim=-1)[1].detach().cpu().numpy().tolist(),
                pred_label_cpu(human_labels_hat),
                torch.max(act_labels, dim=-1)[1].detach().cpu().numpy().tolist(),
                pred_label_cpu(act_labels_hat),
            )
            # metric_loss.add(
            #     branch_loss,
            #     final_loss
            # )

    eval_human_test = Evaluations(metric[0], metric[1], person_classes)
    eval_act_test = Evaluations(metric[2], metric[3], act_classes)
    print(eval_human_test)
    print(eval_act_test)
    return eval_human_test, eval_act_test, metric[0], metric[1], metric[2], metric[3]


def running_mean(x: pd.DataFrame, N: int) -> np.array:
    return x.rolling(window=N, min_periods=1, center=True, axis=1).mean().to_numpy()


def realtime_csi2img(packets, pcap_path, img_path):
    subprocess.check_output(
        f'sudo tcpdump -i wlan0 dst port 5500 -vv -w {pcap_path} -c {packets}',
        shell=True,
    )

    csidata = csiread.NexmonPull46(pcap_path, '43455c0', 80, if_report=False)
    csidata.read()
    csi = pd.DataFrame(10 * np.log10(np.abs(csidata.csi)))  # amplitude
    # csi = pd.DataFrame((np.angle(csidata.csi) + 2 * math.pi) % (2 * math.pi))  # phase

    csi = csi.drop(columns=nullsubcarriers)
    csi = csi.drop(columns=pilotsubcarriers)
    # csi = np.transpose(np.array(csi.drop(columns=pilotsubcarriers))[s[0] : s[1]])
    csi_rm = running_mean(csi, 10)
    csi = np.transpose(csi_rm)

    limits = [0, xlim, -(no_subcarriers / 2), no_subcarriers / 2 - 1]
    _, ax = plt.subplots()
    ax.imshow(csi, cmap="jet", extent=limits, aspect="auto")
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Amplitude (dBm)")
    # plt.xlabel(x_label)
    # plt.ylabel("Subcarrier Index")
    fig = plt.gcf()
    fig.set_size_inches(csipacket / 100, 2.34)  # output = 224 * 224 pixels
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis("off")

    plt.savefig(img_path, pad_inches=0.0, dpi=100)  # dpi = 100
    plt.close('all')
    # plt.show()


def image_preprocess(img_path):
    img = cv2.imread(img_path)
    img = img[..., ::-1]
    # img = cv2.resize(img, (224, 224))
    img = img - (130.23, 184.21, 113.35)  # The pixel mean of the CSI image
    img = img * (0.014, 0.018, 0.014)  # The pixel standard deviation of the CSI image
    return img.astype(np.float32)[np.newaxis, :].transpose(0, 3, 2, 1)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # args = add_args(parser)
    folder = os.path.join(os.getcwd(), f'RealTimeDetectImage')

    print("############setup_dataloader (START)#############")
    # [
    #     val_dataloader,
    #     val_data_num,
    #     each_act_num,
    # ] = get_load_data(args.val_batch_size)
    person_classes = [0, 1, 2]
    act_classes = ["sit", "stand", "walk", "sitdown", "standup"]
    print("############setup_dataloader (END)##############")

    if not os.path.exists(folder):
        os.makedirs(folder)

    # device = torch.device('cpu')

    # pre_model = torch.load(folder, map_location=device)

    # model = BranchyGhostnet(branch_classes=3, final_classes=5, width=1.3).to(device)

    # model.load_state_dict(pre_model)

    # model.set_fast_infer_mode()

    onnx_file = os.path.join(os.getcwd(), 'BranchyGhostnet.onnx')

    ort_session = onnxruntime.InferenceSession(onnx_file)

    pcap_path = os.path.join(folder, f'1.pcap')
    img_path = os.path.join(folder, f'1.jpg')

    # onnx_early_exit_time = []
    # onnx_normal_exit_time = []

    # time.sleep(3)
    while True:
        # time.sleep(1)
        realtime_csi2img(packets=csipacket, pcap_path=pcap_path, img_path=img_path)
        img = image_preprocess(img_path)
        # print(i)

        # torch.cuda.synchronize()
        # start = time.time()
        # human_label_hat, act_label_hat = model(img)
        # # # torch.cuda.synchronize()
        # # end = time.time()
        # _, human_idx = torch.max(nn.functional.softmax(human_label_hat, dim=-1), dim=-1)
        # _, act_idx = torch.max(nn.functional.softmax(act_label_hat, dim=-1), dim=-1)
        # people = person_classes[human_idx.item()]
        # act = act_classes[act_idx.item()]
        # if people == 0:
        #     act = "None"
        # if human_idx.item() == 1:
        # torch_normal_exit_time.append(end - start)
        # print(f"true result: person {person_classes[torch.nonzero(human_label.squeeze()).item()]} act {act_classes[torch.nonzero(act_label.squeeze()).item()]}")
        # else:
        #     # bot.sendMessage(
        #     #     5284329501,
        #     #     f'{datetime.date.today()} Warning! Violate the quarantine order. Current #People: {person_classes[human_idx.item()]}',
        #     # )
        #    torch_early_exit_time.append(end - start)
        ort_input = {ort_session.get_inputs()[0].name: img}
        ort_human_output, ort_act_output = ort_session.run(None, ort_input)
        human_onnx_idx = np.argmax(softmax(ort_human_output), axis=-1)
        act_onnx_idx = np.argmax(softmax(ort_act_output), axis=-1)
        people = person_classes[human_onnx_idx.item()]
        act = act_classes[act_onnx_idx.item()]
        if human_onnx_idx.item() != 1:
            act = 'None'
            bot.sendMessage(
                1234567891,  # the telebot chat id
                f'{datetime.date.today()} Warning! Violate the quarantine order. Current #People: {person_classes[human_onnx_idx.item()]}',
            )
        payload = "{"
        payload += f'"Activity":{act},'
        payload += f'"People":{people}'
        payload += "}"
        ret = client1.publish("v1/devices/me/telemetry", payload)
        # topic-v1/devices/me/telemetry
        print("Please check LATEST TELEMETRY field of the device")
        print(payload)
