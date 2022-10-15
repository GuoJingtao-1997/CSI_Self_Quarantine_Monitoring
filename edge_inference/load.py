import csv
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ghostnet_model import BranchyGhostnet
from utils import setup_seed, get_mean_and_std

# csi_mean = [0.485, 0.456, 0.406]
# csi_std = [0.229, 0.224, 0.225]
csi_mean = [0.485, 0.456, 0.406]
csi_std = [0.2023, 0.1994, 0.2010]
human_path = os.path.join(os.getcwd(), r'Dataset_human_5days')
human_dirs = os.listdir(human_path)
header = [
    "image_id",
    "empty",
    "one",
    "two",
    "stand",
    "sit",
    "walk",
    "sitdown",
    "standup",
]
# header = ["image_id", "empty", "one", "two", "three", "four", "five", "sit", "A1", "A2", "B1", "B2", "C1", "C2"]
# header = ["image_id", "empty", "one", "sit", "A1", "A2", "B1", "B2"]
num_class = len(header) - 1


class CsiLabeledCsv:
    def __init__(self) -> None:

        self.act_list = {}

        for human_acts in human_dirs:
            if human_acts not in [
                "train_human.csv",
                "test_human.csv",
                "train_human_all.csv",
                "test_human_0206.csv",
                'train_human_0206.csv',
            ]:
                self.act_list[human_acts] = len(
                    os.listdir(os.path.join(human_path, human_acts))
                )

        self.train_csv = os.path.join(human_path, r'train_human.csv')
        self.train_all_csv = os.path.join(human_path, r'train_human_all.csv')
        self.test_0206_csv = os.path.join(human_path, r'test_human_0206.csv')
        self.train_0206_csv = os.path.join(human_path, r'train_human_0206.csv')
        self.test_csv = os.path.join(human_path, r'test_human.csv')

    def create_csv(self):

        with open(os.path.join(human_path, r'train_human.csv'), 'a') as train_csv:
            train_writer = csv.writer(train_csv)
            train_writer.writerow(header)
        with open(os.path.join(human_path, r'test_human.csv'), 'a') as test_csv:
            test_writer = csv.writer(test_csv)
            test_writer.writerow(header)
        with open(
            os.path.join(human_path, r'train_human_all.csv'), 'a'
        ) as train_all_csv:
            train_all_writer = csv.writer(train_all_csv)
            train_all_writer.writerow(header)
        with open(
            os.path.join(human_path, r'train_human_0206.csv'), 'a'
        ) as train_0206_csv:
            train_0206_writer = csv.writer(train_0206_csv)
            train_0206_writer.writerow(header)
        with open(
            os.path.join(human_path, r'test_human_0206.csv'), 'a'
        ) as test_0206_csv:
            test_0206_writer = csv.writer(test_0206_csv)
            test_0206_writer.writerow(header)

        for human_acts in human_dirs:
            label_list = [0] * num_class
            act_path = os.path.join(human_path, human_acts)
            imgs = os.listdir(act_path)
            total_imgs = len(imgs)
            human_acts_list = human_acts.split(',')

            for acts in human_acts_list:
                label_list[header.index(acts) - 1] = 1

            if human_acts != "one,falldown":
                train_image = int(0.8 * len(imgs))
            else:
                train_image = 560
            imgs = os.listdir(act_path)
            if human_acts in ["empty", "two,stand"]:
                for img in imgs:
                    i = int(os.path.splitext(img)[0])
                    if (
                        i <= 385
                        or 475 < i <= 860
                        or 950 < i <= 1340
                        or 1425 < i <= 1815
                    ):
                        # if i <= 2300:
                        with open(
                            os.path.join(human_path, r'train_human.csv'), 'a'
                        ) as train_csv:
                            train_writer = csv.writer(train_csv)
                            train_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    elif (
                        380 < i <= 475
                        or 855 < i <= 950
                        or 1330 < i <= 1425
                        or 1805 < i <= 1900
                    ):
                        # elif 2300 < i <= 2700:
                        with open(
                            os.path.join(human_path, r'test_human.csv'), 'a'
                        ) as test_csv:
                            test_writer = csv.writer(test_csv)
                            test_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    if i <= 1900:
                        with open(
                            os.path.join(human_path, r'train_human_all.csv'), 'a'
                        ) as train_all_csv:
                            train_all_writer = csv.writer(train_all_csv)
                            train_all_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    # if 3000 < i <= 3025 or 3050 < i :
                    #     with open(os.path.join(human_path, r'train_human_0206.csv'), 'a') as train_0206_csv:
                    #             train_0206_writer = csv.writer(train_0206_csv)
                    #             train_0206_writer.writerow(
                    #                 [os.path.join(act_path, img), *[label for label in label_list]])
                    # elif 3025 < i <= 3050:
                    #     with open(os.path.join(human_path, r'test_human_0206.csv'), 'a') as test_0206_csv:
                    #             test_0206_writer = csv.writer(test_0206_csv)
                    #             test_0206_writer.writerow(
                    #                 [os.path.join(act_path, img), *[label for label in label_list]])
            else:
                for img in imgs:
                    i = int(os.path.splitext(img)[0])
                    if (
                        i <= 345
                        or 425 < i <= 770
                        or 850 < i <= 1195
                        or 1275 < i <= 1620
                    ):
                        # if i <= 2050:
                        with open(
                            os.path.join(human_path, r'train_human.csv'), 'a'
                        ) as train_csv:
                            train_writer = csv.writer(train_csv)
                            train_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    elif (
                        340 < i <= 425
                        or 765 < i <= 850
                        or 1190 < i <= 1275
                        or 1615 < i <= 1700
                    ):
                        # elif 2050 < i <= 2400:
                        with open(
                            os.path.join(human_path, r'test_human.csv'), 'a'
                        ) as test_csv:
                            test_writer = csv.writer(test_csv)
                            test_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    if i <= 1700:
                        with open(
                            os.path.join(human_path, r'train_human_all.csv'), 'a'
                        ) as train_all_csv:
                            train_all_writer = csv.writer(train_all_csv)
                            train_all_writer.writerow(
                                [
                                    os.path.join(act_path, img),
                                    *[label for label in label_list],
                                ]
                            )
                    # if 2700 < i <= 2750 or 2760 < i:
                    #     with open(os.path.join(human_path, r'train_human_0206.csv'), 'a') as train_0206_csv:
                    #             train_0206_writer = csv.writer(train_0206_csv)
                    #             train_0206_writer.writerow(
                    #                 [os.path.join(act_path, img), *[label for label in label_list]])
                    # elif 2750 < i <= 2760:
                    #     with open(os.path.join(human_path, r'test_human_0206.csv'), 'a') as test_0206_csv:
                    #             test_0206_writer = csv.writer(test_0206_csv)
                    #             test_0206_writer.writerow(
                    #                 [os.path.join(act_path, img), *[label for label in label_list]])


def default_loader(path):
    return Image.open(path).convert("RGB")


class CsiLabeled(Dataset):
    def __init__(
        self, csv, transform=None, target_transform=None, loader=default_loader
    ) -> None:
        super().__init__()

        self.data_info = pd.read_csv(csv, header=None)
        self.imgs = np.asarray_chkfinite(self.data_info.iloc[1:, 0])
        self.labels = np.asarray_chkfinite(self.data_info.iloc[1:, 1:], dtype=float)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img_path = self.imgs[index]
        labels = torch.LongTensor(self.labels[index])

        img = (
            self.transform(self.loader(img_path))
            if self.transform is not None
            else self.loader(img_path)
        )

        return img, labels

    def __len__(self):
        return len(self.imgs)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class CsiUnLabeled(Dataset):
    def __init__(
        self, csv, transform=None, target_transform=None, loader=default_loader
    ) -> None:
        super().__init__()

        self.data_info = pd.read_csv(csv, header=None)
        self.imgs = np.asarray_chkfinite(self.data_info.iloc[1:, 0])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img_path = self.imgs[index]

        img_u1, img_u2 = (
            self.transform(self.loader(img_path))
            if self.transform is not None
            else self.loader(img_path)
        )

        return img_u1, img_u2

    def __len__(self):
        return len(self.imgs)


def get_dataloader_workers():
    """Use num processes to read the data."""
    return 8


def load_data_csi(val_batch_size, val_trainsform):

    setup_seed(42)
    csv_creator = CsiLabeledCsv()

    if 'train_human.csv' in os.listdir(human_path):
        # os.remove(os.path.join(act_path, "test_human.csv"))
        # os.remove(os.path.join(act_path, "train_human.csv"))
        pass
    else:
        csv_creator.create_csv()

    val_data = CsiLabeled(csv=csv_creator.test_csv, transform=val_trainsform)

    # test_0206_data = CsiLabeled(csv=csv_creator.test_0206_csv,
    #                       transform=val_trainsform)

    # train_all_data = CsiLabeled(csv=csv_creator.train_all_csv,
    #                         transform=train_transform)
    # mean, std = get_mean_and_std(train_all_data)
    # print(f"mean {mean}, std {std}")

    # train_num, val_num, test_num = len(train_data), len(val_data), len(test_0206_data)
    val_num = len(val_data)

    return (
        DataLoader(
            val_data,
            val_batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
            pin_memory=False,
            drop_last=True,
        ),
        val_num,
        csv_creator.act_list,
    )
    # DataLoader(test_0206_data, val_batch_size, shuffle=True,
    #            num_workers=get_dataloader_workers(), pin_memory=False, drop_last=True),
    # test_num)


def get_load_data(val_batch_size):

    img_size = 224

    vals = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=csi_mean, std=csi_std)]
    )

    [
        val_dataloader,
        val_data_num,
        each_act_num,
    ] = load_data_csi(val_batch_size, vals)
    print(f'test_num {val_data_num}\ndetail {each_act_num}')

    return (
        val_dataloader,
        val_data_num,
        each_act_num,
    )  # , test_dataloader, test_num


# def get_pretrain_model(pretrain_model, name, classes):

#     model_name = {"ghostnet": ghostnet}
#     model = model_name[name](num_classes = classes)
#     # if torch.cuda.is_available():
#     #     map_location = None
#     model.load_state_dict(torch.load(pretrain_model, map_location=lambda storage, loc: storage))
#     return model


# def get_pretrain_encoder(pretrain_model):

#     encoder = ghostnetEncoder()
#     # if torch.cuda.is_available():
#     #     map_location = None
#     encoder.load_state_dict(torch.load(
#         pretrain_model, map_location=lambda storage, loc: storage))
#     return encoder
