import csv
import os
import math

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
# for 2D-Conv
csi_mean = [0.6317, 0.5258, 0.3464] 
csi_std = [0.2904, 0.3028, 0.2838]
# for 1D-Conv
# csi_mean = [0.5370] 
# csi_std = [0.2131]
human_path = os.path.join(os.getcwd(), r'../Dataset_human_5day_rgb')
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
num_class = len(header) - 1


class CsiLabeledCsv:
    def __init__(self) -> None:

        self.act_list = {}

        for human_acts in human_dirs:
            if human_acts not in [
                "train_human.csv",
                "test_human.csv",
            ]:
                self.act_list[human_acts] = len(
                    os.listdir(os.path.join(human_path, human_acts))
                )

        self.train_csv = os.path.join(human_path, r'train_human.csv')
        self.test_csv = os.path.join(human_path, r'test_human.csv')

    def create_csv(self):

        with open(os.path.join(human_path, r'train_human.csv'), 'a') as train_csv:
            train_writer = csv.writer(train_csv)
            train_writer.writerow(header)
        with open(os.path.join(human_path, r'test_human.csv'), 'a') as test_csv:
            test_writer = csv.writer(test_csv)
            test_writer.writerow(header)

        for human_acts in human_dirs:
            label_list = [0] * num_class
            act_path = os.path.join(human_path, human_acts)
            imgs = os.listdir(act_path)
            human_acts_list = human_acts.split(',')

            for acts in human_acts_list:
                label_list[header.index(acts) - 1] = 1

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

# for semi-supervised learning
def label_unlabel_split(args, labels):
    """return the label and unlabel index"""
    label_per_class = args.num_labeled // args.num_classes
    label_idx = np.array([], dtype=np.int)
    unlabeled_idx = np.array(range(len(labels)), dtype=np.int)

    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        label_idx = np.concatenate((label_idx, idx))
    assert len(label_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        label_idx = np.hstack([label_idx for _ in range(num_expand_x)])
    np.random.shuffle(label_idx)

    return label_idx, unlabeled_idx


def get_dataloader_workers():
    """Use num processes to read the data."""
    return 4


def get_transforms(csi_mean, csi_std):

    img_size = (234, 300)
    trains = transforms.Compose(
        [
            #transforms.Grayscale(num_output_channels=1), # Uncomment it for 1D Conv 
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=csi_mean, std=csi_std),
        ]
    )
    vals = transforms.Compose(
        [#transforms.Grayscale(num_output_channels=1), # Uncomment for 1D Conv 
        transforms.ToTensor(), 
        transforms.Normalize(mean=csi_mean, std=csi_std)]
    )

    return trains, vals

def load_data_csi(val_batch_size, vals, train_batch_size, trains):

    setup_seed(42)
    csv_creator = CsiLabeledCsv()

    if 'train_human.csv' in os.listdir(human_path):
        # os.remove(os.path.join(act_path, "test_human.csv"))
        # os.remove(os.path.join(act_path, "train_human.csv"))
        pass
    else:
        csv_creator.create_csv()

    val_data = CsiLabeled(csv=csv_creator.test_csv, transform=vals)

    train_data = CsiLabeled(csv=csv_creator.train_csv,
                          transform=trains)

    mean, std = get_mean_and_std(train_data)
    print(f"mean {mean}, std {std}")

    # train_num, val_num, test_num = len(train_data), len(val_data), len(test_0206_data)
    val_num, train_num = len(val_data), len(train_data)

    return (
        DataLoader(
            train_data,
            train_batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
            pin_memory=False,
            drop_last=True,
        ),
        train_num,
        DataLoader(
            val_data,
            val_batch_size,
            shuffle=False,
            num_workers=get_dataloader_workers(),
            pin_memory=False,
            drop_last=True,
        ),
        val_num,
        csv_creator.act_list,
    )

def get_load_data(train_batch_size, val_batch_size):

    setup_seed(42)

    trains, vals = get_transforms(csi_mean, csi_std)

    [
        train_dataloader,
        train_data_num,
        val_dataloader,
        val_data_num,
        each_act_num,
    ] = load_data_csi(val_batch_size, vals, train_batch_size, trains)
    print(f'train_num {train_data_num} test_num {val_data_num}\ndetail {each_act_num}')

    return (
        train_dataloader,
        train_data_num,
        val_dataloader,
        val_data_num,
        each_act_num,
    )  # , test_dataloader, test_num



# class TransformTwice:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, inp):
#         out1 = self.transform(inp)
#         out2 = self.transform(inp)
#         return out1, out2


# class CsiUnLabeled(Dataset):
#     def __init__(
#         self, csv, transform=None, target_transform=None, loader=default_loader
#     ) -> None:
#         super().__init__()

#         self.data_info = pd.read_csv(csv, header=None)
#         self.imgs = np.asarray_chkfinite(self.data_info.iloc[1:, 0])
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader

#     def __getitem__(self, index):

#         img_path = self.imgs[index]

#         img_u1, img_u2 = (
#             self.transform(self.loader(img_path))
#             if self.transform is not None
#             else self.loader(img_path)
#         )

#         return img_u1, img_u2

#     def __len__(self):
#         return len(self.imgs)
