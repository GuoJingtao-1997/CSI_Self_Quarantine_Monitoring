import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from ghostnet_model import BranchyGhostnet

from load import get_load_data
from utils import (
    Accumulator,
    Evaluations,
    pred_label_cpu,
    WarmUpLR,
    get_cosine_schedule,
)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        '--train_device',
        type=str,
        default='left',
        metavar='N',
        help='device for model training',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=50,
        metavar='N',
        help='input batch size for training (default: 64)',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=50,
        metavar='N',
        help='input batch size for testing (default: 64)',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        metavar='LR',
        help='learning rate (default: 3e-4)',
    )

    parser.add_argument(
        '--wd', help='weight decay parameter (default: 1e-4);', type=float, default=1e-4
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=400,
        metavar='EP',
        help='how many epochs will be trained locally',
    )

    parser.add_argument('--use_benchmark', action="store_true", help='use benchmark')

    parser.add_argument('--use_gpu', action="store_true", help='use gpu')

    args = parser.parse_args()
    return args


def train(
    model,
    optimizer,
    criterion,
    train_dataloader,
):
    """train phase"""
    model.train()

    for imgs, labels in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc='train',
    ):
        optimizer.zero_grad()
        imgs, labels = imgs.to(device), labels.to(device)
        #imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
        # human_labels_a, act_labels_a = labels_a[:, :3], labels_a[:, 3:]
        # human_labels_b, act_labels_b = labels_b[:, :3], labels_b[:, 3:]
        human_labels, act_labels = labels[:, :3], labels[:, 3:]
        human_labels_hat, act_labels_hat = model(imgs)
        # branch_loss = mixup_criterion(criterion, human_labels_hat, argtorch.max(human_labels_a, dim=-1), torch.argmax(human_labels_b, dim=-1), lam)
        # final_loss = mixup_criterion(criterion, act_labels_hat, torch.argmax(act_labels_a, dim=-1), torch.argmax(act_labels_b, dim=-1), lam)
        branch_loss = criterion(human_labels_hat, human_labels.argmax(dim=-1))
        final_loss = criterion(act_labels_hat, act_labels.argmax(dim=-1))
        total_loss = branch_loss + final_loss
        total_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     metric.add(labels.detach().cpu().numpy().tolist(), pred_label_cpu(labels_hat))

    # evaluation_train = Evaluations(metric[0], metric[1], person_classes)


def test(
    model,
    test_dataloader,
    person_classes, 
    act_classes, 
    device,
):
    """test phase"""
    metric = Accumulator([[]], 4)
    test_size = len(test_dataloader)
    model.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(
            test_dataloader, total=len(test_dataloader), desc='test'
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            human_labels, act_labels = labels[:, :3], labels[:, 3:]
            
            # imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
            human_labels_hat, act_labels_hat = model(imgs)
            # branch_loss = criterion(human_labels_hat, torch.argmax(human_labels, dim=1))
            # final_loss = criterion(act_labels_hat, torch.argmax(act_labels, dim=1))

            metric.add(
                human_labels.argmax(dim=-1).detach().cpu().numpy().tolist(),
                pred_label_cpu(human_labels_hat),
                act_labels.argmax(dim=-1).detach().cpu().numpy().tolist(),
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    result_folder = os.path.join(os.getcwd(), f'branchynet_human_act_1.3_se')
    #transfer_folder = os.path.join(os.getcwd(), f'bilstm_human_act')
    logname_crowd = os.path.join(result_folder, f'crowd.csv')
    logname_act = os.path.join(result_folder, f'act.csv')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

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

    if args.use_gpu:
        device = torch.device('cuda')
        print(
            f'Using GPU: {torch.cuda.get_device_name()}, id: {torch.cuda.current_device()}'
        )
        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            print('Using cudnn.benchmark.')
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    model = BranchyGhostnet(branch_classes=3, final_classes=5, width=1.3).to(device)
    #model = ConvBiLSTMClassifier(encoder_features=234, dropout=0.2).to(device)
    #model = AttentionBiLSTMClassifier(emb_features=234).to(device)
    #model = LSTMClassifier(encoder_features=234, dropout=0.2).to(device)
    #model = LSTMClassifier(encoder_features=234, dropout=0.2).to(device)
    #model = TUNet().to(device)
    #model = FCNBaseline(in_channels=234).to(device)
    #model = MLSTM_FCN(c_in=234).to(device)
    #model = GRU_FCN(234).to(device)
    #model = BiLSTMClassifier(encoder_features=234).to(device)
    #model = gMLP(c_in=234, seq_len=300).to(device)
    
    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=args.wd,
    )


    warmup_scheduler = WarmUpLR(optimizer, args.warmup, args.warmuplr)
    #scheduler = get_cosine_schedule(optimizer, args.epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - 1))

    if not os.path.exists(logname_crowd):
        with open(logname_crowd, 'w') as logcrowd:
            logwriter = csv.writer(logcrowd, delimiter=',')
            logwriter.writerow(
                ['Epochs', 'precision', 'recall', 'f1_score', 'accuracy']
            )

    with open(logname_crowd, 'w') as logcrowd:
        logwriter = csv.writer(logcrowd, delimiter=',')
        logwriter.writerow(["cos_lr"])

    criterion = nn.CrossEntropyLoss()
    human_prec_max = human_recall_max = human_f1score_max = human_accuracy_max = 0.0

    since = time.time()

    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 10)

        all_val = np.array([])

        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        train(
            model,
            optimizer,
            criterion,
            train_dataloader,
            device,
        )
        eval_human_test, eval_act_test, human, human_hat, act, act_hat = test(
            model, val_dataloader, person_classes, act_classes, device
        )

        if (
                eval_human_test.average.precision() > human_prec_max
                and eval_human_test.average.recall() > human_recall_max
                and eval_human_test.average.f1_score() > human_f1score_max
                and eval_human_test.average.accuracy() > human_accuracy_max
            ):
            human_prec_max = eval_human_test.average.precision()
            human_recall_max = eval_human_test.average.recall()
            human_f1score_max = eval_human_test.average.f1_score()
            human_accuracy_max = eval_human_test.average.accuracy()

            with open(logname_crowd, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(
                    [
                        epoch + 1,
                        eval_human_test.average.precision(),
                        eval_human_test.average.recall(),
                        eval_human_test.average.f1_score(),
                        eval_human_test.average.accuracy(),
                    ]
                )

        if (
            eval_act_test.average.precision() > act_prec_max
            and eval_act_test.average.recall() > act_recall_max
            and eval_act_test.average.f1_score() > act_f1score_max
            and eval_act_test.average.accuracy() > act_accuracy_max
        ):
            act_prec_max = eval_act_test.average.precision()
            act_recall_max = eval_act_test.average.recall()
            act_f1score_max = eval_act_test.average.f1_score()
            act_accuracy_max = eval_act_test.average.accuracy()
            print(f'Get highest result for act calssification, Saving the global model......')
            #print(f'Get highest result for act calssification, loss {final_loss:.4f} Saving the global model......')
            torch.save(
                    model.state_dict(),
                    os.path.join(result_folder, f"model.pth"),
                )
            with open(logname_act, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(
                    [
                        epoch + 1,
                        eval_act_test.average.precision(),
                        eval_act_test.average.recall(),
                        eval_act_test.average.f1_score(),
                        eval_act_test.average.accuracy(),
                    ]
                )
            #plot_confuse_matrix(act, act_hat, range(len(act_classes)), transfer_folder)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f} m {time_elapsed % 60:.0f} s')
