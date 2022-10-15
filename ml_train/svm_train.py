import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from load import get_load_data
from utils import (
    Accumulator,
    Evaluations,
    softmax,
    pred_label_cpu,
)

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for testing (default: 64)',
    )

    parser.add_argument('--use_gpu', action="store_true", help='use gpu')

    args = parser.parse_args()
    return args


def test(clf_human, clf_act, test_dataloader, person_classes, act_classes, device, val_data_num, args, scaler):
    test_features = np.zeros(shape=(val_data_num, 150528))
    test_human_labels = np.zeros(shape=(val_data_num))
    test_act_labels = np.zeros(shape=(val_data_num))
    test_features = scaler.transform(test_features)
    metric = Accumulator([[]], 4)

    with torch.no_grad():
        num = -1
        for imgs, labels in tqdm(test_dataloader, total=len(test_dataloader), desc='test'):
            human_labels, act_labels = labels[:, :3], labels[:, 3:]
            human_labels, act_labels = human_labels.numpy(), act_labels.numpy()
            # features = encoder(imgs)
            features = imgs.reshape(args.val_batch_size, -1)
            for j in range(args.val_batch_size):
                num += 1
                test_features[num] = features[j].numpy()
                test_human_labels[num] = np.argmax(human_labels[j], axis=-1)
                test_act_labels[num] = np.argmax(act_labels[j], axis=-1)


        # decision is a voting function
        human_hat = np.array(clf_human.decision_function(test_features))
        human_hat = softmax(human_hat)
        human_hat = np.argmax(human_hat, axis=-1)

        act_hat = np.array(clf_act.decision_function(test_features))
        act_hat = softmax(act_hat)
        act_hat = np.argmax(act_hat, axis=-1)

        metric.add(
                test_human_labels.tolist(),
                human_hat.tolist(),
                test_act_labels.tolist(),
                act_hat.tolist(),
            )
    
    eval_human_test = Evaluations(metric[0], metric[1], person_classes)
    eval_act_test = Evaluations(metric[2], metric[3], act_classes)
    print(eval_human_test)
    print(eval_act_test)

    with open(logname_crowd, 'a') as logcrow, open(logname_act, 'a') as logact:
                logwriter = csv.writer(logcrow, delimiter=',')
                logwriter.writerow(
                    [
                        eval_human_test.average.precision(),
                        eval_human_test.average.recall(),
                        eval_human_test.average.f1_score(),
                        eval_human_test.average.accuracy(),
                    ]
                )
                logwriter = csv.writer(logact, delimiter=',')
                logwriter.writerow(
                    [
                        eval_act_test.average.precision(),
                        eval_act_test.average.recall(),
                        eval_act_test.average.f1_score(),
                        eval_act_test.average.accuracy(),
                    ]
                )
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    result_folder = os.path.join(os.getcwd(), f'svm_human_act')
    #transfer_folder = os.path.join(os.getcwd(), f'transfer_early_exit_human_act')
    logname_crowd = os.path.join(result_folder, f'crowd.csv')
    logname_act = os.path.join(result_folder, f'act.csv')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if not os.path.exists(logname_crowd):
        with open(logname_crowd, 'w') as logcrowd:
            logwriter = csv.writer(logcrowd, delimiter=',')
            logwriter.writerow(
                ['precision', 'recall', 'f1_score', 'accuracy']
            )
    
    if not os.path.exists(logname_act):
        with open(logname_act, 'w') as logact:
            logwriter = csv.writer(logact, delimiter=',')
            logwriter.writerow(
                ['precision', 'recall', 'f1_score', 'accuracy']
            )

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
    train_features = np.zeros(
        shape=(train_data_num, 150528))
    train_human_labels = np.zeros(
        shape=(train_data_num))
    train_act_labels = np.zeros(
        shape=(train_data_num))
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    num = -1
    for imgs, labels in tqdm(train_dataloader, total=len(train_dataloader)):
        human_labels, act_labels = labels[:, :3], labels[:, 3:]
        human_labels, act_labels = human_labels.numpy(), act_labels.numpy()
        # features = encoder(imgs)
        features = imgs.reshape(args.train_batch_size, -1)
        for j in range(args.train_batch_size):
            num += 1
            train_features[num] = features[j].numpy()
            train_human_labels[num] = np.argmax(human_labels[j], axis=-1)
            train_act_labels[num] = np.argmax(act_labels[j], axis=-1)

    print(train_features.shape)

    clf_human = SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma=0.1)
    clf_human.fit(train_features, train_human_labels)

    clf_act = SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma=0.1)
    clf_act.fit(train_features, train_act_labels)

    test(clf_human, clf_act, val_dataloader, person_classes, act_classes, device, val_data_num, args, scaler)