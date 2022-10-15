import random
import os
import ujson
import math
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

__all__ = [
    'LabelSmoothingLossCanonical',
    'WarmUpLR',
    'EarlyStop',
    'Accumulator',
    'softmax',
    'copyStateDict',
    'draw_matrix_acc',
    'mixup_criterion',
    'mixup_data',
    'setup_seed',
    'Evaluations',
    'pred_label_cpu',
    '_make_divisible',
    'get_mean_and_std',
    'seperate_data',
    'split_data',
    'save_file',
    'check_dataset',
    'MMD',
]


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class WarmUpLR(_LRScheduler):
    def __init__(
        self, optimizer, warmup_epoch, warmup_lr_init, last_epoch=-1, verbose=False
    ):

        self.warmup_epoch = warmup_epoch
        self.warmup_lr_init = warmup_lr_init

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return a warm-up learning rate"""
        return [
            self.warmup_lr_init
            + self.last_epoch * (base_lr - self.warmup_lr_init) / (self.warmup_epoch - 1)
            for base_lr in self.base_lrs
        ]

    def __str__(self):
        return 'WarmUpLR'


class EarlyStop:
    """
    Early stop the training if the loss of validation does not improve in the given patience
    """

    def __init__(
        self,
        patience=8,
        verbose=False,
        delta=0.0,
        model_path="checkpoint_encoder.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.val_acc_max = 0
        self.best_score = None
        self.early_stop = False
        self.count = 0.0
        self.model_path = model_path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):

        score = val_acc
        self.change = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score - self.delta:
            self.count += 1
            self.trace_func(f"Early stopping counter {self.count} / {self.patience}")
            if self.count == self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.count = 0

    def save_checkpoint(self, val_acc, model):
        """
        Save model when val_acc increase
        """
        self.change = True
        if self.verbose:
            self.trace_func(
                f"Validation acc increase from {self.val_acc_max:3%} --> {val_acc:3%}. Saving model......"
            )
        torch.save(model.state_dict(), self.model_path)
        self.val_acc_max = val_acc


class Accumulator:
    """For accumulating sums over 'n' variables"""

    def __init__(self, acc_type, n):
        self.data = acc_type * n

    def add(self, *arg):
        self.data = [a + b for a, b in zip(self.data, arg)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Evaluations:
    """Compute the Accuracy, Precision, Recall and F1-score"""

    def __init__(self, y, y_hat, classes) -> None:
        if type(y) != np.ndarray:
            y = np.array(y)
        if type(y_hat) != np.ndarray:
            y_hat = np.array(y_hat)

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.classes = classes

        for class_ in classes:
            index_ = classes.index(class_)
            tp_ = ((y == index_) & (y_hat == index_)).sum()
            self.tp += tp_
            tn_ = ((y != index_) & (y_hat != index_)).sum()
            self.tn += tn_
            fp_ = ((y != index_) & (y_hat == index_)).sum()
            self.fp += fp_
            fn_ = ((y == index_) & (y_hat != index_)).sum()
            self.fn += fn_
            setattr(self, class_, Metrics(tp_, tn_, fp_, fn_))
        setattr(self, "average", Metrics(self.tp, self.tn, self.fp, self.fn))

    def __repr__(self) -> str:
        splitline_str = '*' * 203
        classesline_str = (
            ' ' * 15
            + ' |Aveg|     '
            + ''.join([f' |{self.classes[i]}|      ' for i in range(len(self.classes))])
        )
        preline_str = (
            'precision: \t'
            + f"{getattr(self, 'average').precision():0.4%}"
            + ''.join(
                [
                    f'   {getattr(self,self.classes[i]).precision():0.4%}  '
                    for i in range(len(self.classes))
                ]
            )
        )
        recline_str = (
            'recall: \t'
            + f"{getattr(self, 'average').recall():0.4%}"
            + ''.join(
                [
                    f'   {getattr(self,self.classes[i]).recall():0.4%}  '
                    for i in range(len(self.classes))
                ]
            )
        )
        acurline_str = (
            'accuracy: \t'
            + f"{getattr(self, 'average').accuracy():0.4%}"
            + ''.join(
                [
                    f'   {getattr(self,self.classes[i]).accuracy():0.4%}  '
                    for i in range(len(self.classes))
                ]
            )
        )
        f1score_str = (
            'fl_score: \t'
            + f"{getattr(self, 'average').f1_score():0.4%}"
            + ''.join(
                [
                    f'   {getattr(self,self.classes[i]).f1_score():0.4%}  '
                    for i in range(len(self.classes))
                ]
            )
        )

        return f"{splitline_str}\n{classesline_str}\n{preline_str}\n{recline_str}\n{acurline_str}\n{f1score_str}\n{splitline_str}"

    def __dir__(self):
        dir_ = ['average']
        dir_.extend(self.classes)
        return dir_

    def writelog(self, writer, key='average', path='', global_step=None):
        # write.add_scalar('train/accuracy',accuracy_,global_step=iterations_)
        if key in dir(self):
            writer.add_scalar(
                path + '/{}/precision'.format(key),
                getattr(self, key).precision(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/recall'.format(key),
                getattr(self, key).recall(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/accuracy'.format(key),
                getattr(self, key).accuracy(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/f1_score'.format(key),
                getattr(self, key).f1_score(),
                global_step=global_step,
            )
        elif key == 'ALL':
            for attr_ in dir(self):
                writer.add_scalar(
                    path + '/{}/precision'.format(attr_),
                    getattr(self, attr_).precision(),
                    global_step=global_step,
                )
                writer.add_scalar(
                    path + '/{}/recall'.format(attr_),
                    getattr(self, attr_).recall(),
                    global_step=global_step,
                )
                writer.add_scalar(
                    path + '/{}/accuracy'.format(attr_),
                    getattr(self, attr_).accuracy(),
                    global_step=global_step,
                )
                writer.add_scalar(
                    path + '/{}/f1_score'.format(attr_),
                    getattr(self, attr_).f1_score(),
                    global_step=global_step,
                )
        else:
            writer.add_scalar(
                path + '/{}/precision'.format('average'),
                getattr(self, 'average').precision(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/recall'.format('average'),
                getattr(self, 'average').recall(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/accuracy'.format('average'),
                getattr(self, 'average').accuracy(),
                global_step=global_step,
            )
            writer.add_scalar(
                path + '/{}/f1_score'.format('average'),
                getattr(self, 'average').f1_score(),
                global_step=global_step,
            )


class Metrics:
    """ "Compute the Accuracy, Precision, Recall and F1-score"""

    def __init__(self, tp, tn, fp, fn) -> None:
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)


# def plot_confuse_matrix(y, y_hat, classes, result_path):
#     """plot confuse matrix"""
#     sns.set()
#     f, ax = plt.subplots()
#     confuse_matrix = confusion_matrix(y, y_hat, labels=classes)
#     print(f'confuse_matrix:\n{confuse_matrix}')
#     sns.heatmap(confuse_matrix, annot=True, ax=ax, fmt='g')
#     ax.set_title('confuse matrix for crowd counting')
#     ax.set_xlabel('predict')
#     ax.set_ylabel('true')
#     plt.savefig(os.path.join(
#         result_path, f'crowd_counting_confuse_matrix.jpg'))


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_cosine_schedule(
    optimizer, num_training_epochs, num_cycles=7.0 / 16.0, last_epoch=-1
):
    def _lr_lambda(current_epoch):
        no_progress = float(current_epoch) / float(max(1, num_training_epochs))
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def mixup_data(x, y, alpha=0.5, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v

    return new_state_dict


def pred_label_cpu(y_hat):
    """Find the index of the max output value"""
    return y_hat.argmax(dim=-1).detach().cpu().numpy().tolist()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    temp = np.exp(x - x_max)
    s = np.sum(temp, axis=axis, keepdims=True)
    return temp / s


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Using cudnn.deterministic.')


def plot_confusion_matrix(cm, labels, result_path, cmap=plt.cm.turbo):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    font1 = {'size': 17}
    xlocations = np.array(range(len(labels)))  # （1，c）类别个数作为位置，
    plt.xticks(xlocations, labels, fontsize=17)  # 将x坐标的label旋转0度 ，每个位置上放好对应的label
    plt.yticks(xlocations, labels, fontsize=17)
    plt.ylabel('True label', font1)
    plt.xlabel('Predicted label', font1)
    plt.savefig(result_path, dpi=300, bbox_inches='tight')


def draw_matrix_acc(y_pred, y_true, labels, result_path):
    tick_marks = np.array(range(len(labels))) + 0.5
    ind_array = np.arange(len(labels))
    cm = confusion_matrix(y_true, y_pred)  # 求解confusion matrix
    print(cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        # if c > 0.01:
        plt.text(
            x_val,
            y_val,
            "%0.3f" % (c * 100,),
            color='white',
            fontsize=17,
            va='center',
            ha='center',
        )
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, labels, result_path)
    # show confusion matrix
    # plt.show()


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


alpha = 0.1


def seperate_data(
    data,
    num_clients,
    num_classes,
    batch_size,
    is_train=True,
    niid=False,
    real=True,
    partition=None,
    balance=False,
    class_per_client=6,
):
    x = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    client_statistic = [[] for _ in range(num_clients)]

    dataset_contents, dataset_labels = data

    if is_train:
        least_samples_per_class = 384
    else:
        least_samples_per_class = 80

    if partition == None or partition == "noise":
        dataset = []
        # obtain dataset contents per class
        for i in range(num_classes):
            idx = dataset_labels == i
            dataset.append(dataset_contents[idx])

        if not niid or real:
            class_per_client = num_classes

        class_num_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            if niid and not real:
                selected_clients = selected_clients[
                    : int(num_clients / num_classes * class_per_client)
                ]
                # print(f'niid and not real {selected_clients}')

            num_all = len(dataset[i])
            num_selected_clients = len(selected_clients)
            # print(f'num_selected_clients: {num_selected_clients}')

            if niid and real:
                num_selected_clients = np.random.randint(1, len(selected_clients))
                # print(f'niid and real - num_selected_clients: {num_selected_clients}')
            num_per_client = int(num_all / num_selected_clients)
            # print(f'num_per_client: {num_per_client}')

            if balance:
                if is_train:
                    num_samples = [
                        num_per_client if num_per_client == 410 else num_per_client - 22
                        for _ in range(num_selected_clients)
                    ]
                else:
                    num_samples = [
                        num_per_client if num_per_client == 102 else num_per_client - 6
                        for _ in range(num_selected_clients)
                    ]
                # print(f'balance num_samples: {num_samples}')
            else:
                # num_samples = np.random.randint(
                #     # max(num_per_client / 10, least_samples / num_classes),
                #     least_samples_per_class,
                #     num_per_client,
                #     num_selected_clients - 1,
                # ).tolist()
                num_samples = [
                    random.randrange(
                        least_samples_per_class,
                        num_per_client,
                        batch_size,
                    )
                    for _ in range(num_selected_clients - 1)
                ]
                # print(
                #     f'not balance num_samples: {num_samples}',
                # )
                num_samples.append(num_all - sum(num_samples))
            # print(f'num_samples: {num_samples}')

            if niid:
                selected_clients = list(
                    np.random.choice(
                        selected_clients, num_selected_clients, replace=False
                    )
                )
                # print(f"niid selected_clients {selected_clients}")

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if len(x[client]) == 0:
                    x[client] = dataset[i][idx : idx + num_sample]
                    y[client] = i * np.ones(num_sample)
                else:
                    x[client] = np.append(
                        x[client], dataset[i][idx : idx + num_sample], axis=0
                    )
                    y[client] = np.append(y[client], i * np.ones(num_sample), axis=0)
                idx += num_sample
                client_statistic[client].append((i, num_sample))
                class_num_client[client] -= 1

    elif niid and partition == "dir":
        min_size = 0
        k = num_classes
        n = len(dataset_labels)
        data_idx_map = {}

        while min_size < least_samples_per_class:
            idx_batch = [[] for _ in range(num_clients)]

            for i in range(k):
                idx_k = np.where(dataset_labels == i)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < n / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                # print("proportions 1", proportions)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            np.random.shuffle(idx_batch[i])
            data_idx_map[i] = idx_batch[i]

        # additional codes
        for client in range(num_clients):
            idxs = data_idx_map[client]
            x[client] = dataset_contents[idxs]
            y[client] = dataset_labels[idxs]

            for i in np.unique(y[client]):
                client_statistic[client].append((int(i), int(sum(y[client] == i))))

    else:
        raise EOFError

    del data

    for client in range(num_clients):
        print(
            f"Client: {client}\t Size of data: {len(x[client])}\t Labels: {np.unique(y[client])}"
        )
        print(
            f"Client: {client}\t Samples Per Label: {[i for i in client_statistic[client]]}"
        )
        print("-" * 100)

    return x, y, client_statistic


def split_data(x, y, batch_size):
    # split dataset
    train_data, test_data = [], []
    num_samples = {"train": [], "test": []}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        train_size = math.ceil(len(y[i]) * 0.8 / batch_size) * batch_size
        if min(count) > 1:
            x_train, x_test, y_train, y_test = train_test_split(
                x[i], y[i], train_size=train_size, shuffle=True, stratify=y[i]
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x[i], y[i], train_size=train_size, shuffle=True, stratify=None
            )

        train_data.append({"x": x_train, "y": y_train})
        num_samples["train"].append(len(y_train))
        test_data.append({"x": x_test, "y": y_test})
        num_samples["test"].append(len(y_test))

    print(f'Total number of samples: {sum(num_samples["train"] + num_samples["test"])}')
    print(f'The number of train samples: {num_samples["train"]}')
    print(f'The number of test samples: {num_samples["test"]}')

    del x, y

    return train_data, test_data


def check_dataset(
    config_path,
    train_path,
    test_path,
    num_clients,
    num_classes,
    batch_size,
    niid=False,
    real=True,
    balance=True,
    partition=None,
):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if (
            config['num_clients'] == num_clients
            and config['num_classes'] == num_classes
            and config['non_iid'] == niid
            and config['real_world'] == real
            and config['balance samples'] == balance
            and config['partition'] == partition
            and config['alpha'] == alpha
            and config['batch_size'] == batch_size
        ):
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def save_file(
    config_path,
    train_path,
    test_path,
    train_data,
    test_data,
    num_clients,
    num_classes,
    statistic,
    batch_size,
    domain_shift,
    niid=False,
    real=True,
    balance=True,
    partition=None,
):
    config = {
        'num_clients': num_clients,
        '\nnum_classes': num_classes,
        '\nnon_iid': niid,
        '\nreal_world': real,
        '\npartition': partition,
        '\nbalance': balance,
        '\nSize of samples for labels in clients': statistic,
        '\nalpha': alpha,
        '\nbatch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx + domain_shift) + '_' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx + domain_shift) + '_' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
