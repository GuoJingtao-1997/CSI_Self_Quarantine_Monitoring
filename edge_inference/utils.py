import random
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = [
    'LabelSmoothingLossCanonical',
    'WarmUpLR',
    'EarlyStop',
    'Accumulator',
    'softmax',
    'mixup_criterion',
    'mixup_data',
    'setup_seed',
    'Evaluations',
    'pred_label_cpu',
    'plot_confuse_matrix',
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
            + self.last_epoch * (base_lr - self.warmup_lr_init) / self.warmup_epoch
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


def plot_confuse_matrix(y, y_hat, classes, result_path):
    """plot confuse matrix"""
    sns.set()
    f, ax = plt.subplots()
    confuse_matrix = confusion_matrix(y, y_hat, labels=classes)
    print(f'confuse_matrix:\n{confuse_matrix}')
    sns.heatmap(confuse_matrix, annot=True, ax=ax, fmt='g')
    ax.set_title('confuse matrix for crowd counting')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig(os.path.join(result_path, f'crowd_counting_confuse_matrix.jpg'))


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Using cudnn.deterministic.')
