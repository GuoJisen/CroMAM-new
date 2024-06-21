import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import logging


def setup_logger(name, log_file, file_mode, to_console=False):
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(log_file, mode=file_mode)
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        if to_console:
            logger.addHandler(logging.StreamHandler())

    return logger


def compose_logging(mode, cancer, outcome):
    log1 = setup_logger("meta", os.path.join("logs", f"{cancer}_{outcome}_{mode}_meta.log"), 'w', to_console=True)
    log2 = setup_logger("data", os.path.join("logs", f"{cancer}_{outcome}_{mode}_data.csv"), 'w', to_console=True)
    writer = {"meta": log1,
              "data": log2}
    return writer


def get_filename_extensions(args):
    ext_data = []
    magnification = list(args.magnification.split(','))
    for i in range(len(magnification)):
        ext_data.append('%s_%s_mag-%s_size-%s' % (args.cancer, args.stratify, magnification[i], args.patch_size))
    ext_experiment = 'by-%s_seed-%s' % (args.stratify, args.random_seed)
    ext_split = '%s_by-%s_seed-%s_nest-%s：%s' % (args.cancer, args.stratify,
                                                 args.random_seed, args.outer_fold, args.inner_fold)
    return ext_data, ext_experiment, ext_split


def calculate_metrics(preds, targets):
    f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')
    acc = accuracy_score(targets, preds.argmax(axis=1))
    try:  # AUC
        if preds.shape[1] > 2:
            # multi-class
            auc = roc_auc_score(targets.reshape(-1),
                                # torch.softmax(torch.tensor(preds), dim=1), multi_class='ovr')
                                torch.tensor(preds), multi_class='ovr')
        else:
            # binary
            auc = roc_auc_score(targets.reshape(-1),  # torch.softmax(torch.tensor(preds), dim=1)[:, 1]
                                torch.tensor(preds)[:, 1])
    except ValueError:
        auc = 0.5
    res = {
        'f1': f1,
        'auc': auc,
        'accuracy': acc
    }

    return res


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = 0
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        print("Bad epochs: %s; Patience: %s; Best value: %6.4f" %
              (self.num_bad_epochs, self.patience, float(str(self.best.item())[:6])))

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


class ModelEvaluation(object):
    def __init__(
            self,
            loss_function=None,
            mode='train',
            variables=None,
            device=torch.device('cpu'),
            timestr=None
    ):

        self.data = None
        if variables is None:
            variables = ['preds', 'targets', 'id']
        self.criterion = loss_function
        self.mode = mode
        self.timestr = timestr
        self.variables = variables
        self.device = device
        self.reset()

    def reset(self):
        self.data = dict()
        for var in self.variables:
            self.data[var] = None

    def update(self, batch):
        for k, v in batch.items():
            if isinstance(v, tuple):
                if self.data[k] is None:
                    self.data[k] = list()
                self.data[k] += list(v)
            else:
                # Tensor objects
                if self.data[k] is None:
                    self.data[k] = v.data.cpu().numpy()
                else:
                    self.data[k] = np.concatenate([self.data[k], v.data.cpu().numpy()])

    def evaluate(self):
        metrics = calculate_metrics(
            self.data['preds'],
            self.data['targets'],
        )

        if self.mode != 'test':
            loss_epoch = self.criterion.calculate(
                torch.tensor(self.data['preds']).to(self.device),
                torch.tensor(self.data['targets']).to(self.device))

            metrics['loss'] = loss_epoch.item()

        return metrics

    def save(self, filename):
        values = []
        for k, v in self.data.items():
            values.append(v)
        df = pd.concat(objs=[
            pd.DataFrame(values[0], columns=['pre_0', 'pre_1']),
            pd.DataFrame(values[1], columns=['g_true']),
            pd.DataFrame(values[2], columns=['id_patient'])], axis=1)
        return df
