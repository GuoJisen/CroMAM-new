from pathlib import Path
import os
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
import shutil
from tqdm import tqdm
from utils.helper import EarlyStopping, ModelEvaluation


def data_to_device(imgs, ids, targets, device):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.to(device)
    if isinstance(ids, torch.Tensor):
        ids = ids.to(device)
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device)
    return imgs, ids, targets


def format_results(res):
    """
    Format logging
    """
    line = ""
    for key in sorted(res):
        val = res[key]
        if isinstance(val, str) or isinstance(val, int):
            fmt = "%s: %s\t"
        else:
            fmt = "%s: %8.6f\t"
        line += (fmt % (key, val))
    return line


def lr_scheduler_fc(epoch, max_epochs=100, exponent=0.9, warm_epoch=5, gama=0.7, mode='w'):
    """
    exponent
    """
    if mode == 'p':
        return (1 - epoch / max_epochs) ** exponent
    elif mode == 'w':
        if epoch <= warm_epoch:
            return 0.01 * epoch
        else:
            return (1 - epoch / max_epochs) ** exponent
    else:
        print('lr_scheduler_fc error')


class MultiHybridFitter:
    """
    Helper class for scheduling training and evaluation.
    """

    def __init__(
            self,
            model,
            dataloader,
            checkpoint_to_resume='',
            writer=None,
            args=None,
            timestr='',
            model_name='model',
            loss_function=None,
            fold=0,
            scaler=None
    ):
        self.checkpoints_folder = None
        self.schedulers = None
        self.optimizers = None
        self.writer = writer
        self.args = args
        self.criterion = loss_function
        self.model = model
        self.device = self.model.device
        self.reset_optimizer()
        self.dataloaders = dataloader
        self.es = EarlyStopping(patience=self.args.patience, mode='max')
        self.timestr = timestr
        self.model_name = model_name
        self.checkpoint_to_resume = checkpoint_to_resume
        self.best_metric = 0
        self.fold = fold
        self.current_epoch = 1
        self.scaler = scaler

        if len(checkpoint_to_resume):
            self.resume_checkpoint()

    def resume_checkpoint(self):
        """
        Load pretrained model from last time training.
        """
        ckp = torch.load(self.checkpoint_to_resume)
        self.writer['meta'].info("Loading model checkpoints ... Epoch is %s" % ckp['epoch'])
        self.model.load_state_dict(ckp['state_dict_model'])
        self.optimizers['adam'].load_state_dict(ckp['state_dict_optimizer_adam'])
        self.optimizers['sgd'].load_state_dict(ckp['state_dict_optimizer_sgd'])
        self.schedulers['adam'].load_state_dict(ckp['state_dict_scheduler_adam'])
        self.schedulers['sgd'].load_state_dict(ckp['state_dict_scheduler_sgd'])
        self.current_epoch = ckp['epoch'] + 1

    def reset_optimizer(self):
        """
        Initialize model optimization and learning rate schedulers
        """
        backbone_params = []
        SRPFIM_params = []
        MRFFM_params = []
        classifier_params = []
        # backbone
        for name, param in self.model.backbone1.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        for name, param in self.model.backbone2.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        # SIB
        for name, param in self.model.vit1.named_parameters():
            if param.requires_grad:
                SRPFIM_params.append(param)
        for name, param in self.model.vit2.named_parameters():
            if param.requires_grad:
                SRPFIM_params.append(param)
        # MFB
        for name, param in self.model.clste1.named_parameters():
            if param.requires_grad:
                MRFFM_params.append(param)
        for name, param in self.model.clste2.named_parameters():
            if param.requires_grad:
                MRFFM_params.append(param)
        # classifier
        for name, param in self.model.head.named_parameters():
            if param.requires_grad:
                classifier_params.append(param)
        self.optimizers = {
            'adam': optim.AdamW([
                {
                    'params': backbone_params,
                    'lr': self.args.lr_backbone,
                    'weight_decay': self.args.wd_backbone,
                },
                {
                    'params': SRPFIM_params,
                    'lr': self.args.lr_fusion,
                    'weight_decay': self.args.wd_fusion,
                },
                {
                    'params': MRFFM_params,
                    'lr': self.args.lr_fusion,
                    'weight_decay': self.args.wd_fusion,
                },
                {
                    'params': classifier_params,
                    'lr': self.args.lr_classifier,
                    'weight_decay': self.args.wd_classifier,
                }
            ]),
            'sgd': optim.SGD([
                {
                    'params': backbone_params,
                    'lr': self.args.lr_backbone,
                    'weight_decay': self.args.wd_backbone,
                },
                {
                    'params': classifier_params,
                    'lr': self.args.lr_classifier,
                    'weight_decay': self.args.wd_classifier,
                }
            ])
        }
        self.schedulers = {
            'adam': lr_scheduler.LambdaLR(
                self.optimizers['adam'],
                lr_lambda=lambda epoch: lr_scheduler_fc(epoch, mode='w')
            ),
            'sgd': lr_scheduler.LambdaLR(
                self.optimizers['sgd'],
                lr_lambda=lambda epoch: lr_scheduler_fc(epoch, mode='w')
            )
        }

    def save_checkpoint(
            self,
            epoch,
            is_best,
            save_freq,
            checkpoints_folder):

        state_dict = {
            'epoch': epoch,
            'state_dict_model': self.model.state_dict(),
            'state_dict_optimizer_adam': self.optimizers['adam'].state_dict(),
            'state_dict_optimizer_sgd': self.optimizers['sgd'].state_dict(),
            'state_dict_scheduler_adam': self.schedulers['adam'].state_dict(),
            'state_dict_scheduler_sgd': self.schedulers['sgd'].state_dict(),
        }
        # remaining things related to training
        os.makedirs(checkpoints_folder, exist_ok=True)
        epoch_output_path = os.path.join(checkpoints_folder, "LAST.pt")
        torch.save(state_dict, epoch_output_path)

        if is_best:
            print("Saving new best result!")
            fname_best = Path(checkpoints_folder) / "BEST.pt"
            if os.path.isfile(fname_best):
                os.remove(fname_best)
            shutil.copy(epoch_output_path, fname_best)

        if epoch % save_freq == 0:
            print("Saving new checkpoints!")
            shutil.copy(epoch_output_path, Path(checkpoints_folder) / ("%04d.pt" % epoch))

    def train(self, epoch=0):
        """Model Training of the current epoch"""
        # logger
        self.writer['meta'].info('Training from step %s' % epoch)
        # Model result data processing class
        eval_t = ModelEvaluation(
            loss_function=self.criterion,
            mode='train',
            device=self.device,
            timestr=self.timestr)
        # Display learning rate
        flag = 0
        for group in self.optimizers['adam'].param_groups:
            current_lr = group['lr']
            if flag == 0:
                self.writer['meta'].info(f"Learning backbone lr rate is {current_lr}")
            elif flag == 1:
                self.writer['meta'].info(f"Fusion module  lr rate is {current_lr}")
            elif flag == 3:
                self.writer['meta'].info(f"Learning classifier lr rate is {current_lr}")
            flag += 1
        self.writer['meta'].info('-' * 90)
        del flag

        # train over all training data
        self.model.train()
        data_iterator = zip(self.dataloaders['train1'], self.dataloaders['train2'])
        for (train_imgs1, train_ids1, train_targets1), (train_imgs2, train_ids2, train_targets2) in tqdm(data_iterator,
                                                                                                         total=len(
                                                                                                             self.dataloaders[
                                                                                                                 'train1'])):
            train_imgs1, train_ids1, train_targets1 = data_to_device(train_imgs1, train_ids1, train_targets1,
                                                                     self.device)
            train_imgs2, train_ids2, train_targets2 = data_to_device(train_imgs2, train_ids2, train_targets2,
                                                                     self.device)
            nbatches = train_imgs1.size(0) // self.args.num_patches  # batch size

            # forward and backprop
            with torch.set_grad_enabled(True):
                ppi = self.args.num_patches
                model_outputs = self.model(train_imgs1, train_imgs2, ppi)  # ppi The final classification layer integrates the same WSI results
                train_preds = model_outputs['pred']
                train_targets1 = train_targets1.view(nbatches, self.args.num_patches, -1)[:, 0, :]
                train_ids1 = train_ids1.view(nbatches, self.args.num_patches, -1)[:, 0, :]
                train_loss = self.criterion.calculate(train_preds, train_targets1)

                self.optimizers['adam'].zero_grad()
                self.scaler.scale(train_loss).backward()
                self.scaler.step(self.optimizers['adam'])
                self.scaler.update()

                eval_t.update(
                    {
                        "id": train_ids1,
                        "preds": train_preds,
                        "targets": train_targets1
                    }
                )
        train_res = eval_t.evaluate()
        train_res['epoch'] = epoch
        train_res['mode'] = 'train'
        return train_res

    def evaluate(self, epoch=0):
        """Model evaluation of the current epoch（val）"""
        self.writer['meta'].info('Starting evaluation')
        eval_v = ModelEvaluation(
            loss_function=self.criterion,
            mode='val',
            device=self.device,
            timestr=self.timestr)

        self.model.eval()
        with torch.no_grad():
            data_iterator = zip(self.dataloaders['val1'], self.dataloaders['val2'])
            for (val_imgs1, val_ids1, val_targets1), (val_imgs2, val_ids2, val_targets2) in tqdm(data_iterator,
                                                                                                 total=len(
                                                                                                     self.dataloaders[
                                                                                                         'val1'])):
                val_imgs1, val_ids1, val_targets1 = data_to_device(val_imgs1, val_ids1, val_targets1,
                                                                   self.device)
                val_imgs2, val_ids2, val_targets2 = data_to_device(val_imgs2, val_ids2, val_targets2,
                                                                   self.device)
                nbatches = val_imgs1.size(0) // (self.args.num_val * self.args.num_crops)

                # forward
                val_preds = self.model(val_imgs1, val_imgs2, self.args.num_val)['pred']
                val_targets1 = val_targets1.view(nbatches, self.args.num_val, -1)[:, 0, :]
                val_ids1 = val_ids1.view(nbatches, self.args.num_val, -1)[:, 0, :]

                eval_v.update(
                    {
                        "id": val_ids1,
                        "preds": val_preds,
                        "targets": val_targets1
                    }
                )

        # Save the evaluation result of current epoch
        val_res = eval_v.evaluate()
        val_res['epoch'] = epoch
        val_res['mode'] = 'val'

        if val_res['auc'] > self.best_metric:
            save_path = os.path.join("predictions", self.model_name + f'_fold_{self.fold + 1}',
                                     f"%04d_best{float(str(val_res['auc'])[:6])}.csv" % epoch)
        else:
            save_path = os.path.join("predictions", self.model_name + f'_fold_{self.fold + 1}', "%04d.csv" % epoch)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ids = self.dataloaders['val1'].sampler.data_source.df[['submitter_id', 'id_patient']].drop_duplicates(
            'submitter_id').reset_index(drop=True)
        e_df = eval_v.save(save_path)
        e_df = e_df.merge(ids, on='id_patient', how='left')
        e_df.to_csv(save_path)
        return val_res

    def fit_epoch(self, epoch=0):
        """
        Helpter function to fit the model for current epoch and save model
        """
        train_res = self.train(epoch=epoch)  # Start training
        self.writer['data'].info(format_results(train_res))
        val_res = self.evaluate(epoch=epoch)
        self.writer['data'].info(format_results(val_res))

        self.schedulers['adam'].step()

        # Choose the evaluation metrics for outcome type to optimize
        performance_measure = torch.tensor(val_res['auc'])

        # Save best performed model and at every save_interval
        is_best = False
        if performance_measure > self.best_metric:
            self.best_metric = performance_measure.item()
            print(f"New best result: {float(str(self.best_metric)[:6])}")
            is_best = True

        self.save_checkpoint(
            epoch=epoch,
            is_best=is_best,
            save_freq=self.args.save_interval,
            checkpoints_folder=self.checkpoints_folder
        )

        # Early stopping
        if self.es.step(performance_measure):
            return 1, performance_measure.item()  # early stop 
        return 0, performance_measure.item()

    def fit(self, fold, checkpoints_folder='checkpoints'):
        """Model fitting"""
        self.checkpoints_folder = str(checkpoints_folder) + f'_fold_{fold + 1}'
        fig, a = plt.subplots(1, 1)
        xs = []
        ys = []
        for epoch in range(self.current_epoch, self.args.epochs + 1):  
            return_code, auc = self.fit_epoch(epoch=epoch)
            auc = float(str(auc)[:6])
            # Draw auc value change graph
            plt.cla()
            a.set_ylim(0, 1)
            a.set_title('AUC_Line')
            a.set_xlabel('epoch')
            a.set_ylabel('AUC')
            xs.append(epoch)
            ys.append(auc)
            a.plot(xs, ys, label=f'Val_Best_Auc = {float(str(self.best_metric)[:6])}', linewidth=1, color='r')
            plt.legend()
            plt.pause(0.0001)

            if return_code:
                print("early stopping")
                break
            if epoch == self.args.epochs:
                print("this fold process over")
                break
        magnification = list(self.args.magnification.split(','))
        plt.savefig(
            f'./logs/AucLineChart/{self.args.cancer}_{self.args.outcome}_{self.args.model_name}_val_auc_X{magnification}_fold_{fold + 1}.png') 
        plt.close(fig)
        # Save the best result of each fold
        np.savetxt(f'{os.path.join(self.checkpoints_folder, f"{self.best_metric}.txt")}',
                   np.array([self.best_metric]), fmt='%s', delimiter=' ')
