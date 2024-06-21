import os
import time
from pathlib import Path
import numpy as np
import torch
from utils.useful_function import seed_torch
from data.dataset import get_multi_dataloader
from data.transform import get_transformation
from utils.fitter import MultiHybridFitter
from utils.helper import compose_logging
from utils.loss import FlexLoss
from model.models import CroMAM
from options.base_options import BaseOptions

# Read parameter configuration
opt = BaseOptions()
opt.initialize()
args = opt.parse()
# Setting the random seed
seed_torch(args.random_seed)
# Required WSI magnification of the model
magnification = list(args.magnification.split(','))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
transform = get_transformation(mean=data_stats['mean'], std=data_stats['std'])
num_classes = args.num_classes
# Loading loss function
criterion = FlexLoss(
    device=device,
    outcome=args.outcome
)
scaler = torch.cuda.amp.GradScaler()


def main_function(folds, folds_best):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    loader = get_multi_dataloader(args, transform, folds)

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    writer = compose_logging(args.mode, args.cancer, args.outcome)
    writer['data'].info("Start Fold: %d", folds + 1)
    writer['meta'].info("Start Fold: %d", folds + 1)
    for arg, value in sorted(vars(args).items()):
        if folds == 0:
            writer['meta'].info("Argument %s: %r", arg, value)

    # Initialize the model
    model = CroMAM(
        backbone=args.backbone,
        pretrained=args.pretrain,
        outcome_dim=num_classes,
        dropout=args.dropout,
        device=device,
        branches=args.branches,
        args=args
    )
    if torch.cuda.is_available():
        model = model.cuda(device)
    if folds == 0:
        writer['meta'].info(model)

    # Scheduling training (optimization) and model evaluation.
    hf = MultiHybridFitter(
        model=model,
        writer=writer,
        dataloader=loader,
        checkpoint_to_resume=args.resume,
        timestr=TIMESTR,
        args=args,
        model_name=args.cancer + '_' + args.outcome + "_" + TIMESTR + "_" + args.model_name + "_" + args.magnification,
        loss_function=criterion,
        fold=folds,
        scaler=scaler
    )

    if args.mode == 'train':
        hf.fit(folds, checkpoints_folder=Path(args.checkpoint_dir,
                                              f"{args.cancer}_{args.outcome}_{TIMESTR}_{args.model_name}_{magnification}"))  # 模型从此处开始训练和验证
        folds_best.update({f"fold-{folds + 1}": float(str(hf.best_metric)[:6])})
        return 'val'


if __name__ == '__main__':
    mode = None
    folds_best = {}
    auc_list = []
    for folds in range(5):
        mode = main_function(folds, folds_best)
    if mode == "val":
        for _, v in folds_best.items():
            auc_list.append(v)
        avg_auc = float(str(np.average(auc_list))[:6])
        std_auc = float(str(np.std(auc_list))[:6])
        folds_best.update({f"avg_auc": avg_auc})
        folds_best.update({f"std_auc": std_auc})
        print(f"folds best_metrics:\n {folds_best}")
        print(f"svg_auc: {avg_auc}, std_acu: {std_auc}")
        txtdir = f"./logs/FoldBestResults/{args.cancer}_{args.outcome}_folds_best_metric_{mode}_X{magnification}_{args.model_name}.txt"
        np.savetxt(txtdir, np.array([folds_best]), fmt='%s', delimiter=' ')
    else:
        pass
