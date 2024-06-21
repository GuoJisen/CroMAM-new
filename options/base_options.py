import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Patient Level Prediction Model')

    def initialize(self):
        # specify model structure
        self.parser.add_argument('--cancer',
                                 type=str, default='LGG',
                                 help='Cancer type')
        self.parser.add_argument('--backbone',
                                 type=str, default='resnet-18',
                                 help='backbone model, for example, resnet-50, mobilenet, etc')
        self.parser.add_argument('--model_name',
                                 type=str, default='CroMAM')
        self.parser.add_argument('--outcome',
                                 type=str, default='idh',
                                 help='name of the outcome variable')
        self.parser.add_argument('--num-classes',
                                 type=int, default=2,
                                 help='number of outputs of the model, only used for classification')
        self.parser.add_argument('--magnification',
                                 type=str, default=10,
                                 help='magnification level')

        # specify the path of the meta files
        self.parser.add_argument('--val-meta',
                                 type=str, default='./dataset/',
                                 help='path to the meta file for the evaluation portion')
        self.parser.add_argument('--train-meta',
                                 type=str, default='./dataset/',
                                 help='path to the meta file for the training portion')
        self.parser.add_argument('--patch-meta',
                                 type=str, default='./dataset/patches_meta',
                                 help='path to the meta file for the training portion')

        # specify patch manipulations
        self.parser.add_argument('--crop-size',
                                 type=int, default=224,
                                 help='size of the crop')
        self.parser.add_argument('--num-patches',
                                 type=int, default=16,
                                 help='number of patches to select from one patient during one iteration')

        # learning rate
        self.parser.add_argument('--lr-backbone',
                                 type=float, default=3e-7,
                                 help='learning rate for the backbone module')
        self.parser.add_argument('--lr-fusion',
                                 type=float, default=3e-5,
                                 help='learning rate for the feature fusin module ')
        self.parser.add_argument('--lr-classifier',
                                 type=float, default=3e-5,
                                 help='learning rate for the classifier module')

        # specify experiment details
        self.parser.add_argument('-m', '--mode',
                                 type=str, default='train',
                                 help='mode, train or test')
        self.parser.add_argument('--patience',
                                 type=int, default=10,
                                 help='break the training after how number of epochs of no improvement')
        self.parser.add_argument('--epochs',
                                 type=int, default=100,
                                 help='total number of epochs to train the model')
        self.parser.add_argument('--pretrain',
                                 action='store_true', default=False,
                                 help='whether use a pretrained backbone')
        self.parser.add_argument('--random-seed',
                                 type=int, default=88,
                                 help='random seed of the model')
        self.parser.add_argument('--resume',
                                 type=str, default='',
                                 help='path to the checkpoint file')

        # data specific arguments
        self.parser.add_argument('-b', '--batch-size',
                                 type=int, default=16,
                                 help='batch size')
        self.parser.add_argument('-vb', '--vbatch-size',
                                 type=int, default=1,
                                 help='val batch size')
        self.parser.add_argument('--repeats-per-epoch',
                                 type=int, default=8,
                                 help='how many times to select one patient during each iteration')
        self.parser.add_argument('--num-workers',
                                 type=int, default=0,
                                 help='number of CPU threads')

        # model regularization
        self.parser.add_argument('--dropout',
                                 type=float, default=0,
                                 help='dropout rate, not implemented yet')
        self.parser.add_argument('--wd-backbone',
                                 type=float, default=1e-4,
                                 help='intensity of the weight decay for the backbone module')
        self.parser.add_argument('--wd-fusion',
                                 type=float, default=1e-4,
                                 help='intensity of the weight decay for the feature fusion module')
        self.parser.add_argument('--wd-classifier',
                                 type=float, default=1e-4,
                                 help='intensity of the weight decay for the classifier module')

        # evaluation details
        self.parser.add_argument('--sample-id',
                                 action='store_true', default=False,
                                 help='if true, sample patches by patient; otherwise evaluate the model on all patches')
        self.parser.add_argument('--num-val',
                                 type=int, default=64,
                                 help='number of patches to select from one patient during validation')

        # model monitoring
        self.parser.add_argument('--timestr',
                                 type=str, default='',
                                 help='time stamp of the model')
        self.parser.add_argument('--log-freq',
                                 type=int, default=10,
                                 help='how frequent (in steps) to print logging information')
        self.parser.add_argument('--save-interval',
                                 type=int, default=1,
                                 help='how frequent (in epochs) to save the model checkpoint')
        self.parser.add_argument('--checkpoint-dir',
                                 type=str, default='/media/dell/data/DATA/pj_data/multi-plp/checkpoints',
                                 help='path to model checkpoint directory')
        self.parser.add_argument('--branches',
                                 type=int, default=1,
                                 help='x branches net')

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args
