import argparse
import os
import numpy as np
import torch
from cross.util.utils import save_cls_text_configuration
import yaml
# Make pretrained classifier and model prediction
from cross.COTEACHING import Coteaching
from cross.COTEACHING_Cross import Coteaching_Cross
from torchlight import DictAction

####################################################################################################################
parser = argparse.ArgumentParser(description="main")
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

#args from ctrgcn
parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
parser.add_argument(
    '--work-dir',
    default='./work_dir/temp',
    help='the work folder for storing results')

parser.add_argument('-model_saved_name', default='')

# processor
parser.add_argument(
    '--phase', default='train', help='must be train or test')
parser.add_argument(
    '--save-score',
    type=str2bool,
    default=False,
    help='if ture, the classification score will be stored')
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='the interval for printing messages (#iteration)')
parser.add_argument(
    '--save-interval',
    type=int,
    default=1,
    help='the interval for storing models (#iteration)')
parser.add_argument(
    '--save-epoch',
    type=int,
    default=30,
    help='the start epoch to save model (#iteration)')
parser.add_argument(
    '--eval-interval',
    type=int,
    default=5,
    help='the interval for evaluating models (#iteration)')
parser.add_argument(
    '--print-log',
    type=str2bool,
    default=True,
    help='print logging or not')
parser.add_argument(
    '--show-topk',
    type=int,
    default=[1, 5],
    nargs='+',
    help='which Top K accuracy will be shown')

# feeder
parser.add_argument(
    '--feeder', default='feeder.feeder', help='data loader will be used')
parser.add_argument(
    '--num-worker',
    type=int,
    default=32,
    help='the number of worker for data loader')
parser.add_argument(
    '--train-feeder-args',
    action=DictAction,
    default=dict(),
    help='the arguments of data loader for training')
parser.add_argument(
    '--test-feeder-args',
    action=DictAction,
    default=dict(),
    help='the arguments of data loader for test')

# model
parser.add_argument('--model', default=None, help='the model will be used')
parser.add_argument(
    '--model-args',
    action=DictAction,
    default=dict(),
    help='the arguments of model')
parser.add_argument(
    '--weights',
    default=None,
    help='the weights for network initialization')
parser.add_argument(
    '--ignore-weights',
    type=str,
    default=[],
    nargs='+',
    help='the name of weights which will be ignored in the initialization')

# optim
parser.add_argument(
    '--base-lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument(
    '--step',
    type=int,
    default=[20, 40, 60],
    nargs='+',
    help='the epoch where optimizer reduce the learning rate')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    nargs='+',
    help='the indexes of GPUs for training or testing')
parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
parser.add_argument(
    '--nesterov', type=str2bool, default=False, help='use nesterov or not')
parser.add_argument(
    '--batch-size', type=int, default=256, help='training batch size')
parser.add_argument(
    '--test-batch-size', type=int, default=256, help='test batch size')
parser.add_argument(
    '--start-epoch',
    type=int,
    default=0,
    help='start training from which epoch')
parser.add_argument(
    '--num-epoch',
    type=int,
    default=80,
    help='stop training in which epoch')
parser.add_argument(
    '--weight-decay',
    type=float,
    default=0.0005,
    help='weight decay for optimizer')
parser.add_argument(
    '--lr-decay-rate',
    type=float,
    default=0.1,
    help='decay rate for learning rate')
parser.add_argument('--warm_up_epoch', type=int, default=0)

# data condition
parser.add_argument('--dataset', type=str, default='MNIST', help = 'MNIST, FMNIST, CIFAR10, Food, Clothing')
parser.add_argument('--noise_type', type=str, default='clean', help='clean, sym, asym, idn, idnx')
parser.add_argument('--noisy_ratio', type=float, default=None, help='between 0 and 1')


# classifier condition
parser.add_argument('--class_method', type=str, default=None, help='classifier method')

# experiment condition
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1, help = "Learning rate (Default : 1e-3)")

# etc
parser.add_argument('--set_gpu', type=str, default='0', help='gpu setting')
parser.add_argument('--data_dir', type=str, default=None)
# added
parser.add_argument(
    '--config',
    default='./config/cross_training/default.yaml',
    help='path to the configuration file')

####################################################################################################################
if __name__ == '__main__':
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
        
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset=='NTU60':
        args.model = 'CTRGCN'
        args.dropout = 0.0
    # else:
    args.data_name = args.dataset + '_' + str(100 * args.noisy_ratio) + '_' + args.noise_type + '.pk'
    data_noise = args.dataset + '_' + args.model + '_' + args.noise_type + '_' + str(100 * args.noisy_ratio)

    if args.dataset=="NTU60":
        args.pre_epoch=1
        args.n_classes = 60
        args.total_epochs = 55

        args.data_name = args.dataset
        # args.noise_type = 'clean'
        args.causalnl_z_dim = 100
        args.batch_size = 64

    else:  # wrong data name
        print('Wrong dataset name')
        args.n_classes = None
        args.data_name = None
        args.total_epochs = None

    leaf_dir = args.class_method + '_pre_epoch_' + str(args.pre_epoch) + '_epoch_' + str(args.total_epochs) + '_seed_0'

    args.cls_dir = os.path.join('classifier_model', 'result', data_noise, leaf_dir) + '/'
    args.model_dir = os.path.join('classifier_model', 'result_model', data_noise,
                                  args.class_method) + '/pre_epoch_' + str(args.pre_epoch) + \
                     '_epoch_' + str(args.total_epochs) + '_dropout_ratio_' + str(args.dropout * 100) + '_seed_0' + '_'
    # args.model_folder=os.path.join('classifier_model', 'result_model', data_noise,
                                #   args.class_method)+'/'
    # classifier model
    os.makedirs(args.cls_dir, exist_ok=True)
    os.makedirs(os.path.join('classifier_model/result_model/', data_noise, args.class_method), exist_ok=True)


    model = Coteaching(args)


    pre_acc, train_class_acc, train_label_acc, test_acc, epoch = model.run()
    save_cls_text_configuration(args, pre_acc, train_class_acc, train_label_acc, test_acc, epoch)