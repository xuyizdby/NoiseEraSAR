#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
from model.ctrgcn import *
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
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
        default=0,
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
    
    parser.add_argument(
    '--gss-forget-rate',
    type=float,
    default=0.8,
    help='forget rate for global sample selection')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
    
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
            
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=False,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        self.model1=Model(**self.arg.model_args)
        self.model1.load_state_dict(torch.load('pretrained/cs/08/joint.pk'))
        self.model1 = self.model1.cuda(self.output_device)

        self.model2=Model(**self.arg.model_args)
        self.model2.load_state_dict(torch.load('pretrained/cs/08/bone.pk'))        
        self.model2 = self.model2.cuda(self.output_device)
        
        self.model3=Model(**self.arg.model_args)
        self.model3.load_state_dict(torch.load('pretrained/cs/08/vel.pk'))        
        self.model3 = self.model3.cuda(self.output_device)

        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.metric=nn.CrossEntropyLoss(reduction='none').cuda(output_device)
        # print(self.model1.state_dict().keys())

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer1 = optim.SGD(
                self.model1.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            # The above code is accessing the `optimizer2` attribute of the current object or class.
            self.optimizer2 = optim.SGD(
                self.model2.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            # The above code is accessing the `optimizer2` attribute of the current object or class.
            self.optimizer3 = optim.SGD(
                self.model3.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer1 = optim.Adam(
                self.model1.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
            self.optimizer2 = optim.Adam(
                self.model2.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
            self.optimizer3 = optim.Adam(
                self.model3.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
        
    def adjust_learning_rate_sub(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer1.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer2.param_groups:
                param_group['lr'] = lr
            
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def train(self, epoch, save_model=False,split='train'):
        self.model.train()
        print(self.model1.training,self.model2.training,self.model3.training)
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader[split]
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, data_b, data_v,label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                data_b = data_b.float().cuda(self.output_device)
                data_v = data_v.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data,data_b,data_v)
            value, predict_label = torch.max(output.data, 1)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            acc = torch.mean((predict_label == label).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
                            
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, data_b, data_v, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    data_b = data_b.float().cuda(self.output_device)
                    data_v = data_v.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data,data_b,data_v)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)
                
    def find_clean_samples(self, loss_1, loss_2, loss_3, labels, index, forget_rate=0.8):

        loss_1 = torch.tensor(loss_1)
        loss_2 = torch.tensor(loss_2)
        loss_3 = torch.tensor(loss_3)
        
        _, net1_index = torch.sort(loss_1)
        _, net2_index = torch.sort(loss_2)
        _, net3_index = torch.sort(loss_3)
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(labels))
        net1_index, net2_index, net3_index = net1_index[:num_remember], net2_index[:num_remember], net3_index[:num_remember]
        
        net1_index, net2_index, net3_index = net1_index.data.cpu().numpy().tolist(), net2_index.data.cpu().numpy().tolist(), net3_index.data.cpu().numpy().tolist()
        
        clean_indexes=list(set(net1_index).union(set(net2_index),set(net3_index)))
        clean_indexes=index[clean_indexes]
        return clean_indexes
      
    def eval_sub(self, epoch, save_score=False, loader_name=['train'], wrong_file=None, result_file=None):
        # self.metric=nn.CrossEntropyLoss(reduction='none').cuda(output_device)
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        for ln in loader_name:
            loss_value1 = []
            loss_value2 = []
            loss_value3 = []
            score_frag1 = []
            score_frag2 = []
            score_frag3 = []
            label_list = []
            pred_list = []
            acc_value1 = []
            acc_value2 = []
            acc_value3 = []
            indexes = []

            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            test_len=len(self.data_loader[ln].dataset)

            for batch_idx, (data, data_b, data_v, label, index) in enumerate(process):

                label_list.append(label.data.cpu().numpy().tolist())
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    data_b = data_b.float().cuda(self.output_device)
                    data_v = data_v.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output1 = self.model1(data)
                    output2 = self.model2(data_b)
                    output3 = self.model3(data_v)
                    loss1 = self.metric(output1, label)
                    loss2 = self.metric(output2, label)
                    loss3 = self.metric(output3, label)
                    
                    indexes.append(index.data.cpu().numpy().tolist())
                    loss_value1.append(loss1.data.cpu().numpy().tolist())
                    loss_value2.append(loss2.data.cpu().numpy().tolist())
                    loss_value3.append(loss3.data.cpu().numpy().tolist())
                    score_frag1.append(output1.data.cpu().numpy())
                    score_frag2.append(output2.data.cpu().numpy())
                    score_frag3.append(output3.data.cpu().numpy())
                    
                    _, predict_label1 = torch.max(output1.data, 1)
                    _, predict_label2 = torch.max(output2.data, 1)
                    _, predict_label3 = torch.max(output3.data, 1)
                    pred_list.append(predict_label1.data.cpu().numpy())
                    acc1 = torch.sum((predict_label1 == label.data).float())
                    acc2 = torch.sum((predict_label2 == label.data).float())
                    acc3 = torch.sum((predict_label3 == label.data).float())
                    # print(acc1)
                    acc_value1.append(acc1.data.item())
                    acc_value2.append(acc2.data.item())
                    acc_value3.append(acc3.data.item())


                    step += 1
                    
            self.print_log(
                'Mean val acc: {:.2f}%. and {:.2f}%. and {:.2f}%.'.format((np.sum(acc_value1)/test_len)*100,(np.sum(acc_value2)/test_len)*100,(np.sum(acc_value3)/test_len)*100))


            print("====generating clean samples====")
            label_list=np.concatenate(label_list)
            loss_value1=np.concatenate(loss_value1) 
            loss_value2=np.concatenate(loss_value2)  
            loss_value3=np.concatenate(loss_value3)   
            indexes=np.concatenate(indexes)
            self.clean_indexes=self.find_clean_samples(loss_value1,loss_value2,loss_value3,label_list,indexes,self.arg.gss_forget_rate)  
            print(self.clean_indexes.shape)   

    def start(self):
        if self.arg.phase == 'train':
            print(self.arg.train_feeder_args['data_path'])
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model1)}')

            self.eval_sub(1, save_score=self.arg.save_score, loader_name=['train'])
            
            Feeder = import_class(self.arg.feeder)
            self.data_loader['train_2'] = torch.utils.data.DataLoader(
                dataset=Feeder(clean_indexes=self.clean_indexes,**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                pin_memory=False,
                worker_init_fn=init_seed)
            print(len(Feeder(clean_indexes=self.clean_indexes,**self.arg.train_feeder_args)))

            self.model=Model_moe([self.model1,self.model2,self.model3],60,model_args=self.arg.model_args)
            self.model = self.model.cuda(self.output_device)
            # fix the triple experts
            # for name, param in self.model.named_parameters():
            #     if 'sub_models' in name:
            #         param.requires_grad=False
            # params = filter(lambda p: p.requires_grad, self.model.parameters())
            if self.arg.optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.arg.base_lr,
                    momentum=0.9,
                    nesterov=self.arg.nesterov,
                    weight_decay=self.arg.weight_decay)
            elif self.arg.optimizer == 'Adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay)
            # self.eval(0, save_score=self.arg.save_score, loader_name=['test'])
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                self.train(epoch, save_model=save_model,split='train_2')

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

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

    arg = parser.parse_args()
    # init_seed(arg.seed)
    processor = Processor(arg)
    init_seed(arg.seed)
    processor.start()
