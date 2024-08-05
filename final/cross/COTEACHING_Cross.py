#############################################################################
# Original paper: Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels (NIPS 18) https://proceedings.neurips.cc/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf
# Official code: https://github.com/bhanML/Co-teaching
#############################################################################
from model.ctrgcn import *

import torch.optim as optim
import numpy as np
import time
import os
import random
from cross.util.dataloader import load_dataset
from cross.util.utils import plot_, save_data
import sys
from tqdm import tqdm
import pickle

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


class Coteaching_Cross:
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.n_classes
        self.data_name = self.args.data_name
        self.time = time.time()
        if args.dataset=="NTU60":
            Feeder = import_class(self.args.feeder)
            
            self.trainloader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.train_feeder_args),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)
            
            self.testloader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.test_feeder_args),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
            
            self.len_train, self.len_test=len(Feeder(**self.args.train_feeder_args)),len(Feeder(**self.args.test_feeder_args))
            
        else:
            self.dataset = load_dataset(self.data_name, batch_size=args.batch_size, dir=args.data_dir)
            Ttloader, self.len_train, self.len_test = self.dataset.train_test()
            self.trainloader, self.testloader = Ttloader['train'], Ttloader['test']
        print("test_cm===>",self.args.train_feeder_args['cm'])
        print('\n===> Training Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.args.model == 'CNN_MNIST':
            self.net1 = CNN_MNIST(self.n_classes, self.args.dropout)
            self.net2 = CNN_MNIST(self.n_classes, self.args.dropout)
        elif self.args.model == 'CNN_CIFAR':
            self.net1 = CNN(self.n_classes, self.args.dropout)
            self.net2 = CNN(self.n_classes, self.args.dropout)
        elif self.args.model == 'Resnet50Pre':
            self.net1 = ResNet50Pre(self.n_classes, self.args.dropout)
            self.net2 = ResNet50Pre(self.n_classes, self.args.dropout)
        elif self.args.model == 'CTRGCN':
            self.net1_j = CTRGCN(**self.args.model_args)
            self.net1_b= CTRGCN(**self.args.model_args)
            self.net1_m= CTRGCN(**self.args.model_args)
            
            self.net2_j = CTRGCN(**self.args.model_args)
            self.net2_b = CTRGCN(**self.args.model_args)
            self.net2_m = CTRGCN(**self.args.model_args)



        self.net1_j.to(self.device)
        self.net1_b.to(self.device)
        self.net1_m.to(self.device)        
        self.net2_j.to(self.device)
        self.net2_b.to(self.device)
        self.net2_m.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # mean
        self.optimizer1_j = optim.Adam(self.net1_j.parameters(), lr=self.args.lr)
        self.optimizer1_b = optim.Adam(self.net1_b.parameters(), lr=self.args.lr)
        self.optimizer1_m = optim.Adam(self.net1_m.parameters(), lr=self.args.lr)
        self.optimizer2_j = optim.Adam(self.net2_j.parameters(), lr=self.args.lr)
        self.optimizer2_b = optim.Adam(self.net2_b.parameters(), lr=self.args.lr)
        self.optimizer2_m = optim.Adam(self.net2_m.parameters(), lr=self.args.lr)

        # self.optimizer1 = optim.SGD(
        #         self.model.parameters(),
        #         lr=self.arg.base_lr,
        #         momentum=0.9,
        #         nesterov=self.arg.nesterov,
        #         weight_decay=self.arg.weight_decay)
        # self.optimizer2 = optim.SGD(
        # self.model.parameters(),
        # lr=self.arg.base_lr,
        # momentum=0.9,
        # nesterov=self.arg.nesterov,
        # weight_decay=self.arg.weight_decay)
    def define_configuration(self):
        self.metric = nn.CrossEntropyLoss(reduction='none').to(self.device)
        forget_rate = self.args.noisy_ratio
        num_gradual = 10
        self.rate_schedule = np.ones(self.args.total_epochs) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)

        if self.args.dataset in ['MNIST','FMNIST','CIFAR10']:
            lr_decay_start = 80
        elif self.args.dataset in ['CIFAR100', 'Animal']:
            lr_decay_start = 100
        else:
            lr_decay_start = int(self.args.total_epochs * 0.6)

        self.alpha_plan = [self.args.lr]*self.args.total_epochs
        self.beta1_plan = [0.9]*self.args.total_epochs

        for i in range(lr_decay_start, self.args.total_epochs):
            self.alpha_plan[i] = float(self.args.total_epochs - i) / (self.args.total_epochs - lr_decay_start) * self.args.lr
            self.beta1_plan[i] = 0.1

        return

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
        return

    def loss_function(self, forget_rate, output1, output2, labels):
        loss_1 = self.metric(output1, labels)
        loss_2 = self.metric(output2, labels)

        _, net1_index = torch.sort(loss_1)
        _, net2_index = torch.sort(loss_2)
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(labels))
        net1_index, net2_index = net1_index[:num_remember], net2_index[:num_remember]
        loss1 = self.criterion(output1[net2_index], labels[net2_index])
        loss2 = self.criterion(output2[net1_index], labels[net1_index])

        return loss1, loss2

    def update_model(self, epoch):
        self.net1_j.train()
        self.net1_b.train()
        self.net1_m.train()
        self.net2_j.train()
        self.net2_b.train()
        self.net2_m.train()
        
        self.adjust_learning_rate(self.optimizer1_j, epoch)
        self.adjust_learning_rate(self.optimizer1_b, epoch)
        self.adjust_learning_rate(self.optimizer1_m, epoch)
        self.adjust_learning_rate(self.optimizer2_j, epoch)
        self.adjust_learning_rate(self.optimizer2_b, epoch)
        self.adjust_learning_rate(self.optimizer2_m, epoch)
        
        epoch_loss, epoch_class_accuracy1, epoch_class_accuracy2, epoch_label_accuracy1, epoch_label_accuracy2 = 0, 0, 0, 0, 0
        process = tqdm(self.trainloader, ncols=40)
        for index, (images, b_images, v_images, labels, _)  in enumerate(process):
            images = images.to(self.device)
            labels = labels.to(self.device)

            b_images = b_images.to(self.device)
            v_images = v_images.to(self.device)
            _, outputs1_j = self.net1_j(images)
            _, outputs1_b = self.net1_b(b_images)
            _, outputs1_m = self.net1_m(v_images)
            _, outputs2_j = self.net2_j(images)
            _, outputs2_b = self.net2_b(b_images)
            _, outputs2_m = self.net2_m(v_images)
            
            outputs1=0.6*outputs1_j+0.4*outputs1_b+0.4*outputs1_m
            outputs2=0.6*outputs2_j+0.4*outputs2_b+0.4*outputs2_m
            # loss
            loss1, loss2 = self.loss_function(self.rate_schedule[epoch], outputs1, outputs2, labels)

            self.optimizer1_j.zero_grad()
            self.optimizer1_b.zero_grad()
            self.optimizer1_m.zero_grad()
            loss1.backward()
            self.optimizer1_j.step()
            self.optimizer1_b.step()
            self.optimizer1_m.step()
            epoch_loss += loss1.item() * len(labels)

            self.optimizer2_j.zero_grad()
            self.optimizer2_b.zero_grad()
            self.optimizer2_m.zero_grad()
            loss2.backward()
            self.optimizer2_j.step()
            self.optimizer2_b.step()
            self.optimizer2_m.step()
            epoch_loss += loss2.item() * len(labels)

            # accuracy
            _, model_label1 = torch.max(outputs1, dim=1)
            _, model_label2 = torch.max(outputs2, dim=1)
            # no true label
            epoch_class_accuracy1 += (labels == model_label1).sum().item()
            epoch_label_accuracy1 += (labels == model_label1).sum().item()
            # no true label
            epoch_class_accuracy2 += (labels == model_label2).sum().item()
            epoch_label_accuracy2 += (labels == model_label2).sum().item()

        epoch_class_accuracy = max(epoch_class_accuracy1,epoch_class_accuracy2)
        epoch_label_accuracy = max(epoch_label_accuracy1,epoch_label_accuracy2)
        time_elapse = time.time() - self.time
        return epoch_loss/2, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        with torch.no_grad():
            score_frag_1 = []
            score_frag_2 = []
            # calculate test accuracy
            self.net1_j.eval()
            self.net1_b.eval()
            self.net1_m.eval()
            self.net2_j.eval()
            self.net2_b.eval()
            self.net2_m.eval()
            epoch_class_accuracy1, epoch_class_accuracy2 = 0, 0
            process = tqdm(self.testloader, ncols=40)
            for index, (images,b_images, v_images, classes, index) in enumerate(process):
                images = images.to(self.device)
                classes = classes.to(self.device)
                b_images = b_images.to(self.device)
                v_images = v_images.to(self.device)
                
                _, outputs1_j = self.net1_j(images)
                _, outputs1_b = self.net1_b(b_images)
                _, outputs1_m = self.net1_m(v_images)
                _, outputs2_j = self.net2_j(images)
                _, outputs2_b = self.net2_b(b_images)
                _, outputs2_m = self.net2_m(v_images)
                
                outputs1=0.6*outputs1_j+0.4*outputs1_b+0.4*outputs1_m
                outputs2=0.6*outputs2_j+0.4*outputs2_b+0.4*outputs2_m
 
                #save outputs
                score_frag_1.append(outputs1.data.cpu().numpy())
                score_frag_2.append(outputs2.data.cpu().numpy())
                # accuracy
                model_label1 = np.argmax(outputs1.detach().cpu().numpy(), axis=1)
                model_label2 = np.argmax(outputs2.detach().cpu().numpy(), axis=1)
                epoch_class_accuracy1 += (classes.cpu().numpy() == model_label1).sum().item()
                epoch_class_accuracy2 += (classes.cpu().numpy() == model_label2).sum().item()

            score1 = np.concatenate(score_frag_1)
            score2 = np.concatenate(score_frag_2)
            epoch_class_accuracy = max(epoch_class_accuracy1, epoch_class_accuracy2)
            time_elapse = time.time() - self.time
        return epoch_class_accuracy, int(epoch_class_accuracy1<epoch_class_accuracy2) ,time_elapse, score1, score2

    def save_result(self, epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc):
        self.loss_train.append(epoch_loss / self.len_train)
        self.train_class_acc.append(epoch_class_acc / self.len_train)
        self.train_label_acc.append(epoch_label_acc / self.len_train)
        self.test_acc.append(epoch_test_acc/self.len_test)

        print('Train', epoch_loss / self.len_train, epoch_class_acc / self.len_train, epoch_label_acc / self.len_train)
        print('Test', epoch_test_acc / self.len_test)

        return

    def run(self):
        # initialize
        self.define_configuration()

        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []
        self.epoch_acc_clean, self.epoch_class_acc_noisy, self.epoch_label_acc_noisy = [], [], []

        best_test_acc=0.0
        pre_model_path=""
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model(epoch)
            epoch_test_acc, network_name, time_test, score1, score2 = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.save_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)
            if epoch_test_acc>best_test_acc:
                print("==save best result==")
                if epoch>0:
                    # file=os.listdir(self.args.model_folder)
                    os.remove(pre_model_path)
                    os.remove(pre_score_path)
                        # save one network with higher test accuracy
                if network_name==0:
                    net = self.net1
                    score=score1
                else:
                    net= self.net2
                    score=score2
                # score
                score_dict = score
                with open(self.args.model_dir + str(epoch) +'_'+'score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)
                # model
                torch.save(net.state_dict(), self.args.model_dir + str(epoch) +'_'+'classifier.pk')
                pre_model_path=self.args.model_dir + str(epoch) +'_'+'classifier.pk'
                pre_score_path=self.args.model_dir + str(epoch) +'_'+'score.pkl'
                # new baseline of test accuracy
                best_test_acc=epoch_test_acc

        plot_(self.args.cls_dir, self.loss_train, 'train_loss')
        plot_(self.args.cls_dir, self.train_class_acc, 'train_class_accuracy')
        plot_(self.args.cls_dir, self.train_label_acc, 'train_label_accuracy')
        plot_(self.args.cls_dir, self.test_acc, 'test_accuracy')




        # save_data(os.path.join(self.args.cls_dir, self.data_name), self.dataset, self.device, net)
        # torch.save(net.state_dict(), self.args.model_dir + 'classifier.pk')

        return 0, self.train_class_acc[-1], self.train_label_acc[-1], self.test_acc[-1], epoch
