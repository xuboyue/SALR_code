import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import math
import torchvision.datasets as datasets
import os
import torchvision.models as models

#### use the following parameters FYI
n_epochs = 100
batch_size_train = 4096
batch_size_test = 4096
learning_rate = 0.8
LR_SGA = 0.005
momentum = 0.9
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
###########

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')


class normalized_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(normalized_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(normalized_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            total_norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # param_norm = p.grad.data.norm(2)
                # total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                d_p = d_p / total_norm

                p.data.add_(-group['lr'], d_p)

        return loss

    def forward(data, target, model, criterion, epoch=0, training=True, optimizer=None):

        losses = 0  # AverageMeter()
        grad_vec = None

        if training:
            optimizer_1 = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
            optimizer_1.zero_grad()  # only zerout at the beginning

        inputs = data
        target = target
        output = model(inputs)
        loss = criterion(output, target)
        losses += loss.item()
        loss.backward()
        # losses += loss.item()
        # optimizer.step() # no step in this case

        # output = model(inputs)
        # loss = criterion(output, target)
        # losses += loss.item()
        # loss.backward()

        # reshape and averaging gradients
        if training:
            for p in model.parameters():
                # p.grad.data.div_(len(data_loader))
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

        # logging.info('{phase} - \t'
        #             'Loss {loss.avg:.4f}\t'
        #             'Prec@1 {top1.avg:.3f}\t'
        #             'Prec@5 {top5.avg:.3f}'.format(
        #              phase='TRAINING' if training else 'EVALUATING',
        #              loss=losses, top1=top1, top5=top5))

        return {'loss': losses}, grad_vec

    class projected_SGD(torch.optim.Optimizer):
        def __init__(self, params, lr=0, momentum=0, dampening=0,
                     weight_decay=0, nesterov=False):
            defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                            weight_decay=weight_decay, nesterov=nesterov)
            if nesterov and (momentum <= 0 or dampening != 0):
                raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            super(projected_SGD, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(projected_SGD, self).__setstate__(state)
            for group in self.param_groups:
                group.setdefault('nesterov', False)

        def step(self, closure=None):
            """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']

                total_norm = 0

                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data

                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    d_p = np.sign(d_p)
                    p.data.add_(-group['lr'], d_p)
                    print(d_p)
            return loss

    def get_minus_cross_entropy(data, target, model, criterion, training=False):

        model.eval()
        result, grads = forward(data, target, model, criterion, 0,
                                training=training, optimizer=None)
        return (-result['loss'], None if grads is None else grads.cpu().numpy().astype(np.float64))

    learning_rate_alpha = 0.001
    epss = 5e-4

    def get_sharpness_ascent(data, target, model_ori, criterion, f, network, optimizer, manifolds=0, ):

        model = model_ori
        f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
        f_x0 = -f_x0
        optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
        inputs = data
        labels = target

        optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
        temp = optimizer.param_groups[0]['params']  # [0]
        for j in range(5):

            i = 0
            optimizer_PGA.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            (loss).backward()
            # with torch.no_grad():
            for p in network.parameters():
                p_1 = p + optimizer_PGA.param_groups[0]['lr'] * p.grad.data
                optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
                i += 1
            i = 0
            with torch.no_grad():
                for name, param in network.named_parameters():
                    param.copy_(optimizer_PGA.param_groups[0]['params'][i])
                    i += 1

        optimizer.zero_grad()
        output = network(data)
        loss_new = criterion(output, target)
        # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
        # print("sharp os:", loss_new.item() - f_x0)

        f_x = loss_new.item()
        # f_x = -f_x
        sharpness = (f_x - f_x0)  # /(1+f_x0)*100

        return sharpness

    def get_sharpness_descent(data, target, model_ori, criterion, f, network, optimizer, manifolds=0):

        model = model_ori
        f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
        f_x0 = -f_x0
        optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
        inputs = data
        labels = target

        optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
        temp = optimizer.param_groups[0]['params']  # [0]

        for i in range(5):
            i = 0
            optimizer_PGA.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            (loss).backward()
            # with torch.no_grad():
            for p in network.parameters():
                p_1 = p - optimizer_PGA.param_groups[0]['lr'] * p.grad.data
                optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
                i += 1
            i = 0
            with torch.no_grad():
                for name, param in network.named_parameters():
                    param.copy_(optimizer_PGA.param_groups[0]['params'][i])
                    i += 1
        optimizer.zero_grad()
        output = network(data)
        loss_new = criterion(output, target)
        # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
        # print("sharp os:", loss_new.item() - f_x0)

        f_x = loss_new.item()
        # f_x = -f_x
        sharpness = (f_x0 - f_x)  # /(1+f_x0)*100

        return sharpness


def forward(data, target, model, criterion, epoch=0, training=True, optimizer=None):

    losses = 0 # AverageMeter()
    grad_vec = None

    if training:
      optimizer_1 = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
      optimizer_1.zero_grad()  # only zerout at the beginning

    inputs = data
    target = target
    output = model(inputs)
    loss = criterion(output, target)
    losses += loss.item()
    loss.backward()
    # losses += loss.item()
        # optimizer.step() # no step in this case

        # output = model(inputs)
        # loss = criterion(output, target)
        # losses += loss.item()
        # loss.backward()

    # reshape and averaging gradients
    if training:
      for p in model.parameters():
        # p.grad.data.div_(len(data_loader))
        if grad_vec is None:
            grad_vec = p.grad.data.view(-1)
        else:
            grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

    #logging.info('{phase} - \t'
    #             'Loss {loss.avg:.4f}\t'
    #             'Prec@1 {top1.avg:.3f}\t'
    #             'Prec@5 {top5.avg:.3f}'.format(
    #              phase='TRAINING' if training else 'EVALUATING',
    #              loss=losses, top1=top1, top5=top5))

    return {'loss': losses}, grad_vec


class projected_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(projected_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(projected_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            total_norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                d_p = np.sign(d_p)
                p.data.add_(-group['lr'], d_p)
                print(d_p)
        return loss



def get_minus_cross_entropy(data, target, model, criterion, training=False):

  model.eval()
  result, grads = forward(data, target, model, criterion, 0,
                 training=training, optimizer=None)
  return (-result['loss'], None if grads is None else grads.cpu().numpy().astype(np.float64))

learning_rate_alpha = 0.001
epss = 5e-4


def get_sharpness_ascent(data, target, model_ori, criterion, f,network, optimizer, manifolds=0, ):

  model = model_ori
  f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
  f_x0 = -f_x0
  optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
  inputs = data
  labels = target

  optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
  temp = optimizer.param_groups[0]['params']  # [0]
  for j in range(5):

      i = 0
      optimizer_PGA.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      (loss).backward()
      # with torch.no_grad():
      for p in network.parameters():
          p_1 = p + optimizer_PGA.param_groups[0]['lr'] * p.grad.data
          optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
          i += 1
      i = 0
      with torch.no_grad():
          for name, param in network.named_parameters():
              param.copy_(optimizer_PGA.param_groups[0]['params'][i])
              i += 1

  optimizer.zero_grad()
  output = network(data)
  loss_new = criterion(output, target)
  # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
  # print("sharp os:", loss_new.item() - f_x0)


  f_x = loss_new.item()
  # f_x = -f_x
  sharpness = (f_x - f_x0) #/(1+f_x0)*100

  return sharpness


def get_sharpness_descent(data, target, model_ori, criterion, f, network, optimizer, manifolds=0):

  model = model_ori
  f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
  f_x0 = -f_x0
  optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
  inputs = data
  labels = target

  optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
  temp = optimizer.param_groups[0]['params']  # [0]

  for i in range(5):
      i = 0
      optimizer_PGA.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      (loss).backward()
      # with torch.no_grad():
      for p in network.parameters():
          p_1 = p - optimizer_PGA.param_groups[0]['lr'] * p.grad.data
          optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
          i += 1
      i = 0
      with torch.no_grad():
          for name, param in network.named_parameters():
              param.copy_(optimizer_PGA.param_groups[0]['params'][i])
              i += 1
  optimizer.zero_grad()
  output = network(data)
  loss_new = criterion(output, target)
  # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
  # print("sharp os:", loss_new.item() - f_x0)

  f_x = loss_new.item()
  # f_x = -f_x
  sharpness = (f_x0 - f_x) #/(1+f_x0)*100

  return sharpness



sharp_array = []
LR_array = []


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input = input.cuda(async=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SALR
        if epoch >= 1 and i % 2 == 0:

            sharp_a = get_sharpness_ascent(input, target, model, criterion, loss.item(), network=model,
                                           optimizer=optimizer, manifolds=0)
            sharp_d = get_sharpness_descent(input, target, model, criterion, loss.item(), network=model,
                                            optimizer=optimizer, manifolds=0)
            sharp = sharp_a + sharp_d
            sharp_array.append(sharp)
            sharp_array_1 = np.array(sharp_array)
            # sharp_array_1 = reject_outliers(sharp_array_1, 3)
            # sharp = sharp / sharp_array.max()
            if sharp > np.percentile(sharp_array_1, 51) or sharp < np.percentile(sharp_array_1, 49):
                optimizer.param_groups[0]['lr'] = 1 * learning_rate * sharp / np.percentile(sharp_array_1,
                                                                                            50)  # ( (np.percentile(sharp_array_1, 85) ) ) # (sharp_array.mean() + sharp_array.std()) # sharp_array.max()
            else:
                optimizer.param_groups[0]['lr'] = 1 * learning_rate
            if optimizer.param_groups[0]['lr'] > learning_rate * 5: optimizer.param_groups[0]['lr'] = learning_rate * 5
            # print("Base is " , learning_rate, "; new LR is ", optimizer.param_groups[0]['lr'], "; sharp is ", sharp)
            LR_array.append(optimizer.param_groups[0]['lr'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
