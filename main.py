"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os

import datetime

import pickle
import random
import numpy as np

import torch
from torch.autograd import Variable

from model import RN, CNN_MLP, RFS


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relations-from-Stream sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'RFS'], default='RN', 
                    help='model type')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from model epoch stored')
parser.add_argument('--template', type=str, default='{}_2item-span_{:03d}.pth',  # default='%%s-%%03d.pkl',  
                    help='template for model name, expecting model type and integer epoch')

parser.add_argument('--gumbel_temp', type=float, default=-1,
                    help='Gumbel temperature (if >0)')
parser.add_argument('--gumbel_hurdle', type=float, default=0,
                    help='Multiply temperature by 90%% if training is over this hurdle')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    print("Running with GPU enabled")
    torch.cuda.manual_seed(args.seed)

if args.model=='RFS': 
  model = RFS(args)
elif args.model=='CNN_MLP': 
  model = CNN_MLP(args)
else:
  model = RN(args)

# For loading the data (possibly a symlink to relational-networks/data)
data_dirs = './data'
  
bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 11)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    t0 = datetime.datetime.now()
    # total_accuracy: ToDo
    
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rel = model.train_(input_img, input_qst, label)

        tensor_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:3.0f}%)] Relations accuracy: {:3.0f}% | Non-relations accuracy: {:3.0f}%'.format(
                    epoch, batch_idx * bs * 2, 
                    len(rel[0]) * 2, 
                    100. * batch_idx * bs/ len(rel[0]), 
                    accuracy_rel, accuracy_norel
                 ))
                 
    epoch_duration = (datetime.datetime.now()-t0).total_seconds()
    print("  This epoch elapsed time : %.0fsecs, remaining : %.0fmins" % (epoch_duration, (args.epochs-epoch)*epoch_duration/60.))
                                                                                                                           
            

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n  Test set after epoch {:2d} : Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
                epoch, accuracy_rel, accuracy_norel))

    
def load_data():
    print('loading data...')
    filename = os.path.join(data_dirs, 'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)
    

rel_train, rel_test, norel_train, norel_test = load_data()

if args.resume>0:
    filename = args.template % (args.model, args.resume, )
    print(filename)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(args.resume+1, args.resume+args.epochs + 1):
    train(epoch, rel_train, norel_train)
    test(epoch, rel_test, norel_test)
    model.save_model(args.template, epoch)
