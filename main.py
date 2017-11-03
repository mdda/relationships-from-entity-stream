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
from model_hard import RFSH


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relations-from-Stream sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'RFS', 'RFSH'], default='RN', 
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

parser.add_argument('--rnn_hidden_size', type=int, default=32, 
                    help='size of RNN hidden vectors (default: 32)')

parser.add_argument('--process_coords', action='store_true', default=False,
                    help='Process the coordinates with 1x1 covolutions, instead of just concatting')

parser.add_argument('--highway', type=int, default=0, 
                    help='add highway network of type (>0)')

parser.add_argument('--seq_len', type=int, default=2, 
                    help='length of entities question and answer streams')

parser.add_argument('--coord_extra_len', type=int, default=2, 
                    help='size of coordinate information added per position')

parser.add_argument('--train_tricky', action='store_true', default=False,
                    help='Also learn the additional "tricky relationships"')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Stores interim results in the model for external examination')

# Extras for 'Hard Attention' model(s)
#parser.add_argument('--gumbel_temp', type=float, default=-1,
#                    help='Gumbel temperature (if >0)')
#parser.add_argument('--gumbel_hurdle', type=float, default=0,
#                    help='Multiply temperature by 90%% if training is over this hurdle')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#print(args.seq_len)

args.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    print("Running with GPU enabled")
    torch.cuda.manual_seed(args.seed)

if args.model=='RFS': 
  model = RFS(args)
elif args.model=='RFHS': 
  model = RFHS(args)
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
    img = torch.from_numpy(np.asarray(data[0][bs*i : bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i : bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i : bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, norel, birel, trirel):
    model.train()

    if not len(birel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(norel)
    random.shuffle(birel)
    random.shuffle(trirel)

    norel = cvt_data_axis(norel)
    birel = cvt_data_axis(birel)
    trirel = cvt_data_axis(trirel)

    t0 = datetime.datetime.now()
    accuracy_norels, accuracy_birels, accuracy_trirels = [], [], []
    
    for batch_idx in range(len(norel[0]) // bs):
        tensor_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)
        accuracy_norels.append(accuracy_norel)
        
        tensor_data(birel, batch_idx)
        accuracy_birel = model.train_(input_img, input_qst, label)
        accuracy_birels.append(accuracy_birel)

        accuracy_trirel, example_factor = 0.0, 2
        if args.train_tricky:
          example_factor = 3
          tensor_data(trirel, batch_idx)
          accuracy_trirel = model.train_(input_img, input_qst, label)
          accuracy_trirels.append(accuracy_trirel)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:3.0f}%)] Non-relations accuracy: {:3.0f}% | Relations accuracy: {:3.0f}% | Tricky accuracy: {:3.0f}% | '.format(
                    epoch, batch_idx * bs * example_factor, 
                    len(norel[0]) * example_factor, 
                    100. * batch_idx * bs/ len(norel[0]), 
                    accuracy_norel, accuracy_birel, accuracy_trirel, 
                 ))
                 
    av_accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    av_accuracy_birel = sum(accuracy_birels) / len(accuracy_birels)
    av_accuracy_trirel = sum(accuracy_trirels) / len(accuracy_norels)  # The trirels and norels should be the same length in any case
    
    epoch_duration = (datetime.datetime.now()-t0).total_seconds()
    print("  This epoch elapsed time : %.0fsecs, remaining : %.0fmins" % (epoch_duration, (args.resume+args.epochs-epoch)*epoch_duration/60.))
                                                                                                                           
            

def test(epoch, norel, birel, trirel):
    model.eval()
    if not len(birel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    norel = cvt_data_axis(norel)
    birel = cvt_data_axis(birel)
    trirel = cvt_data_axis(trirel)

    accuracy_norels, accuracy_birels, accuracy_trirels = [], [], []
    for batch_idx in range(len(norel[0]) // bs):
        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

        tensor_data(birel, batch_idx)
        accuracy_birels.append(model.test_(input_img, input_qst, label))

        if args.train_tricky:
          tensor_data(trirel, batch_idx)
          accuracy_trirels.append(model.test_(input_img, input_qst, label))

    av_accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    av_accuracy_birel = sum(accuracy_birels) / len(accuracy_birels)
    av_accuracy_trirel = sum(accuracy_trirels) / len(accuracy_norels)  # The trirels and norels should be the same length in any case
    
    print('\n  Test set after epoch {:2d} : Non-relation accuracy: {:.0f}% | Relation accuracy: {:.0f}% | Tricky accuracy: {:.0f}% \n'.format(
                epoch, av_accuracy_norel, av_accuracy_birel, av_accuracy_trirel, ))

    
def load_data():
    print('loading data...')
    filename = os.path.join(data_dirs, 'sort-of-clevr++.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
      
    norel_train, birel_train, trirel_train = [],[],[]
    norel_test, birel_test, trirel_test = [],[],[]
    
    print('processing data...')

    for img, norelations, birelations, trirelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))
        for qst,ans in zip(birelations[0], birelations[1]):
            birel_train.append((img,qst,ans))
        for qst,ans in zip(trirelations[0], trirelations[1]):
            trirel_train.append((img,qst,ans))

    for img, norelations, birelations, trirelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
        for qst,ans in zip(birelations[0], birelations[1]):
            birel_test.append((img,qst,ans))
        for qst,ans in zip(trirelations[0], trirelations[1]):
            trirel_test.append((img,qst,ans))
    
    return (norel_train, norel_test, birel_train, birel_test, trirel_train, trirel_test)
    

norel_train, norel_test, birel_train, birel_test, trirel_train, trirel_test = load_data()

if args.resume>0:
    #print(args.template, type(args.model), type(args.resume), ) 
    filename = args.template.format(args.model, args.resume, )
    #print(filename)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(args.resume+1, args.resume+args.epochs + 1):
    train(epoch, norel_train, birel_train, trirel_train)
    test(epoch, norel_test, birel_test, trirel_test)
    model.save_model(args.template, epoch)
