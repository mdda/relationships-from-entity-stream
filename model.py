import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        # This is now 24 channels in a 5x5 grid


        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)

  

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        coord_oi = torch.FloatTensor(args.batch_size, 2)
        coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            coord_oi = coord_oi.cuda()
            coord_oj = coord_oj.cuda()
        self.coord_oi = Variable(coord_oi)
        self.coord_oj = Variable(coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,25,1)
        qst = torch.unsqueeze(qst, 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,63)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)



# https://www.reddit.com/r/MachineLearning/comments/6d44i7/d_how_to_use_gumbelsoftmax_for_policy_gradient/
# The gumbel-softmax is for a more specific case that being able to approximate a gradient 
# for any non-differentiable function. Softmax is exactly what is says on the tin; a soft-max. 
# The max function is not differentiable but is often used to sample from a distribution 
# by taking the highest probability. The softmax can be used to approximate the max function 
# and is differentiable. So what you can do is take the max in the forward pass but use softmax 
# during the backward pass in order to be able to pass gradients though it. 
# You can then anneal the softmax function temperature so that the approximation gets closer and closer 
# to the true max function during training to lower the error in the approximation.


# Blog post : 
#   http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
#   TF code : https://gist.github.com/ericjang/1001afd374c2c3b7752545ce6d9ed349

# Keras notebook version : https://github.com/EderSantana/gumbel

# Theano / Lasagne : https://github.com/yandexdataschool/gumbel_lstm
#   For the 'hard' version, plain argmax is used (at https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_softmax.py#L81) 


# afaik, unlike max, argmax (index of maximum) will have zero/NA gradient by definition 
#  since infinitely small changes in the vector won't change index of the maximum unless there are two exactly equal elements.


# From : https://github.com/pytorch/pytorch/issues/639
#def gumbel_sampler(input, tau, temperature):
#    noise = torch.rand(input.size())
#    noise.add_(1e-9).log_().neg_()
#    noise.add_(1e-9).log_().neg_()
#    noise = Variable(noise)
#    x = (input + noise) / tau + temperature
#    x = F.softmax(x.view(input.size(0), -1))
#    return x.view_as(input)

# From : https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530
def sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(input, temperature=1.):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.log_softmax(x)
    return x.view_as(input)


class Harden(nn.Module):
    # https://discuss.pytorch.org/t/cannot-override-torch-round-after-upgrading-to-the-latest-pytorch-version/6396 ?
    def __init__(self, args):
        super(Harden, self).__init__()
        self.y_onehot = torch.FloatTensor(args.batch_size, args.input_len)
        
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
    # https://discuss.pytorch.org/t/creating-one-hot-vector-from-indices-given-as-a-tensor/2171/3
    # https://github.com/mrdrozdov-github/pytorch-extras#one_hot
    def forward(self, vec):
        self.y_onehot.zero_()
        self.y_onehot.scatter_(1, vec, 1)      
        return 

    def backward(self, grads):
        return grads  # This is an identity pass-through


#  https://github.com/jcjohnson/pytorch-examples

#class Harden(torch.autograd.Function):
#  """
#  We can implement our own custom autograd Functions by subclassing
#  torch.autograd.Function and implementing the forward and backward passes
#  which operate on Tensors.
#  """
#  def forward(self, input):
#    """
#    In the forward pass we receive a Tensor containing the input and return a
#    Tensor containing the output. You can cache arbitrary Tensors for use in the
#    backward pass using the save_for_backward method.
#    """
#    self.save_for_backward(input)
#    return input.clamp(min=0)
#
#  def backward(self, grad_output):
#    """
#    In the backward pass we receive a Tensor containing the gradient of the loss
#    with respect to the output, and we need to compute the gradient of the loss
#    with respect to the input.
#    """
#    input, = self.saved_tensors
#    grad_input = grad_output.clone()
#    grad_input[input < 0] = 0
#    return grad_input

class RFS(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RFS')
        
        self.conv = ConvInputModel()  
        # output is 24 channels in a 5x5 grid
        
        coord_oi = torch.FloatTensor(args.batch_size, 2)
        coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            coord_oi = coord_oi.cuda()
            coord_oj = coord_oj.cuda()
        self.coord_oi = Variable(coord_oi)
        self.coord_oj = Variable(coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            coord_tensor = coord_tensor.cuda()
        self.coord_tensor = Variable(coord_tensor)
        
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.question_size   = 11
        self.answer_size     = 10
        self.rnn_hidden_size = 16 # > question_size and answer_size
        
        self.key_size = self.query_size   = 12
        self.value_size   = 16  # 24+2+2 = key_size + value_size

        ent_stream_rnn_start = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            ent_stream_rnn_start = ent_stream_rnn_start.cuda()
        self.ent_stream_rnn_start = Variable(ent_stream_rnn_start)
        
        self.ent_stream_rnn = nn.GRUCell(self.value_size, self.rnn_hidden_size)   #input_size, hidden_size, bias=True)
        
        self.stream_rnn_to_query = nn.Linear(self.rnn_hidden_size, self.query_size)

        # No parameters needed for softmax attention...  
        # Temperature for Gumbel?

        self.stream_question_rnn = nn.GRUCell(self.value_size, self.rnn_hidden_size)
        self.stream_answer_rnn   = nn.GRUCell(self.rnn_hidden_size, self.rnn_hidden_size)

        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5) = (batch#, channels, x-s, y-s)
        
        """g"""
        batch_size = x.size()[0]  # minibatch
        n_channels = x.size()[1]
        #d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(batch_size, n_channels, d*d).permute(0,2,1)
        
        # Split the x_flat into (keys) and (values)
        
        ks_nocoords = x_flat.narrow(3, 0, self.key_size-2)
        vs_nocoords = x_flat.narrow(3, self.key_size-2, self.value_size-2)
        
        # add coordinates
        ks = torch.cat([ks_nocoords, self.coord_tensor], 2)
        vs = torch.cat([vs_nocoords, self.coord_tensor], 2)
        
        seq_len=8
        
        ent_stream_rnn_hidden = F.pad(qst, (0, self.rnn_hidden_size - self.question_size), "constant", 0)
        ent_stream_rnn_input = self.ent_stream_rnn_start
        
        stream_values = [] # Will be filled by RNN and attention process
        for i in range(seq_len):
          ent_stream_rnn_hidden = self.ent_stream_rnn(ent_stream_rnn_input, ent_stream_rnn_hidden)
          
          # Convert the ent_stream hidden layer to a query
          qs = self.stream_rnn_to_query( ent_stream_rnn_hidden )
          
          # Now do the dot-product with the keys (flattened image-like)
          ent_similarity = torch.bmm( ks, torch.unsqueeze(qs, 2))
          
          # And softmax (etc) to get the weights
          ent_weights = torch.softmax( ent_similarity )
          
          # Now multiply through to get the resulting values
          stream_next_value = torch.bmm( ent_weights, vs )
          
          stream_values.append(stream_next_value)
          ent_stream_rnn_input = stream_next_value


        # Now interpret the values from the stream
        stream_question_hidden = F.pad(qst, (0, self.rnn_hidden_size - self.question_size), "constant", 0)
        stream_answer_hidden   = zeros.zeros(batch_size, self.rnn_hidden_size)
        
        for stream_question_rnn_input in stream_values:
          stream_question_hidden = self.stream_question_rnn(stream_question_rnn_input, stream_question_hidden)

          stream_answer_hidden   = self.stream_answer_rnn(stream_question_hidden, stream_answer_hidden)
          
        # Final answer is in stream_answer_hidden
        return stream_answer_hidden.narrow(1, 0, self.answer_size)


