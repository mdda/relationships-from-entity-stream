import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import Parameter


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

    def save_model(self, save_template, epoch):
        #torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))
        torch.save(self.state_dict(), save_template.format(self.name, epoch))


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


        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
            
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        
        coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            coord_tensor = coord_tensor.cuda()
        self.coord_tensor = Variable(coord_tensor)
        
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        # x_flat = (64 x 25 x 24)
        
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
    res = Variable(noise)
    if input.is_cuda:
      res = res.cuda()
    return res

def gumbel_softmax_sample(input, temperature=0.5):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    #x = F.log_softmax(x)
    x = F.softmax(x)
    return x.view_as(input)


class Harden(nn.Module):
    # https://discuss.pytorch.org/t/cannot-override-torch-round-after-upgrading-to-the-latest-pytorch-version/6396 ?
    def __init__(self, args):
        super(Harden, self).__init__()
        #self.y_onehot = torch.FloatTensor(args.batch_size, args.input_len)
        #self.batch_size = args.batch_size
        
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
    # https://discuss.pytorch.org/t/creating-one-hot-vector-from-indices-given-as-a-tensor/2171/3
    # https://github.com/mrdrozdov-github/pytorch-extras#one_hot
    def forward(self, vec):
        #self.y_onehot.zero_()
        #self.y_onehot.scatter_(1, vec, 1)      
        #return self.y_onehot
        
        values, indices = vec.max(1)

        y_onehot = torch.FloatTensor( vec.size() )
        if vec.is_cuda:
          y_onehot = y_onehot.cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, indices, 1)
        return y_onehot

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
        super(RFS, self).__init__(args, 'RFS')
        self.debug = args.debug
        
        self.conv = ConvInputModel()  
        # output is 24 channels in a 5x5 grid

        self.coord_extra_len = args.coord_extra_len

        # prepare coord tensor
        def cvt_coord(idx):
            i, j = idx/5, idx%5
            if self.coord_extra_len==2:
                return [(i-2)/2., (j-2)/2.]
            if self.coord_extra_len==6:
                return [
                  (i-2)/2., (i%2), (1. if (i>0) else 0.), 
                  (j-2)/2., (j%2), (1. if (j>0) else 0.), 
                ]
            
        np_coord_tensor = np.zeros((args.batch_size, 25, self.coord_extra_len))
        for idx in range(25):
            np_coord_tensor[:,idx,:] = np.array( cvt_coord(idx) )
        
        coord_tensor = torch.FloatTensor(args.batch_size, 25, self.coord_extra_len)
        if args.cuda:
            coord_tensor = coord_tensor.cuda()
        self.coord_tensor = Variable(coord_tensor)
        
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.question_size   = 11
        self.answer_size     = 10
        
        self.rnn_hidden_size = args.rnn_hidden_size  # must be > question_size and answer_size
        
        # 24+self.coord_extra_len+self.coord_extra_len = key_size + value_size
        if self.coord_extra_len==2:
            self.key_size = self.query_size   = 12
            self.value_size   = 16  
        else:
            self.key_size = self.query_size   = 20
            self.value_size   = 16

        self.process_coords = args.process_coords
        if self.process_coords:
            print("Create additional 1x1 convolutions to process coords additionally per point")
            self.coord_tensor_permuted = self.coord_tensor.permute(0,2,1)
            
            d_in, d_out = 24+self.coord_extra_len, self.key_size+self.value_size
            #print(d_in, d_out)
            if not (d_out == 24+self.coord_extra_len+self.coord_extra_len):
              print("Sizing of coordinate-enhanced 5x5 images does not match additional conv layers")
              exit(1)
            
            # These are 1d convs (since only 1x1 kernels anyway, and better shapes for below...)
            self.conv1 = nn.Conv1d(d_in, d_in, kernel_size=1, padding=0)
            self.batchNorm1 = nn.BatchNorm2d(d_in)  # d_hidden==d_in here
            self.conv2 = nn.Conv1d(d_in, d_out, kernel_size=1, padding=0)
            self.batchNorm2 = nn.BatchNorm2d(d_out)


        k_blank = torch.randn( (1, 1, self.key_size) )
        if args.cuda:
            k_blank = k_blank.cuda()
        self.k_blank = Parameter(k_blank, requires_grad=True)

        v_blank = torch.zeros( (1, 1, self.value_size) )
        if args.cuda:
            v_blank = v_blank.cuda()
        self.v_blank = Variable(v_blank, requires_grad=False)  # This is just fixed at ==0 == 'STOP'

        #seq_len=8
        #seq_len=4 
        #seq_len=2 # Works well enough to be on a par with RN
        #seq_len=1
        
        self.seq_len = args.seq_len
        

        ent_stream_rnn1_hidden_pad = torch.randn( (1, self.rnn_hidden_size-self.question_size) )
        if args.cuda:
            ent_stream_rnn1_hidden_pad = ent_stream_rnn1_hidden_pad.cuda()
        self.ent_stream_rnn1_hidden_pad = Parameter(ent_stream_rnn1_hidden_pad, requires_grad=True)
        #print("ent_stream_rnn1_hidden_pad.size() : ", self.ent_stream_rnn1_hidden_pad.size())  # (5)

        ent_stream_rnn1_start = torch.randn( (1, self.value_size) )  
        if args.cuda:
            ent_stream_rnn1_start = ent_stream_rnn1_start.cuda()
        self.ent_stream_rnn1_start = Parameter(ent_stream_rnn1_start, requires_grad=True)

        self.ent_stream_rnn1 = nn.GRUCell(self.value_size, self.rnn_hidden_size)   #input_size, hidden_size, bias=True)


        ent_stream_rnn2_hidden = torch.randn( (1, self.rnn_hidden_size) )
        if args.cuda:
            ent_stream_rnn2_hidden = ent_stream_rnn2_hidden.cuda()
        self.ent_stream_rnn2_hidden = Parameter(ent_stream_rnn2_hidden, requires_grad=True)

        self.ent_stream_rnn2 = nn.GRUCell(self.rnn_hidden_size, self.rnn_hidden_size)   #input_size, hidden_size, bias=True)
        
        self.stream_rnn_to_query = nn.Linear(self.rnn_hidden_size, self.query_size)

        # No parameters needed for softmax attention...  
        # Temperature for Gumbel?


        stream_question_hidden_pad = torch.randn( (1, self.rnn_hidden_size-self.question_size) )
        if args.cuda:
            stream_question_hidden_pad = stream_question_hidden_pad.cuda()
        self.stream_question_hidden_pad = Parameter(stream_question_hidden_pad, requires_grad=True)

        self.stream_question_rnn = nn.GRUCell(self.value_size, self.rnn_hidden_size)

        stream_answer_hidden   = torch.randn( (1, self.rnn_hidden_size) )
        if args.cuda:
            stream_answer_hidden = stream_answer_hidden.cuda()
        self.stream_answer_hidden = Parameter(stream_answer_hidden, requires_grad=True)

        self.stream_answer_rnn   = nn.GRUCell(self.rnn_hidden_size, self.rnn_hidden_size)

        self.stream_answer_to_output = nn.Linear(self.rnn_hidden_size, self.answer_size)
        
        #for param in self.parameters():
        #    print(type(param.data), param.size())        
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5) = (batch#, channels, x-s, y-s)
        
        """g"""
        batch_size = x.size()[0]  # minibatch
        n_channels = x.size()[1]  # output features of CNN  (24 normally or 28 if process_coords)
        d = x.size()[2]           # grid size over image

        if self.process_coords:
            # Add in the coordinates here...
            #print("process_coords : x_from-cnn.size(): ", x.size()) 
            
            x_flatter = x.view(batch_size, n_channels, d*d)
            #print("x_flatter.size(): ", x_flatter.size()) 
            
            #print("coord_tensor.size(): ", self.coord_tensor.size()) 
            #print("coord_tensor.permuted.size(): ", self.coord_tensor.permute(0,2,1).size()) 
            #print("coord_tensor_permuted.size(): ", self.coord_tensor_permuted.size()) 

            #x_plus = torch.cat([x_flatter, self.coord_tensor.permute(0,2,1) ], 1)
            x_plus = torch.cat([x_flatter, self.coord_tensor_permuted ], 1)
            #print("x_plus.size(): ", x_plus.size()) 
            
            x = self.conv1(x_plus)
            x = F.relu(x)
            x = self.batchNorm1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.batchNorm2(x)
            
            #print("x_after-1x1s.size(): ", x.size())   # 32,28,25
            
            x_flat = x.view(batch_size, self.key_size+self.value_size, d*d).permute(0,2,1)
            # x_flat = (64 x 25 x 28)
            
            ks_image = x_flat.narrow(2, 0, self.key_size)
            vs_image = x_flat.narrow(2, self.key_size, self.value_size)
          
        else:
            #print("Just concat coordinates : x_from-cnn.size(): ", x.size()) 
            x_flat = x.view(batch_size, n_channels, d*d).permute(0,2,1)
            # x_flat = (64 x 25 x 24)
            
            ks_nocoords = x_flat.narrow(2, 0, self.key_size-self.coord_extra_len)
            vs_nocoords = x_flat.narrow(2, self.key_size-self.coord_extra_len, self.value_size-self.coord_extra_len)
            
            # add coordinates (since these haven't been included yet)
            ks_image = torch.cat([ks_nocoords, self.coord_tensor], 2)
            vs_image = torch.cat([vs_nocoords, self.coord_tensor], 2)

            #print("ks_image.size() : ", ks_image.size())  # (32,25,12)
            #print("vs_image.size() : ", vs_image.size())  # (32,25,16)

        
        # add the 'end of choices' element
        #print("self.k_blank.size() : ", self.k_blank.size())  # (1,1,12)
        #print("self.k_blank.expand().size() : ", self.k_blank.expand( (batch_size, 1, self.key_size) ).size() )  # (32,1,12)
        ks = torch.cat([ks_image, self.k_blank.expand( (batch_size, 1, self.key_size) )], 1)
        #print("ks.size() : ", ks.size())  # (32,26,12)
        
        vs = torch.cat([vs_image, self.v_blank.expand( (batch_size, 1, self.value_size) )], 1)
        #print("vs.size() : ", vs.size())  # (32,26,16)
        
        #print("qst.size() : ", qst.size())  # (32,11)


        seq_len = self.seq_len

        ent_stream_rnn1_hidden = torch.cat( 
             [qst, self.ent_stream_rnn1_hidden_pad.expand( (batch_size, self.rnn_hidden_size-self.question_size) )], 1)
        #print("ent_stream_rnn_hidden.size() : ", ent_stream_rnn_hidden.size())  # (32,16)
        
        ent_stream_rnn1_input  = self.ent_stream_rnn1_start.expand(  (batch_size, self.value_size) )
        ent_stream_rnn2_hidden = self.ent_stream_rnn2_hidden.expand( (batch_size, self.rnn_hidden_size) )
        
        stream_logits, ent_similarities, stream_values = [],[],[] # Will be filled by RNN and attention process
        for i in range(seq_len):
          #print("ent_stream_rnn_input.size()  : ", ent_stream_rnn_input.size())   # (32,16)
          #print("ent_stream_rnn_hidden.size() : ", ent_stream_rnn_hidden.size())  # (32,16)
          ent_stream_rnn1_hidden = self.ent_stream_rnn1(ent_stream_rnn1_input, ent_stream_rnn1_hidden)

          ent_stream_rnn2_hidden = self.ent_stream_rnn2(ent_stream_rnn1_hidden, ent_stream_rnn2_hidden)

          ent_stream_logits = ent_stream_rnn2_hidden
          
          if self.debug:
            stream_logits.append( ent_stream_logits )
      
          # Convert the ent_stream hidden layer to a query
          qs = self.stream_rnn_to_query( ent_stream_logits )
          #print("qs.size() : ", qs.size())  # (32,12)

          #print("qs.unsqueeze(2).size() : ", torch.unsqueeze(qs, 2).size())  # (32,12,1)
                    
          # Now do the dot-product with the keys (flattened image-like)
          ent_similarity = torch.bmm( ks, torch.unsqueeze(qs, 2) )
          #print("ent_similarity.size() : ", ent_similarity.size())  # (32,26,1)

          if self.debug:
            ent_similarities.append( torch.squeeze( ent_similarity) )

          if True:
            # Softmax to get the weights
            #ent_weights = torch.nn.Softmax()( torch.squeeze( ent_similarity) )  #WORKED
            ent_weights = F.softmax( torch.squeeze( ent_similarity) )
            
          if False:
            # Gumbel-Softmax to get the weights:
            ent_weights = gumbel_softmax_sample( torch.squeeze( ent_similarity), temperature=0.2 )
            
          #print("ent_weights.size() : ", ent_weights.size())  # (32,26)
          #print("ent_weights.unsqueeze(2).size() : ", torch.unsqueeze(ent_weights,2).size())  # (32,26,1)  
          #print("ent_weights.unsqueeze(1).size() : ", torch.unsqueeze(ent_weights,1).size())  # (32,1,26)

          
          # Now multiply through to get the resulting values
          stream_next_value = torch.squeeze( torch.bmm( torch.unsqueeze(ent_weights,1), vs ) )
          #print("stream_next_value.size() : ", stream_next_value.size())  # (32, 16)
          
          stream_values.append(stream_next_value)
          ent_stream_rnn1_input = stream_next_value


        # Now interpret the values from the stream
        stream_question_hidden = torch.cat( 
                         [qst, self.stream_question_hidden_pad.expand( (batch_size, self.rnn_hidden_size-self.question_size) )], 1)
                         
        stream_answer_hidden   = self.stream_answer_hidden.expand( (batch_size, self.rnn_hidden_size) )
        #print("stream_answer_hidden0", stream_answer_hidden)
        
        for stream_question_rnn_input in stream_values:
          #print("stream_question_rnn_input.size() : ", stream_question_rnn_input.size())  # (32,16)
          #print("stream_question_hidden.size() : ", stream_question_hidden.size())  # (32,16)
          stream_question_hidden = self.stream_question_rnn(stream_question_rnn_input, stream_question_hidden)

          #print("stream_question_hidden.size() : ", stream_question_hidden.size())  # (32,16)
          #print("stream_answer_hidden.size() : ", stream_answer_hidden.size())  # (32,16)
          stream_answer_hidden   = self.stream_answer_rnn(stream_question_hidden, stream_answer_hidden)
          #print("stream_answer_hidden", stream_answer_hidden)
          
        # Final answer is in stream_answer_hidden (final value)
        #ans = stream_answer_hidden.narrow(1, 0, self.answer_size)  # No: Let's do a final linear on it...
        #print("ans.size() : ", ans.size())  # (32,10)

        ans = self.stream_answer_to_output( stream_answer_hidden )

        if self.debug:
          self.stream_logits = stream_logits
          self.ent_similarities = ent_similarities
          self.stream_values = stream_values
          self.ans_logits = ans
        
        return F.log_softmax(ans)  # log_softmax is what's expected


