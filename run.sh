#!/bin/bash

#python sort_of_clevr_generator.py

#python -u main.py --model=RN      --epochs=20  | tee  model/training_RN.log   
# Test set after epoch  1 : Relation accuracy: 67% | Non-relation accuracy: 61%
# Test set after epoch  5 : Relation accuracy: 71% | Non-relation accuracy: 61%
# Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 68%
# Test set after epoch 20 : Relation accuracy: 88% | Non-relation accuracy: 100%
# 252secs per epoch on 760 GTX


#python -u main.py --model=CNN_MLP --epochs=100 | tee  model/training_CNN_MLP.log
# Test set after epoch  1 : Relation accuracy: 53% | Non-relation accuracy: 59%
# Test set after epoch  5 : Relation accuracy: 72% | Non-relation accuracy: 59%
# Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 62%
# Test set after epoch 20 : Relation accuracy: 66% | Non-relation accuracy: 61%
# Test set after epoch 40 : Relation accuracy: 68% | Non-relation accuracy: 61%
# Test set after epoch 50 : Relation accuracy: 62% | Non-relation accuracy: 62%
# Test set after epoch 70 : Relation accuracy: 69% | Non-relation accuracy: 66%
# Test set after epoch 100 : Relation accuracy: 67% | Non-relation accuracy: 67%
#  43secs per epoch on 760 GTX


#python -u main.py --model=RFS     --epochs=20  | tee  model/training_RFS.log
# No softmax on output (Do'h)
# Epoch   0 (end) :   Test set: Relation accuracy: 35% | Non-relation accuracy: 17%
# Epoch  10 (end) :   Test set: Relation accuracy:  0% | Non-relation accuracy: 34%

# hidden_size=16
# Epoch 100 (end) :   Test set: Relation accuracy: 43% | Non-relation accuracy: 55%

# hidden_size=64
#  starts to plateau at ~8 epochs of training
# Epoch  10 (end) :   Test set: Relation accuracy: 45% | Non-relation accuracy: 52%
# Epoch  20 (end) :   Test set: Relation accuracy: 45% | Non-relation accuracy: 51%


# Test set after epoch  1 : Relation accuracy: 34% | Non-relation accuracy: 16%
# Test set after epoch  2 : Relation accuracy: 35% | Non-relation accuracy: 33%
# Test set after epoch  5 : Relation accuracy: 43% | Non-relation accuracy: 50%
# Test set after epoch 10 : Relation accuracy: 43% | Non-relation accuracy: 51%
# Test set after epoch 20 : Relation accuracy: 43% | Non-relation accuracy: 51%
#  76secs per epoch on Titax X (Maxwell)


# Additional rnn layer to create stream queries :  model/training_RFS-02-2rnn-layers-in-stream.log
#  Test set after epoch  1 : Relation accuracy: 33% | Non-relation accuracy: 16%
#  Test set after epoch  2 : Relation accuracy: 34% | Non-relation accuracy: 17%
#  Test set after epoch  5 : Relation accuracy: 43% | Non-relation accuracy: 51%
#  Test set after epoch 10 : Relation accuracy: 44% | Non-relation accuracy: 54%
#  Test set after epoch 15 : Relation accuracy: 44% | Non-relation accuracy: 55%
#  Test set after epoch 16 : Relation accuracy: 45% | Non-relation accuracy: 55%
#  Test set after epoch 17 : Relation accuracy: 46% | Non-relation accuracy: 55%
#  Test set after epoch 18 : Relation accuracy: 47% | Non-relation accuracy: 57%
#  Test set after epoch 19 : Relation accuracy: 49% | Non-relation accuracy: 56%
#  Test set after epoch 20 : Relation accuracy: 50% | Non-relation accuracy: 56%

# More :: python -u main.py --model=RFS     --epochs=20 --resume epoch_RFS_20.pth | tee --append model/training_RFS-02-2rnn-layers-in-stream.log

#  Test set after epoch 25 : Relation accuracy: 58% | Non-relation accuracy: 57%
#  Test set after epoch 30 : Relation accuracy: 63% | Non-relation accuracy: 58%
#  Test set after epoch 35 : Relation accuracy: 67% | Non-relation accuracy: 58%
#  Test set after epoch 40 : Relation accuracy: 68% | Non-relation accuracy: 57%

# More :: python -u main.py --model=RFS     --epochs=10 --resume epoch_RFS_20.pth | tee --append model/training_RFS-02-2rnn-layers-in-stream.log

#  Test set after epoch 45 : Relation accuracy: 70% | Non-relation accuracy: 59%
#  Test set after epoch 50 : Relation accuracy: 71% | Non-relation accuracy: 58%


# python -u main.py --model=RFS --epochs=20  | tee  model/training_RFS-key-is-param.log



# Run with higher learning rates initially:

python -u main.py --model=RFS --epochs=20 --lr=0.001  | tee model/training_RFS-key-is-param.log
#  Test set after epoch 20 : Relation accuracy: 41% | Non-relation accuracy: 81%

python -u main.py --model=RFS --epochs=30 --lr=0.0003 --resume epoch_RFS_20.pth | tee --append  model/training_RFS-key-is-param.log
#  Test set after epoch 50 : Relation accuracy: 80% | Non-relation accuracy: 98%

python -u main.py --model=RFS --epochs=150 --lr=0.0001 --resume epoch_RFS_30.pth | tee --append  model/training_RFS-key-is-param.log
# 220mins expected: ~<4hr


python -u main.py --model=RFS --epochs=50 --lr=0.001 --resume 0 --template model/{}_2item-span_{:03d}.pth | tee --append logs/RFS_2item-span.log
