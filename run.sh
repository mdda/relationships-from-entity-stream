#!/bin/bash

#python sort_of_clevr_generator.py

#python -u main.py --model=RN      --epochs=20  | tee  model/training_RN.log   
# Test set after epoch  1 : Relation accuracy: 67% | Non-relation accuracy: 61%
# Test set after epoch  5 : Relation accuracy: 71% | Non-relation accuracy: 61%
# Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 68%
# Test set after epoch 20 : Relation accuracy: 88% | Non-relation accuracy: 100%
# 252secs per epoch on 760 GTX


#python -u main.py --model=CNN_MLP --epochs=100 | tee  model/training_CNN_MLP.log
# Epoch 4 (end) :  Test set: Relation accuracy: 68% | Non-relation accuracy: 59%


#python -u main.py --model=RFS     --epochs=20  | tee  model/training_RFS.log
# Epoch   0 (end) :   Test set: Relation accuracy: 35% | Non-relation accuracy: 17%
# Epoch  10 (end) :   Test set: Relation accuracy:  0% | Non-relation accuracy: 34%

# hidden_size=16
# Epoch 100 (end) :   Test set: Relation accuracy: 43% | Non-relation accuracy: 55%

# hidden_size=64
#  starts to plateau at ~8 epochs of training
# Epoch  10 (end) :   Test set: Relation accuracy: 45% | Non-relation accuracy: 52%
# Epoch  20 (end) :   Test set: Relation accuracy: 45% | Non-relation accuracy: 51%




