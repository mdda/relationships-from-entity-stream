#!/bin/bash

# python sort_of_clevr_generator.py

#python -u main.py --model=RN --epochs=50  | tee --append model/training_RN_dataset++.log   
# Test set after epoch  1 : Relation accuracy: 67% | Non-relation accuracy: 61%
# Test set after epoch  5 : Relation accuracy: 71% | Non-relation accuracy: 61%
# Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 68%
# Test set after epoch 20 : Relation accuracy: 88% | Non-relation accuracy: 100%
# 252secs per epoch on 760 GTX


#python -u main.py --model=CNN_MLP --epochs=100 | tee --append model/training_CNN_MLP.log
# Test set after epoch  1 : Relation accuracy: 53% | Non-relation accuracy: 59%
# Test set after epoch  5 : Relation accuracy: 72% | Non-relation accuracy: 59%
# Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 62%
# Test set after epoch 20 : Relation accuracy: 66% | Non-relation accuracy: 61%
# Test set after epoch 40 : Relation accuracy: 68% | Non-relation accuracy: 61%
# Test set after epoch 50 : Relation accuracy: 62% | Non-relation accuracy: 62%
# Test set after epoch 70 : Relation accuracy: 69% | Non-relation accuracy: 66%
# Test set after epoch 100 : Relation accuracy: 67% | Non-relation accuracy: 67%
#  43secs per epoch on 760 GTX


#python -u main.py --model=RFES     --epochs=20  | tee --append model/training_RFES.log
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


# Additional rnn layer to create stream queries :  model/training_RFES-02-2rnn-layers-in-stream.log
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

# More :: python -u main.py --model=RFES     --epochs=20 --resume epoch_RFES_20.pth | tee --append model/training_RFES-02-2rnn-layers-in-stream.log

#  Test set after epoch 25 : Relation accuracy: 58% | Non-relation accuracy: 57%
#  Test set after epoch 30 : Relation accuracy: 63% | Non-relation accuracy: 58%
#  Test set after epoch 35 : Relation accuracy: 67% | Non-relation accuracy: 58%
#  Test set after epoch 40 : Relation accuracy: 68% | Non-relation accuracy: 57%

# More :: python -u main.py --model=RFES     --epochs=10 --resume epoch_RFES_20.pth | tee --append model/training_RFES-02-2rnn-layers-in-stream.log

#  Test set after epoch 45 : Relation accuracy: 70% | Non-relation accuracy: 59%
#  Test set after epoch 50 : Relation accuracy: 71% | Non-relation accuracy: 58%


# python -u main.py --model=RFES --epochs=20  | tee  model/training_RFES-key-is-param.log



# Run with higher learning rates initially:

#python -u main.py --model=RFES --epochs=20 --lr=0.001  | tee model/training_RFES-key-is-param.log
#  Test set after epoch 20 : Relation accuracy: 41% | Non-relation accuracy: 81%

#python -u main.py --model=RFES --epochs=30 --lr=0.0003 --resume epoch_RFES_20.pth | tee --append  model/training_RFES-key-is-param.log
#  Test set after epoch 50 : Relation accuracy: 80% | Non-relation accuracy: 98%

#python -u main.py --model=RFES --epochs=150 --lr=0.0001 --resume epoch_RFES_30.pth | tee --append  model/training_RFES-key-is-param.log
# 220mins expected: ~<4hr


#python -u main.py --model=RFES --epochs=50 --lr=0.001 --resume 0 --template model/{}_2item-span_{:03d}.pth | tee --append logs/RFES_2item-span.log
(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span.log 
#  Test set after epoch  1 : Relation accuracy: 54% | Non-relation accuracy: 57%
#  Test set after epoch  2 : Relation accuracy: 73% | Non-relation accuracy: 60%
#  Test set after epoch  5 : Relation accuracy: 73% | Non-relation accuracy: 63%
#  Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 61%
#  Test set after epoch 20 : Relation accuracy: 80% | Non-relation accuracy: 86%
#  Test set after epoch 30 : Relation accuracy: 93% | Non-relation accuracy: 99%  !!
#  Test set after epoch 40 : Relation accuracy: 94% | Non-relation accuracy: 99%
#  Test set after epoch 50 : Relation accuracy: 92% | Non-relation accuracy: 99%

python -u main.py --model=RFES --epochs=30 --lr=0.001 --resume 0 --template model/{}_1item-span_{:03d}.pth | tee --append logs/RFES_1item-span.log
# This should do well on the nonrel, but fail on the rel .. Hmmm : Not so much as expected...
# grep Test logs/RFES_1item-span.log 
#  Test set after epoch  1 : Relation accuracy: 45% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 44% | Non-relation accuracy: 56%
#  Test set after epoch  5 : Relation accuracy: 52% | Non-relation accuracy: 75%
#  Test set after epoch 10 : Relation accuracy: 50% | Non-relation accuracy: 77%
#  Test set after epoch 20 : Relation accuracy: 75% | Non-relation accuracy: 78%
#  Test set after epoch 30 : Relation accuracy: 78% | Non-relation accuracy: 78%


#python -u main.py --model=RFES --epochs=30 --lr=0.001 --resume 0 --template model/{}_1item-span-gumbel_{:03d}.pth | tee --append logs/RFES_1item-span-gumbel.log
# Just see what Gumbel does in the place of softmax
# (env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_1item-span-gumbel.log 
#  Test set after epoch  1 : Relation accuracy: 48% | Non-relation accuracy: 52%
#  Test set after epoch  2 : Relation accuracy: 58% | Non-relation accuracy: 54%
#  Test set after epoch  5 : Relation accuracy: 67% | Non-relation accuracy: 58%
#  Test set after epoch 10 : Relation accuracy: 69% | Non-relation accuracy: 60%
#  Test set after epoch 20 : Relation accuracy: 71% | Non-relation accuracy: 60%
#  Test set after epoch 30 : Relation accuracy: 70% | Non-relation accuracy: 59%

(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-gumbel.log 
#  Test set after epoch  1 : Relation accuracy: 48% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 55% | Non-relation accuracy: 56%
#  Test set after epoch  5 : Relation accuracy: 70% | Non-relation accuracy: 58%
#  Test set after epoch 10 : Relation accuracy: 71% | Non-relation accuracy: 60%
#  Test set after epoch 20 : Relation accuracy: 47% | Non-relation accuracy: 57%
#  Test set after epoch 30 : Relation accuracy: 71% | Non-relation accuracy: 60%

#  Fix with log_softmax() -> softmax()    # Actually seems worse...
#  Test set after epoch  1 : Relation accuracy: 46% | Non-relation accuracy: 50%
#  Test set after epoch  2 : Relation accuracy: 50% | Non-relation accuracy: 54%
#  Test set after epoch  5 : Relation accuracy: 56% | Non-relation accuracy: 56%
#  Test set after epoch 10 : Relation accuracy: 55% | Non-relation accuracy: 56%
#  Test set after epoch 20 : Relation accuracy: 59% | Non-relation accuracy: 56%
#  Test set after epoch 30 : Relation accuracy: 59% | Non-relation accuracy: 56%


#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --template model/{}_2item-span-seed10_{:03d}.pth | tee --append logs/RFES_2item-span-seed10.log
# hidden_size=32 (as before)
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-seed10.log   # WORKS (better) with different seed
#  Test set after epoch  1 : Relation accuracy: 48% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 55% | Non-relation accuracy: 58%
#  Test set after epoch  5 : Relation accuracy: 73% | Non-relation accuracy: 86%
#  Test set after epoch 10 : Relation accuracy: 84% | Non-relation accuracy: 99%   
#  Test set after epoch 20 : Relation accuracy: 93% | Non-relation accuracy: 100%   !!
#  Test set after epoch 30 : Relation accuracy: 92% | Non-relation accuracy: 100%
#  Test set after epoch 40 : Relation accuracy: 93% | Non-relation accuracy: 100%
#  Test set after epoch 50 : Relation accuracy: 93% | Non-relation accuracy: 100%

#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=64 --resume 0 --seed 10 --template model/{}_2item-span-hidden64-seed10_{:03d}.pth | tee --append logs/RFES_2item-span-hidden64-seed10.log
# hidden_size=64
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-hidden64-seed10.log 
#  Test set after epoch  1 : Relation accuracy: 45% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 43% | Non-relation accuracy: 52%
#  Test set after epoch  5 : Relation accuracy: 67% | Non-relation accuracy: 60%
#  Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 61%
#  Test set after epoch 20 : Relation accuracy: 74% | Non-relation accuracy: 61%
#  Test set after epoch 30 : Relation accuracy: 74% | Non-relation accuracy: 61%
#  Test set after epoch 40 : Relation accuracy: 82% | Non-relation accuracy: 98%
#  Test set after epoch 50 : Relation accuracy: 90% | Non-relation accuracy: 99%    !!


#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --process_coords --seed 10 --template model/{}_2item-span-process_coords-seed10_{:03d}.pth | tee --append logs/RFES_2item-span-process_coords.log
# hidden_size=32.  But add coords to CNN output and do 2 further layers of 1x1 covolutions
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-process_coords.log
#  Test set after epoch  1 : Relation accuracy: 49% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 70% | Non-relation accuracy: 56%
#  Test set after epoch  5 : Relation accuracy: 72% | Non-relation accuracy: 63%
#  Test set after epoch 10 : Relation accuracy: 12% | Non-relation accuracy: 48%
#  Test set after epoch 11 : Relation accuracy: 57% | Non-relation accuracy: 66%
#  Test set after epoch 15 : Relation accuracy: 75% | Non-relation accuracy: 68%
#  Test set after epoch 17 : Relation accuracy: 75% | Non-relation accuracy: 74%
#  Test set after epoch 18 : Relation accuracy: 79% | Non-relation accuracy: 82%
#  Test set after epoch 19 : Relation accuracy: 80% | Non-relation accuracy: 96%
#  Test set after epoch 20 : Relation accuracy: 82% | Non-relation accuracy: 97%
#  Test set after epoch 30 : Relation accuracy: 94% | Non-relation accuracy: 99%
#  Test set after epoch 40 : Relation accuracy: 94% | Non-relation accuracy: 99%
#  Test set after epoch 50 : Relation accuracy: 95% | Non-relation accuracy: 99%


#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --template model/{}_2item-span-again-seed10_{:03d}.pth | tee --append logs/RFES_2item-span-again.log
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-again.log
#  Test set after epoch  1 : Relation accuracy: 43% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 43% | Non-relation accuracy: 52%
#  Test set after epoch  3 : Relation accuracy: 44% | Non-relation accuracy: 52%
#  Test set after epoch  4 : Relation accuracy: 44% | Non-relation accuracy: 68%
#  Test set after epoch  5 : Relation accuracy: 42% | Non-relation accuracy: 90%
#  Test set after epoch  6 : Relation accuracy: 51% | Non-relation accuracy: 98%
#  Test set after epoch  7 : Relation accuracy: 55% | Non-relation accuracy: 99%
#  Test set after epoch  8 : Relation accuracy: 61% | Non-relation accuracy: 99%
#  Test set after epoch  9 : Relation accuracy: 73% | Non-relation accuracy: 99%
#  Test set after epoch 10 : Relation accuracy: 79% | Non-relation accuracy: 99%
#  Test set after epoch 20 : Relation accuracy: 85% | Non-relation accuracy: 99%
#  Test set after epoch 30 : Relation accuracy: 93% | Non-relation accuracy: 99%
#  Test set after epoch 40 : Relation accuracy: 93% | Non-relation accuracy: 100%
#  Test set after epoch 50 : Relation accuracy: 94% | Non-relation accuracy: 99%


#  -- added 'tricky' relationships to dataset...
# python -u main.py --model=RN --epochs=50 --template model/{}_{:03d}.pth | tee --append logs/training_RN_dataset++.log   
# python -u main.py --model=RN --epochs=50 --template model/{}_tricky_{:03d}.pth --train_tricky | tee --append logs/training_RN_dataset++_tricky.log   
#   Unfortunately, the standard RN model can cope with that too - though they are learned later than the birels (as to be expected, probably)



#  -- updated to three more subtle 'tricky' relationships to dataset...
# python -u main.py --model=RN --epochs=50 --template model/{}_tricky_{:03d}.pth --train_tricky | tee --append logs/training_RN_dataset++_tricky.log   

# python -u main.py --model=CNN_MLP --epochs=50 --template model/{}_tricky_{:03d}.pth --train_tricky | tee --append model/training_CNN_dataset++_tricky.log   
#  996  cp /mnt/data/Personal/logs.tmp/CNNMLP_tricky_050.pth model/
#  999  cp /mnt/data/Personal/logs.tmp/training_CNN_dataset++_tricky.log  logs/

# python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFES_2item-span-tricky.log
# python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 40 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFES_2item-span-tricky.log

# python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --seq_len 3 --template model/{}_3item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFES_3item-span-tricky.log
#  997  cp /mnt/data/Personal/logs.tmp/RFES_3item-span-seed10-tricky_050.pth model/
#  998  cp /mnt/data/Personal/logs.tmp/RFES_3item-span-tricky.log  logs/



# python -u main.py --model=RN --epochs=50 --template model/{}_tricky2_{:03d}.pth --train_tricky | tee --append logs/training_RN_dataset++_tricky2.log   
  
# python -u main.py --model=RFES --epochs=100 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky2_{:03d}.pth | tee --append logs/RFES_2item-span-tricky2.log
# python -u main.py --model=RFES --epochs=100 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 1 --train_tricky --template model/{}_2item-span-seed1-tricky2_{:03d}.pth | tee --append logs/RFES_2item-span-seed1-tricky2.log
## Peformance on norel and birel pretty much the same.  trirel a little worse (50s vs 60) - but should be winning...


# python -u main.py --model=RFES --epochs=100 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6  --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-6coord-tricky2_{:03d}.pth | tee --append logs/RFES_2item-span-6coord-tricky2.log
# python -u main.py --model=RFES --epochs=50 --lr=0.0003 --rnn_hidden_size=32 --coord_extra_len=6 --resume 50 --seed 10 --train_tricky --template model/{}_2item-span-seed10-6coord-tricky2_{:03d}.pth | tee --append logs/RFES_2item-span-6coord-tricky2.log

# python -u main.py --model=RN  --epochs=50 --seed 10 --template model/{}_{:03d}.pth | tee --append logs/training_RN_seed10.log   

(env3) [andrewsm@simlim relationships-from-entity-stream]$ grep Test logs/RFES_2item-span-6coord.log 
  Test set after epoch  1 : Non-relation accuracy: 54% | Relation accuracy: 48% | Tricky accuracy: 0% 
  Test set after epoch  2 : Non-relation accuracy: 63% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch  3 : Non-relation accuracy: 64% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch  4 : Non-relation accuracy: 65% | Relation accuracy: 73% | Tricky accuracy: 0% 
  Test set after epoch  5 : Non-relation accuracy: 63% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch  6 : Non-relation accuracy: 64% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch  7 : Non-relation accuracy: 64% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch  8 : Non-relation accuracy: 64% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch  9 : Non-relation accuracy: 65% | Relation accuracy: 73% | Tricky accuracy: 0% 
  Test set after epoch 10 : Non-relation accuracy: 68% | Relation accuracy: 73% | Tricky accuracy: 0% 
  Test set after epoch 11 : Non-relation accuracy: 85% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 12 : Non-relation accuracy: 94% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 13 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 14 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 15 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 16 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 17 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 18 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 19 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 20 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 21 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 22 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 23 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 24 : Non-relation accuracy: 100% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 25 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 26 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 27 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 28 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 29 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 30 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 31 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 32 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 33 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 34 : Non-relation accuracy: 99% | Relation accuracy: 96% | Tricky accuracy: 0% 
  Test set after epoch 35 : Non-relation accuracy: 100% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 36 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 37 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 38 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 39 : Non-relation accuracy: 99% | Relation accuracy: 96% | Tricky accuracy: 0% 
**  Test set after epoch 40 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 41 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 42 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 43 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 44 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 45 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 46 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
  Test set after epoch 47 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 48 : Non-relation accuracy: 99% | Relation accuracy: 96% | Tricky accuracy: 0% 
  Test set after epoch 49 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 50 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 


# python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_2item-span-seed10-6coord_{:03d}.pth | tee --append logs/RFES_2item-span-6coord.log
# grep Test logs/RFES_2item-span-6coord.log 

# For comparison (back to non-tricky stuff to see whether 99/95 beats it)
# python -u main.py --model=RN  --epochs=50 --seed 10 --template model/{}_{:03d}.pth | tee --append logs/training_RN_seed10.log 
# grep Test logs/training_RN_seed10.log 
  Test set after epoch  1 : Non-relation accuracy: 61% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch  2 : Non-relation accuracy: 60% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch  3 : Non-relation accuracy: 61% | Relation accuracy: 53% | Tricky accuracy: 0% 
  Test set after epoch  4 : Non-relation accuracy: 63% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch  5 : Non-relation accuracy: 63% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch  6 : Non-relation accuracy: 63% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch  7 : Non-relation accuracy: 71% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch  8 : Non-relation accuracy: 81% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch  9 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 10 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 11 : Non-relation accuracy: 99% | Relation accuracy: 82% | Tricky accuracy: 0% 
  Test set after epoch 12 : Non-relation accuracy: 99% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 13 : Non-relation accuracy: 99% | Relation accuracy: 83% | Tricky accuracy: 0% 
  Test set after epoch 14 : Non-relation accuracy: 99% | Relation accuracy: 82% | Tricky accuracy: 0% 
  Test set after epoch 15 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 16 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 17 : Non-relation accuracy: 100% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 18 : Non-relation accuracy: 100% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 19 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 20 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 21 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 22 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 23 : Non-relation accuracy: 100% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 24 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 25 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 26 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 27 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 28 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 29 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 30 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 31 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 32 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 33 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 34 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 35 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 36 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 37 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 38 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 39 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 40 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 41 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 42 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 43 : Non-relation accuracy: 99% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 44 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 45 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 46 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 47 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 48 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 49 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 50 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 


#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_2item-span-seed10-6coord-relu_{:03d}.pth | tee --append logs/RFES_2item-span-6coord-relu.log
# Seems to work slightly better

#python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --highway 1 --template model/{}_2item-span-seed10-6coord-highway1_{:03d}.pth | tee --append logs/RFES_2item-span-6coord-highway1.log
# Works, but not clear that much is gained by adding highway network

python -u main.py --model=RFES --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --highway 1 --train_tricky --template model/{}_2item-span-seed10-6coord-highway1-tricky2_{:03d}.pth | tee --append logs/RFES_2item-span-6coord-highway1-tricky2.log
# Again, seems to get stuck like when ReLU was before the highway unit : Perhaps it's a seed (i.e. robustness) problem.  :: NOT CONVINCING


python -u main.py --model=RFESH --epochs=1 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_2item-span-seed10-6coord-fuzz1.0_{:03d}.pth | tee --append logs/RFESH_2item-span-6coord-fuzz1.0.log

python -u main.py --model=RFESH --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_2item-span-seed10-6coord-fuzz1.0-emph2.0_{:03d}.pth | tee --append logs/RFESH_2item-span-6coord-fuzz1.0-emph2.0.log

python -u main.py --model=RFESH --epochs=50 --lr=0.001 --seq_len=4 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_4item-span-seed10-6coord-fuzz1.0-emph5.0_{:03d}.pth | tee --append logs/RFESH_4item-span-6coord-fuzz1.0-emph5.0.log


python -u main.py --model=RFESH --epochs=150 --lr=0.001 --seq_len=6 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_6item-span-seed10-6coord-fuzz1.0-plain-hardtest_{:03d}.pth | tee --append logs/RFESH_6item-span-6coord-fuzz1.0-plain-hardtest.log
(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFESH_6item-span-6coord-fuzz1.0-plain-hardtest.log
  Test set after epoch  1 : Non-relation accuracy: 51% | Relation accuracy: 41% | Tricky accuracy: 0% 
  Test set after epoch  2 : Non-relation accuracy: 51% | Relation accuracy: 48% | Tricky accuracy: 0% 
  Test set after epoch  3 : Non-relation accuracy: 54% | Relation accuracy: 58% | Tricky accuracy: 0% 
  Test set after epoch  4 : Non-relation accuracy: 57% | Relation accuracy: 55% | Tricky accuracy: 0% 
  Test set after epoch  5 : Non-relation accuracy: 57% | Relation accuracy: 59% | Tricky accuracy: 0% 
  Test set after epoch  6 : Non-relation accuracy: 59% | Relation accuracy: 60% | Tricky accuracy: 0% 
  Test set after epoch  7 : Non-relation accuracy: 60% | Relation accuracy: 61% | Tricky accuracy: 0% 
  Test set after epoch  8 : Non-relation accuracy: 61% | Relation accuracy: 60% | Tricky accuracy: 0% 
  Test set after epoch  9 : Non-relation accuracy: 61% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 10 : Non-relation accuracy: 60% | Relation accuracy: 63% | Tricky accuracy: 0% 
  Test set after epoch 11 : Non-relation accuracy: 61% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 12 : Non-relation accuracy: 62% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 13 : Non-relation accuracy: 63% | Relation accuracy: 65% | Tricky accuracy: 0% 
  Test set after epoch 14 : Non-relation accuracy: 61% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 15 : Non-relation accuracy: 61% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 16 : Non-relation accuracy: 63% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 17 : Non-relation accuracy: 62% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 18 : Non-relation accuracy: 60% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 19 : Non-relation accuracy: 60% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 20 : Non-relation accuracy: 63% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 21 : Non-relation accuracy: 60% | Relation accuracy: 65% | Tricky accuracy: 0% 
  Test set after epoch 22 : Non-relation accuracy: 62% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 23 : Non-relation accuracy: 61% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 24 : Non-relation accuracy: 62% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 25 : Non-relation accuracy: 61% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 26 : Non-relation accuracy: 62% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 27 : Non-relation accuracy: 61% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 28 : Non-relation accuracy: 61% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 29 : Non-relation accuracy: 63% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 30 : Non-relation accuracy: 61% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 31 : Non-relation accuracy: 62% | Relation accuracy: 60% | Tricky accuracy: 0% 
  Test set after epoch 32 : Non-relation accuracy: 62% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 33 : Non-relation accuracy: 63% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 34 : Non-relation accuracy: 61% | Relation accuracy: 63% | Tricky accuracy: 0% 
  Test set after epoch 35 : Non-relation accuracy: 61% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 36 : Non-relation accuracy: 62% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 37 : Non-relation accuracy: 62% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 38 : Non-relation accuracy: 61% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 39 : Non-relation accuracy: 63% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 40 : Non-relation accuracy: 68% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 41 : Non-relation accuracy: 68% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 42 : Non-relation accuracy: 70% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 43 : Non-relation accuracy: 67% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 44 : Non-relation accuracy: 69% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 45 : Non-relation accuracy: 72% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 46 : Non-relation accuracy: 73% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 47 : Non-relation accuracy: 75% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 48 : Non-relation accuracy: 79% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 49 : Non-relation accuracy: 82% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 50 : Non-relation accuracy: 89% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 51 : Non-relation accuracy: 90% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 52 : Non-relation accuracy: 92% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 53 : Non-relation accuracy: 96% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 54 : Non-relation accuracy: 97% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 55 : Non-relation accuracy: 96% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 56 : Non-relation accuracy: 97% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 57 : Non-relation accuracy: 97% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 58 : Non-relation accuracy: 98% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 59 : Non-relation accuracy: 97% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 60 : Non-relation accuracy: 97% | Relation accuracy: 73% | Tricky accuracy: 0% 
  Test set after epoch 61 : Non-relation accuracy: 98% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 62 : Non-relation accuracy: 97% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 63 : Non-relation accuracy: 97% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 64 : Non-relation accuracy: 98% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 65 : Non-relation accuracy: 98% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 66 : Non-relation accuracy: 97% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 67 : Non-relation accuracy: 97% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 68 : Non-relation accuracy: 98% | Relation accuracy: 71% | Tricky accuracy: 0% 
  Test set after epoch 69 : Non-relation accuracy: 98% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 70 : Non-relation accuracy: 98% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 71 : Non-relation accuracy: 98% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 72 : Non-relation accuracy: 98% | Relation accuracy: 74% | Tricky accuracy: 0% 
  Test set after epoch 73 : Non-relation accuracy: 97% | Relation accuracy: 70% | Tricky accuracy: 0% 
  Test set after epoch 74 : Non-relation accuracy: 98% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 75 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 76 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 77 : Non-relation accuracy: 97% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 78 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 79 : Non-relation accuracy: 99% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 80 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 81 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 82 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 83 : Non-relation accuracy: 99% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 84 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 85 : Non-relation accuracy: 98% | Relation accuracy: 76% | Tricky accuracy: 0% 
  Test set after epoch 86 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 87 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 88 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 89 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 90 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 91 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 92 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 93 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 94 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 95 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 96 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 97 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 98 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 99 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 100 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 101 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 102 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 103 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 104 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 105 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 106 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 107 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 108 : Non-relation accuracy: 99% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 109 : Non-relation accuracy: 98% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 110 : Non-relation accuracy: 99% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 111 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 112 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 113 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 114 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 115 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 116 : Non-relation accuracy: 99% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 117 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 118 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 119 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 120 : Non-relation accuracy: 98% | Relation accuracy: 76% | Tricky accuracy: 0% 
  Test set after epoch 121 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 122 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 123 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 124 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 125 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 126 : Non-relation accuracy: 99% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 127 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 128 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 129 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 130 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 131 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 132 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 133 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 134 : Non-relation accuracy: 98% | Relation accuracy: 77% | Tricky accuracy: 0% 
  Test set after epoch 135 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 136 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 137 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 138 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 139 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 140 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 141 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 142 : Non-relation accuracy: 98% | Relation accuracy: 75% | Tricky accuracy: 0% 
  Test set after epoch 143 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 144 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 145 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 146 : Non-relation accuracy: 98% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 147 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 148 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 149 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 150 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 

python -u main.py --model=RFESH --epochs=500 --lr=0.001 --seq_len=6 --rnn_hidden_size=64 --coord_extra_len=6 --resume 0 --seed 11 --template model/{}_6item-span-seed11-hidden64-6coord-fuzz1.0-plain-hardtest_{:03d}.pth | tee --append logs/RFESH_6item-span-seed11-hidden64-6coord-fuzz1.0-plain-hardtest.log
grep Test logs/RFESH_6item-span-seed11-hidden64-6coord-fuzz1.0-plain-hardtest.log

  Test set after epoch  1 : Non-relation accuracy: 54% | Relation accuracy: 50% | Tricky accuracy: 0% 
  Test set after epoch  2 : Non-relation accuracy: 53% | Relation accuracy: 51% | Tricky accuracy: 0% 
  Test set after epoch  3 : Non-relation accuracy: 54% | Relation accuracy: 54% | Tricky accuracy: 0% 
  Test set after epoch  4 : Non-relation accuracy: 54% | Relation accuracy: 57% | Tricky accuracy: 0% 
  Test set after epoch  5 : Non-relation accuracy: 54% | Relation accuracy: 58% | Tricky accuracy: 0% 
  Test set after epoch  6 : Non-relation accuracy: 57% | Relation accuracy: 60% | Tricky accuracy: 0% 
  Test set after epoch  7 : Non-relation accuracy: 59% | Relation accuracy: 59% | Tricky accuracy: 0% 
  Test set after epoch  8 : Non-relation accuracy: 58% | Relation accuracy: 57% | Tricky accuracy: 0% 
  Test set after epoch  9 : Non-relation accuracy: 59% | Relation accuracy: 58% | Tricky accuracy: 0% 
  Test set after epoch 10 : Non-relation accuracy: 61% | Relation accuracy: 58% | Tricky accuracy: 0% 
  Test set after epoch 11 : Non-relation accuracy: 60% | Relation accuracy: 61% | Tricky accuracy: 0% 
  Test set after epoch 12 : Non-relation accuracy: 58% | Relation accuracy: 59% | Tricky accuracy: 0% 
  Test set after epoch 13 : Non-relation accuracy: 61% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 14 : Non-relation accuracy: 62% | Relation accuracy: 61% | Tricky accuracy: 0% 
  Test set after epoch 15 : Non-relation accuracy: 61% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 16 : Non-relation accuracy: 61% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 17 : Non-relation accuracy: 60% | Relation accuracy: 60% | Tricky accuracy: 0% 
  Test set after epoch 18 : Non-relation accuracy: 62% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 19 : Non-relation accuracy: 61% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 20 : Non-relation accuracy: 62% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 21 : Non-relation accuracy: 63% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 22 : Non-relation accuracy: 63% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 23 : Non-relation accuracy: 62% | Relation accuracy: 62% | Tricky accuracy: 0% 
  Test set after epoch 24 : Non-relation accuracy: 63% | Relation accuracy: 64% | Tricky accuracy: 0% 
  Test set after epoch 25 : Non-relation accuracy: 64% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 26 : Non-relation accuracy: 63% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 27 : Non-relation accuracy: 62% | Relation accuracy: 69% | Tricky accuracy: 0% 
  Test set after epoch 28 : Non-relation accuracy: 62% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 29 : Non-relation accuracy: 63% | Relation accuracy: 65% | Tricky accuracy: 0% 
  Test set after epoch 30 : Non-relation accuracy: 62% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 31 : Non-relation accuracy: 63% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 32 : Non-relation accuracy: 62% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 33 : Non-relation accuracy: 62% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 34 : Non-relation accuracy: 62% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 35 : Non-relation accuracy: 63% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 36 : Non-relation accuracy: 63% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 37 : Non-relation accuracy: 66% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 38 : Non-relation accuracy: 71% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 39 : Non-relation accuracy: 77% | Relation accuracy: 65% | Tricky accuracy: 0% 
  Test set after epoch 40 : Non-relation accuracy: 77% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 41 : Non-relation accuracy: 80% | Relation accuracy: 66% | Tricky accuracy: 0% 
  Test set after epoch 42 : Non-relation accuracy: 83% | Relation accuracy: 67% | Tricky accuracy: 0% 
  Test set after epoch 43 : Non-relation accuracy: 85% | Relation accuracy: 68% | Tricky accuracy: 0% 
  Test set after epoch 44 : Non-relation accuracy: 85% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 45 : Non-relation accuracy: 87% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 46 : Non-relation accuracy: 94% | Relation accuracy: 72% | Tricky accuracy: 0% 
  Test set after epoch 47 : Non-relation accuracy: 97% | Relation accuracy: 73% | Tricky accuracy: 0% 
  Test set after epoch 48 : Non-relation accuracy: 98% | Relation accuracy: 78% | Tricky accuracy: 0% 
  Test set after epoch 49 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 50 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 51 : Non-relation accuracy: 97% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 52 : Non-relation accuracy: 98% | Relation accuracy: 82% | Tricky accuracy: 0% 
  Test set after epoch 53 : Non-relation accuracy: 98% | Relation accuracy: 83% | Tricky accuracy: 0% 
  Test set after epoch 54 : Non-relation accuracy: 98% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 55 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 56 : Non-relation accuracy: 98% | Relation accuracy: 83% | Tricky accuracy: 0% 
  Test set after epoch 57 : Non-relation accuracy: 98% | Relation accuracy: 81% | Tricky accuracy: 0% 
  Test set after epoch 58 : Non-relation accuracy: 98% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 59 : Non-relation accuracy: 98% | Relation accuracy: 83% | Tricky accuracy: 0% 
  Test set after epoch 60 : Non-relation accuracy: 98% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 61 : Non-relation accuracy: 98% | Relation accuracy: 83% | Tricky accuracy: 0% 
  Test set after epoch 62 : Non-relation accuracy: 98% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 63 : Non-relation accuracy: 98% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 64 : Non-relation accuracy: 98% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 65 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 66 : Non-relation accuracy: 99% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 67 : Non-relation accuracy: 98% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 68 : Non-relation accuracy: 98% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 69 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 70 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 71 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 72 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 73 : Non-relation accuracy: 98% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 74 : Non-relation accuracy: 98% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 75 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 76 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 77 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 78 : Non-relation accuracy: 99% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 79 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 80 : Non-relation accuracy: 99% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 81 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 82 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 83 : Non-relation accuracy: 99% | Relation accuracy: 85% | Tricky accuracy: 0% 
  Test set after epoch 84 : Non-relation accuracy: 98% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 85 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 86 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 87 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 88 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 89 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 90 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 91 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 92 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 93 : Non-relation accuracy: 98% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 94 : Non-relation accuracy: 98% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 95 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 96 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 97 : Non-relation accuracy: 98% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 98 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 99 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 100 : Non-relation accuracy: 99% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 101 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 102 : Non-relation accuracy: 98% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 103 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 104 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 105 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 106 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 107 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 108 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 109 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 110 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 111 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 112 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 113 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 114 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 115 : Non-relation accuracy: 98% | Relation accuracy: 84% | Tricky accuracy: 0% 
  Test set after epoch 116 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 117 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 118 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 119 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 120 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 121 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 122 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 123 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 124 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 125 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 126 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 127 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 128 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 129 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 130 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 131 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 132 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 133 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 134 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 135 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 136 : Non-relation accuracy: 99% | Relation accuracy: 86% | Tricky accuracy: 0% 
  Test set after epoch 137 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 138 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 139 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 140 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 141 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 142 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 143 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 144 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 145 : Non-relation accuracy: 99% | Relation accuracy: 79% | Tricky accuracy: 0% 
  Test set after epoch 146 : Non-relation accuracy: 99% | Relation accuracy: 80% | Tricky accuracy: 0% 
  Test set after epoch 147 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 148 : Non-relation accuracy: 98% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 149 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 150 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 151 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 152 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 153 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 154 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 155 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 156 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 157 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 158 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 159 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 160 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 161 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 162 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 163 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 164 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 165 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 166 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 167 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 168 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 169 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 170 : Non-relation accuracy: 99% | Relation accuracy: 87% | Tricky accuracy: 0% 
  Test set after epoch 171 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 172 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 173 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 174 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 175 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 176 : Non-relation accuracy: 98% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 177 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 178 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 179 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 180 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 181 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 182 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 183 : Non-relation accuracy: 98% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 184 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 185 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 186 : Non-relation accuracy: 99% | Relation accuracy: 88% | Tricky accuracy: 0% 
  Test set after epoch 187 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 188 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 189 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 190 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 191 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 192 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 193 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 194 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 195 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 196 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 197 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 198 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 199 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 200 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 201 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 202 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 203 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 204 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 205 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 206 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 207 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 208 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 209 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 210 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 211 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 212 : Non-relation accuracy: 98% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 213 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 214 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 215 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 216 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 217 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 218 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 219 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 220 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 221 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 222 : Non-relation accuracy: 98% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 223 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 224 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 225 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 226 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 227 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 228 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 229 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 230 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 231 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 232 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 233 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 234 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 235 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 236 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 237 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 238 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 239 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 240 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 241 : Non-relation accuracy: 99% | Relation accuracy: 90% | Tricky accuracy: 0% 
  Test set after epoch 242 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 243 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 244 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 245 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 246 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 247 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 248 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 249 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 250 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 251 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 252 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 253 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 254 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 255 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 256 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 257 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 258 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 259 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 260 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 261 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 262 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 263 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 264 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 265 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 266 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 267 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 268 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 269 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 270 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 271 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 272 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 273 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 274 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 275 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 276 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 277 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 278 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 279 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 280 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 281 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 282 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 283 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 284 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 285 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 286 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 287 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 288 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 289 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 290 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 291 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 292 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 293 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 294 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 295 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 296 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 297 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 298 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 299 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 300 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 301 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 302 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 303 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 304 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 305 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 306 : Non-relation accuracy: 98% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 307 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 308 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 309 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 310 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 311 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 312 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 313 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 314 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 315 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 316 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 317 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 318 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 319 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 320 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 321 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 322 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 323 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 324 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 325 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 326 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 327 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 328 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 329 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 330 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 331 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 332 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 333 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 334 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 335 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 336 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 337 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 338 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 339 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 340 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 341 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 342 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 343 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 344 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 345 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 346 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 347 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 348 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 349 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 350 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 351 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 352 : Non-relation accuracy: 98% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 353 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 354 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 355 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 356 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 357 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 358 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 359 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 360 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 361 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 362 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 363 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 364 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 365 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 366 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 367 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 368 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 369 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 370 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 371 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 372 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 373 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 374 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 375 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 376 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 377 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 378 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 379 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 380 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 381 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 382 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 383 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 384 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 385 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 386 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 387 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 388 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 389 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 390 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 391 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 392 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 393 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 394 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 395 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 396 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 397 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 398 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 399 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 400 : Non-relation accuracy: 99% | Relation accuracy: 89% | Tricky accuracy: 0% 
  Test set after epoch 401 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 402 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 403 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 404 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 405 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 406 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 407 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 408 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 409 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 410 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 411 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 412 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 413 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 414 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 415 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 416 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 417 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 418 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 419 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 420 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 421 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 422 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 423 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 424 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 425 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 426 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 427 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 428 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 429 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 430 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 431 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 432 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 433 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 434 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 435 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 436 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 437 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 438 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 439 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 440 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 441 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 442 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 443 : Non-relation accuracy: 99% | Relation accuracy: 91% | Tricky accuracy: 0% 
  Test set after epoch 444 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 445 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 446 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 447 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 448 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 449 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 450 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 451 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 452 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 453 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 454 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 455 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 456 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 457 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 458 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 459 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 460 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 461 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 462 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 463 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 464 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 465 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 466 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 467 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 468 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 469 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 470 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 471 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 472 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 473 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 474 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 475 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 476 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 477 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 478 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 479 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 480 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 481 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 482 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 483 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 484 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 485 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 486 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 487 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 488 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 489 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 490 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 491 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 492 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 493 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 494 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 495 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 496 : Non-relation accuracy: 99% | Relation accuracy: 92% | Tricky accuracy: 0% 
  Test set after epoch 497 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 498 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 499 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 500 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 501 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 502 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 503 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 504 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 505 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 506 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 507 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 508 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 509 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 510 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 511 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 512 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 513 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 514 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 515 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 516 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 517 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 518 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 519 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 520 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 521 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 522 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 523 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 524 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 525 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 526 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 527 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 528 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 529 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 530 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 531 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 532 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 533 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 534 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 535 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 536 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 537 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 538 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 539 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 540 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 541 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 542 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 543 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 544 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 545 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 546 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 547 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 548 : Non-relation accuracy: 99% | Relation accuracy: 94% | Tricky accuracy: 0% 
  Test set after epoch 549 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 
  Test set after epoch 550 : Non-relation accuracy: 99% | Relation accuracy: 93% | Tricky accuracy: 0% 


python -u main.py --model=RFESH --epochs=500 --lr=0.001 --seq_len=6 --rnn_hidden_size=64 --coord_extra_len=6 --resume 0 --seed 11 --train_tricky --template model/{}_6item-span-seed11-hidden64-6coord-fuzz1.0-plain-hardtest-tricky_{:03d}.pth | tee --append logs/RFESH_6item-span-seed11-hidden64-6coord-fuzz1.0-plain-hardtest-tricky.log


python -u main.py --model=CNN_MLP --epochs=100 | tee --append model/training_CNN_MLP.log
mv model/training_CNN_MLP.log logs/CNN_MLP_base.log


python -u main.py --model=RN      --epochs=50  --seed 10  | tee --append logs/training_RN_seed10-tmp.log
python -u main.py --model=CNN_MLP --epochs=100 --seed 10  | tee --append logs/training_CNN_MLP-tmp.log
#python -u main.py --model=RFES    --epochs=100 --seed 10 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --seq_len=6 | tee --append logs/training_RFES-tmp.log
python -u main.py --model=RFES    --epochs=100 --seed 10  --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=2 --seq_len=6 | tee --append logs/training_RFES-tmp.log

#python -u main.py --model=RFESH   --epochs=400 --seed 10 --template model/{}_{:03d}-tmp.pth --lr=0.001 --rnn_hidden_size=64 --coord_extra_len=6 --seq_len=6 \
#   | tee --append logs/training_RFESH-tmp.log

#  Seems stuck at 60%
#python -u main.py --model=RFESH   --epochs=400 --seed 10 --template model/{}_{:03d}-tmp.pth --lr=0.0008 --rnn_hidden_size=64 --coord_extra_len=6 --seq_len=6 \
#   | tee --append logs/training_RFESH-tmp.log
