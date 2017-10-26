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


#python -u main.py --model=RFS     --epochs=20  | tee --append model/training_RFS.log
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

#python -u main.py --model=RFS --epochs=20 --lr=0.001  | tee model/training_RFS-key-is-param.log
#  Test set after epoch 20 : Relation accuracy: 41% | Non-relation accuracy: 81%

#python -u main.py --model=RFS --epochs=30 --lr=0.0003 --resume epoch_RFS_20.pth | tee --append  model/training_RFS-key-is-param.log
#  Test set after epoch 50 : Relation accuracy: 80% | Non-relation accuracy: 98%

#python -u main.py --model=RFS --epochs=150 --lr=0.0001 --resume epoch_RFS_30.pth | tee --append  model/training_RFS-key-is-param.log
# 220mins expected: ~<4hr


#python -u main.py --model=RFS --epochs=50 --lr=0.001 --resume 0 --template model/{}_2item-span_{:03d}.pth | tee --append logs/RFS_2item-span.log
(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span.log 
#  Test set after epoch  1 : Relation accuracy: 54% | Non-relation accuracy: 57%
#  Test set after epoch  2 : Relation accuracy: 73% | Non-relation accuracy: 60%
#  Test set after epoch  5 : Relation accuracy: 73% | Non-relation accuracy: 63%
#  Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 61%
#  Test set after epoch 20 : Relation accuracy: 80% | Non-relation accuracy: 86%
#  Test set after epoch 30 : Relation accuracy: 93% | Non-relation accuracy: 99%  !!
#  Test set after epoch 40 : Relation accuracy: 94% | Non-relation accuracy: 99%
#  Test set after epoch 50 : Relation accuracy: 92% | Non-relation accuracy: 99%

python -u main.py --model=RFS --epochs=30 --lr=0.001 --resume 0 --template model/{}_1item-span_{:03d}.pth | tee --append logs/RFS_1item-span.log
# This should do well on the nonrel, but fail on the rel .. Hmmm : Not so much as expected...
# grep Test logs/RFS_1item-span.log 
#  Test set after epoch  1 : Relation accuracy: 45% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 44% | Non-relation accuracy: 56%
#  Test set after epoch  5 : Relation accuracy: 52% | Non-relation accuracy: 75%
#  Test set after epoch 10 : Relation accuracy: 50% | Non-relation accuracy: 77%
#  Test set after epoch 20 : Relation accuracy: 75% | Non-relation accuracy: 78%
#  Test set after epoch 30 : Relation accuracy: 78% | Non-relation accuracy: 78%


#python -u main.py --model=RFS --epochs=30 --lr=0.001 --resume 0 --template model/{}_1item-span-gumbel_{:03d}.pth | tee --append logs/RFS_1item-span-gumbel.log
# Just see what Gumbel does in the place of softmax
# (env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_1item-span-gumbel.log 
#  Test set after epoch  1 : Relation accuracy: 48% | Non-relation accuracy: 52%
#  Test set after epoch  2 : Relation accuracy: 58% | Non-relation accuracy: 54%
#  Test set after epoch  5 : Relation accuracy: 67% | Non-relation accuracy: 58%
#  Test set after epoch 10 : Relation accuracy: 69% | Non-relation accuracy: 60%
#  Test set after epoch 20 : Relation accuracy: 71% | Non-relation accuracy: 60%
#  Test set after epoch 30 : Relation accuracy: 70% | Non-relation accuracy: 59%

(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-gumbel.log 
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


#python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --template model/{}_2item-span-seed10_{:03d}.pth | tee --append logs/RFS_2item-span-seed10.log
# hidden_size=32 (as before)
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-seed10.log   # WORKS (better) with different seed
#  Test set after epoch  1 : Relation accuracy: 48% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 55% | Non-relation accuracy: 58%
#  Test set after epoch  5 : Relation accuracy: 73% | Non-relation accuracy: 86%
#  Test set after epoch 10 : Relation accuracy: 84% | Non-relation accuracy: 99%   
#  Test set after epoch 20 : Relation accuracy: 93% | Non-relation accuracy: 100%   !!
#  Test set after epoch 30 : Relation accuracy: 92% | Non-relation accuracy: 100%
#  Test set after epoch 40 : Relation accuracy: 93% | Non-relation accuracy: 100%
#  Test set after epoch 50 : Relation accuracy: 93% | Non-relation accuracy: 100%

#python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=64 --resume 0 --seed 10 --template model/{}_2item-span-hidden64-seed10_{:03d}.pth | tee --append logs/RFS_2item-span-hidden64-seed10.log
# hidden_size=64
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-hidden64-seed10.log 
#  Test set after epoch  1 : Relation accuracy: 45% | Non-relation accuracy: 51%
#  Test set after epoch  2 : Relation accuracy: 43% | Non-relation accuracy: 52%
#  Test set after epoch  5 : Relation accuracy: 67% | Non-relation accuracy: 60%
#  Test set after epoch 10 : Relation accuracy: 73% | Non-relation accuracy: 61%
#  Test set after epoch 20 : Relation accuracy: 74% | Non-relation accuracy: 61%
#  Test set after epoch 30 : Relation accuracy: 74% | Non-relation accuracy: 61%
#  Test set after epoch 40 : Relation accuracy: 82% | Non-relation accuracy: 98%
#  Test set after epoch 50 : Relation accuracy: 90% | Non-relation accuracy: 99%    !!


#python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --process_coords --seed 10 --template model/{}_2item-span-process_coords-seed10_{:03d}.pth | tee --append logs/RFS_2item-span-process_coords.log
# hidden_size=32.  But add coords to CNN output and do 2 further layers of 1x1 covolutions
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-process_coords.log
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


#python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --template model/{}_2item-span-again-seed10_{:03d}.pth | tee --append logs/RFS_2item-span-again.log
#(env3) [andrewsm@square relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-again.log
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

# python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFS_2item-span-tricky.log
# python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 40 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFS_2item-span-tricky.log

# python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --seq_len 3 --template model/{}_3item-span-seed10-tricky_{:03d}.pth | tee --append logs/RFS_3item-span-tricky.log
#  997  cp /mnt/data/Personal/logs.tmp/RFS_3item-span-seed10-tricky_050.pth model/
#  998  cp /mnt/data/Personal/logs.tmp/RFS_3item-span-tricky.log  logs/



# python -u main.py --model=RN --epochs=50 --template model/{}_tricky2_{:03d}.pth --train_tricky | tee --append logs/training_RN_dataset++_tricky2.log   
  
# python -u main.py --model=RFS --epochs=100 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-tricky2_{:03d}.pth | tee --append logs/RFS_2item-span-tricky2.log
# python -u main.py --model=RFS --epochs=100 --lr=0.001 --rnn_hidden_size=32 --resume 0 --seed 1 --train_tricky --template model/{}_2item-span-seed1-tricky2_{:03d}.pth | tee --append logs/RFS_2item-span-seed1-tricky2.log
## Peformance on norel and birel pretty much the same.  trirel a little worse (50s vs 60) - but should be winning...


# python -u main.py --model=RFS --epochs=100 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6  --resume 0 --seed 10 --train_tricky --template model/{}_2item-span-seed10-6coord-tricky2_{:03d}.pth | tee --append logs/RFS_2item-span-6coord-tricky2.log
# python -u main.py --model=RFS --epochs=50 --lr=0.0003 --rnn_hidden_size=32 --coord_extra_len=6 --resume 50 --seed 10 --train_tricky --template model/{}_2item-span-seed10-6coord-tricky2_{:03d}.pth | tee --append logs/RFS_2item-span-6coord-tricky2.log

# python -u main.py --model=RN  --epochs=50 --seed 10 --template model/{}_{:03d}.pth | tee --append logs/training_RN_seed10.log   

(env3) [andrewsm@simlim relationships-from-entity-stream]$ grep Test logs/RFS_2item-span-6coord.log 
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
  Test set after epoch 40 : Non-relation accuracy: 99% | Relation accuracy: 95% | Tricky accuracy: 0% 
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


# python -u main.py --model=RFS --epochs=50 --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=6 --resume 0 --seed 10 --template model/{}_2item-span-seed10-6coord_{:03d}.pth | tee --append logs/RFS_2item-span-6coord.log
# grep Test logs/RFS_2item-span-6coord.log 

# For comparison (back to non-tricky stuff to see whether 99/95 beats it)
# python -u main.py --model=RN  --epochs=50 --seed 10 --template model/{}_{:03d}.pth | tee --append logs/training_RN_seed10.log 

