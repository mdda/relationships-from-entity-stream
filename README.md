# relationships-from-entity-stream

Research presented at the [NIPS 2017 ViGiL Workshop](https://nips2017vigil.github.io/) : 

*  [Poster presented](http://redcatlabs.com/downloads/research/2017-12-08_NIPS-2017-ViGIL-Poster_v12.pdf)
*  [Paper submitted](http://redcatlabs.com/downloads/research/2017-12-08_nips2017_ViGIL-workshop_mdda.pdf) - and [on the ViGiL site](https://github.com/nips2017vigil/nips2017vigil.github.io/raw/master/papers/2017/relationships_from%20entity.pdf)


This research extended the work in the DeepMind "Relation Networks" paper : [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf).

### Abstract

>    Relational reasoning is a central component of intelligent behavior, 
>    but has proven difficult for neural networks to learn.  The Relation Network (RN) 
>    module was recently proposed by DeepMind to solve such problems, 
>    and demonstrated state-of-the-art results on a number of datasets.  However, 
>    the RN module scales quadratically in the size of the input, 
>    since it calculates relationship factors between every patch in the visual field, 
>    including those that do not correspond to entities.  In this paper, 
>    we describe an architecture that enables relationships to be determined 
>    from a stream of entities obtained by an attention mechanism over the input field.  The model 
>    is trained end-to-end, and demonstrates 
>    equivalent performance with greater interpretability 
>    while requiring only a fraction of the model parameters of the original RN module.  

The aim of this repo is to make the results of the NIPS ViGIL Workshop paper fully 
reproducible in a turn-key fashion.  The code in the repo on the submission date 
produced the RN, RFS and RFSH results cleanly - though I'm still trying to find the 
run that produced the (not particularly relevant) CNN scores.  


## Sort-of-CLEVR

Sort-of-CLEVR is simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/).  This is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle), and placed in a image.

Non-relational questions are composed of 3 subtypes:

1) Shape of certain colored object
2) Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
3) Vertical location of certain colored object : whether it is on the upside of the image or downside of the image

Theses questions are "non-relational" because the agent only need to focus on certain object.

Relational questions are composed of 3 subtypes:

1) Shape of the object which is closest to the certain colored object
1) Shape of the object which is furthest to the certain colored object
3) Number of objects which have the same shape with the certain colored object

These questions are "relational" because the agent has to consider the relations between objects.

Questions are encoded into a vector of size of 11 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 3 for one-hot vector of 3 subtypes.

<img src="./data/sample.png" width="256">

i.e. : With the sample image shown, we can generate non-relational questions like:

1) What is the shape of the red object? => Circle (even though it does not really look like "circle"...)
2) Is green object placed on the left side of the image? => yes
3) Is orange object placed on the upside of the image? => no

And relational questions:

1) What is the shape of the object closest to the red object? => square
2) What is the shape of the object furthest to the orange object? => circle
3) How many objects have same shape with the blue object? => 3

There is an additional 'tricky' set of questions included in the code, 
which were not part of the original Sort-of-CLEVR setup : 

1) How many things are colinear with 2 chosen colours?
2) How many things are eqidistant from 2 chosen colours?
3) How many things are on clockwise side of line joining 2 chosen colours?


These are also testable using the included models with a ```--tricky``` parameter added.  However, 
they didn't seem to draw out any wothwhile differences between the models, so the workshop paper
didn't report these extended numbers.


## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch](http://pytorch.org/)
- [opencv](http://opencv.org/)  # For sort-of-clevr-generator



## Usage 

Create the sort-of-clevr dataset : 

```
python sort_of_clevr_generator.py
```

There seems to be an issue with (exact) reproducability in PyTorch, even though the ```--seed``` values
are set in ```random.```, ```numpy.``` and ```torch.``` contexts.  If there is something
else that needs to be done, please file an issue.

Train (and test) each of the models in turn : 

**Relation Network**
```
python -u main.py --model=RN --epochs=50 --seed 10 --template model/{}_{:03d}.pth \
  | tee --append logs/training_RN.log
# Each epoch ~86secs on Titan X (Maxwell)
grep Test logs/training_RN_seed10.log    # To plot out the Test performance curve
```

**CNN-MLP**
```
python -u main.py --model=CNN_MLP --epochs=100 --seed 10 --template model/{}_{:03d}.pth \
  | tee --append logs/training_CNN-MLP.log
# Each epoch ~26secs on Titan X (Maxwell) - only gets to NonRel=70%, BiRel=67%
```

**Relationships from Entity Stream (soft attention)**
```
python -u main.py --model=RFES --epochs=100 --seed 10 --template model/{}_{:03d}.pth \
   --lr=0.001 --rnn_hidden_size=32 --coord_extra_len=2 --seq_len=6 \
   | tee --append logs/training_RFES.log
# Each epoch ~51secs on Titan X (Maxwell)
```

**Relationships from Entity Stream (hard attention)**
```
python -u main.py --model=RFESH --epochs=400 --seed 10 --template model/{}_{:03d}.pth \
   --lr=0.001 --rnn_hidden_size=64 --coord_extra_len=6 --seq_len=6 \
   | tee --append logs/training_RFESH.log
# Each epoch ~55secs on Titan X (Maxwell) (still confirming)
```


## Results

The results for all the RN/RFES/RFESH in the NIPS 2017 ViGIL workshop paper 
should be reproducible from the code in this repo : If you have difficulty, 
please let the author know.  Clearly, there is some code-cleanup required...

<!--
| | Relational Networks (20th epoch) | CNN + MLP (without RN, 100th epoch) |
| --- | --- | --- |
| Non-relational question | 99% | 66% |
| Relational question | 89% | 66% |
!-->

|                | Non-relational questions | Relational questions |
| ---                                 | --- | --- |
| Relational Networks (50th epoch)    |    99% |    94% |
| *CNN + MLP (per RN paper, 100th epoch)* |    *98%* |    *62%* |
| CNN + MLP (as here, 100th epoch)    |    67% |    67% |
| Rels from Ent Stream 'soft' (RFES)   |    99% |    95% |
| Rels from Ent Stream 'hard' (RFES.H)  |    99% |    93% |


(*) the first CNN+MLP result set is as shown in the original Relation Network paper, and isn't reproduced
by this code (most likely because the CNN_MLP layers are much smaller).  The second CNN_MLP
version is what is reproduced here (along with all the other results) - and little effort was 
put into reproducing the CNN numbers, since the focus was mainly on achieving results competitive
(or better) than the Relation Networks paper, while having a more satisfying internal structure.


### Code Credits

The basic implementation here was derived from [kimhc6028/relational-networks](/kimhc6028/relational-networks), which 
credits [@gngdb](https://github.com/gngdb) for speeding up the model by 10 times.

The implementation of the "Relationship from Entity Stream" is the bulk of the new code here, and
the Sort-of-CLEVR generator has also been cleaned up (and extended, though that code wasn't used in the results submitted).

