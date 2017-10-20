import cv2
import os
import numpy as np
import random

import pickle

import argparse

parser = argparse.ArgumentParser(description='PyTorch Relations-from-Stream sort-of-CLVR dataset builder')
parser.add_argument('--dir', type=str, default='./data',  
                    help='Directory in which to store the dataset')
parser.add_argument('--add_tricky', action='store_true', default=True,
                    help='Add the tricky cases')

parser.add_argument('-f', type=str, default='',  help='Fake for Jupyter notebook import')
                 
args = parser.parse_args()
dirs = args.dir 

train_size, test_size = 9800, 200

img_size, size = 75, 5  # Size of img total, radius of sprite

question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
"""Question:[r, g, b, o, k, y, q1, q2, s1, s2, s3]"""

# answer is returned as an integer index within the following:
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
"""Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]""" # for counting

nb_questions = 10   # questions generated about each image

colors = [
    (0,0,255),     ##r  red
    (0,255,0),     ##g  green
    (255,0,0),     ##b  blue
    (0,156,255),   ##o  orange
    (128,128,128), ##k  grey
    (0,255,255)    ##y  yellow
]

def center_generate(objects):
    # Generates a set of centers that do not overlap
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_dataset(nb_questions=nb_questions):
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id,center,'c'))


    """Non-relational questions"""
    norel_questions, norel_answers = [], []
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[6] = 1
        subtype = random.randint(0,2)
        question[subtype+8] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:
            """query shape->rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)

    
    """Relational questions"""
    birel_questions,   birel_answers   = [], []
    for i in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[7] = 1
        subtype = random.randint(0,2)
        question[subtype+8] = 1
        birel_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """count->1~6"""  
            """Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        birel_answers.append(answer)


    """Tricky questions"""
    trirel_questions,   trirel_answers   = [], []
    for i in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[6] = 1  # Both 6 and 7 set
        question[7] = 1  # Both 6 and 7 set
        subtype = random.randint(0,2)
        question[subtype+8] = 1
        trirel_questions.append(question)

        if subtype == 0:
            """three colours are ordered clockwise -> yes/no"""
            # TODO!
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            """three colours enclose another object -> yes/no"""
            # TODO!
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """What shape is between two colours -> rectangle/circle"""
            # TODO!
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        trirel_answers.append(answer)

    norelations = (norel_questions, norel_answers)
    birelations = (birel_questions, birel_answers)
    trirelations = (trirel_questions, trirel_answers)
    
    img = img/255.
    dataset = (img, norelations, birelations, trirelations)
    return dataset

#"""Question:[r, g, b, o, k, y, q1, q2, s1, s2, s3]"""
# Answer is returned as an integer index within the following:
#"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
#"""Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]""" # for counting

## Ideas for tougher questions :
# For the 3 highlighted colours, are they clockwise (in order)
# For the 3 highlighted colours, do they enclose another object
# For the 2 highlighted colours, what shape is between them?

# For the 3 highlighted colours, are they in a row?  (any orientation - tricky to define)

## Not so tough
# For the n highlighted colours, are they all the same shape?  
#   But two different => no.  So don't have to think more than two deep...
# For the 3 highlighted colours, are they in a row?  (horizontal or vertical)
#   Can cheat by counting total in a row or column if orientated


## alternative within Jupyter notebook :
# import sort_of_clevr_generator

if __name__ == "__main__":
    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))

    print('building test datasets...')
    test_datasets = [build_dataset() for _ in range(test_size)]
    print('building train datasets...')
    train_datasets = [build_dataset() for _ in range(train_size)]

    #img_count = 0
    #cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))

    print('saving datasets...')
    filename = os.path.join(dirs,'sort-of-clevr++.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((train_datasets, test_datasets), f)
    print('datasets saved at {}'.format(filename))


