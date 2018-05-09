import sys
import collections
import os
import numpy as np


def get_raw_data(path):
    with open(path) as f:
        lines = f.read().replace('\n','<eos>').split()
    return lines

def create_word_to_id(lines):
    data = np.hstack([line.replace('\n','<eos>').split() for line in lines])
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    p = count_pairs[0:10]
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    with open("./data.dat","w") as f:
        for w,i in word_to_id.items():
            print ("%s %d"%(w,i))
            f.write("%s %d\n" %(w,i))

    return word_to_id

def read_word_to_id():
    with open("./data.dat","r") as f:
        word_to_id = [line.split() for line in f.readlines()]
        word_to_id = [ (w,int(i)) for w,i in word_to_id]
        return dict(word_to_id)
    
def gen_data(path):
    lines = get_raw_data(path)
    if not os.path.exists('./data.dat'):
        word_to_id=create_word_to_id(lines)
    else:
        word_to_id=read_word_to_id() 

    print ("lines:",lines[0:50])
    return [ word_to_id[w] for w in lines]
    #return [ word_to_id[w] for w in line.split() for line in lines]
    

#print (gen_data(sys.argv[1]))
