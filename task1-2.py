from __future__ import print_function
import time

from collections import defaultdict
#from plot_attention import plot_attention
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np
import pdb

import re

parser = argparse.ArgumentParser()
parser.add_argument("--task2", dest="task2", action="store_true")
parser.add_argument("--iter", type=int, default=10, help="Number of iterations.")

args = parser.parse_args()

# format of files: each line is "ID"\t"string"
# "791363815107465216"	"20h51 : +20min (malaise voyageur avec prise en charge de la personne à Trilport) #ligneP"
train_file = "data/id_tweets"
train_categories = "data/cat_tweets"
#dev_file = "../data/parallel/dev.ja"
#test_file = "../data/parallel/test.en"

# w2i_src = defaultdict(lambda: len(w2i_src))
# w2i_trg = defaultdict(lambda: len(w2i_trg))




def read_data(filename, categories, unk_threshold):
    p = re.compile('^"(.*)"\t"(.*)"$')

    corpus = {}
    with open(filename, "r") as f_src:
        for line in f_src:
            m = p.match(line)
            if m:
                corpus[m.group(1)] = m.group(2)

    # TODO: postprocess txt (ie, line carrier)

    p =  re.compile('^(.*)\|(.*)$')
    with open(categories, "r") as f_cat:
        for line in f_cat:
            m = p.match(line)
            if m:
                corpus[m.group(1)] = (corpus[m.group(1)], m.group(2))

    lex = {}
    for k in corpus:
        s,_ = corpus[k]
        l = s.split()
        for w in l:
            if w in lex:
                lex[w] = lex[w] +1
            else:
                lex[w] = 1
    cpt = 1
    llex = [k for k in lex if lex[k] >= unk_threshold]
    lex = {}
    for k in llex:
        lex[k] = cpt
        cpt = cpt+1

    return corpus,lex


train,lexicon = read_data(train_file, train_categories, 10)
train = list(train.values())

if args.task2:
    train = [ (x,y) for (x,y) in train if y != 'INCONNU']

# DyNet Starts
model = dy.Model()
trainer = dy.AmsgradTrainer(model)

# Model parameters
CHAR_EMBED_SIZE = 32
WORD_EMBED_SIZE = 64

CHAR_HIDDEN_SIZE = 64
WORD_HIDDEN_SIZE = 128

BATCH_SIZE = 16

#Especially in early training, the model can generate basically infinitly without generating an EOS
#have a max sent size that you end at
MAX_SENT_SIZE = 140*4

# Lookup parameters for byte embeddings
CHAR_LOOKUP = model.add_lookup_parameters((255, CHAR_EMBED_SIZE))
WORD_LOOKUP = model.add_lookup_parameters((len(lexicon) + 1, WORD_EMBED_SIZE))


DROPOUT_RATE = 0.5


#we want a bidirectional LSTM for attention over the source
CHAR_LSTMS = [
    dy.LSTMBuilder(1, CHAR_EMBED_SIZE, CHAR_HIDDEN_SIZE, model),
    dy.LSTMBuilder(1, CHAR_EMBED_SIZE, CHAR_HIDDEN_SIZE, model)]

WORD_LSTMS = [
    dy.LSTMBuilder(1, WORD_EMBED_SIZE, WORD_HIDDEN_SIZE, model),
    dy.LSTMBuilder(1, WORD_EMBED_SIZE, WORD_HIDDEN_SIZE, model)]


#hidden size to embbed-layer before softmax
W_m_p = model.add_parameters((WORD_EMBED_SIZE, WORD_HIDDEN_SIZE))
b_m_p = model.add_parameters(WORD_EMBED_SIZE)

#the softmax from the hidden size
output_size = 2
if args.task2:
    output_size = 4
W_sm_p = model.add_parameters((output_size, WORD_EMBED_SIZE))         # Weights of the softmax
b_sm_p = model.add_parameters((output_size)) # Softmax bias


def map_task1(label):
    if label == 'INCONNU':
        return 0
    return 1

def reverse_map_task1(i):
    if i == 0:
        return 'INCONNU'
    else:
        return 'TRANSPORT'

def map_task2(label):
    if label=='POSITIF':
        return 0
    if label=='NEGATIF':
        return 1
    if label == 'NEUTRE':
        return 2
    #MIXPOSNEG
    return 3

def reverse_map_task2(i):
    if i == 0:
        return 'POSITIF'
    if i == 1:
        return 'NEGATIF'
    if i == 2:
        return 'NEUTRE'
    if i == 3:
        return 'MIXPOSNEG'


def calc_output(sents, train_mode):
    dy.renew_cg()

    cache = {}

    cf_init, cb_init = [b.initial_state() for b in CHAR_LSTMS]
    wf_init, wb_init = [b.initial_state() for b in WORD_LSTMS]

    #get input/output for T1
    #get list of tokens
    xs = [x.split() for (x,_) in sents]

    for x in xs:
        for w in x:
            if w not in cache:
                t = [dy.lookup(CHAR_LOOKUP,c) for c in w.encode()]
                fw = [x.output() for x in cf_init.add_inputs(t)]
                bw = [x.output() for x in cb_init.add_inputs(reversed(t))]
                wid = 0
                if w in lexicon:
                    wid = lexicon[w]
                cache[w] = dy.lookup(WORD_LOOKUP,wid) + fw[-1] + bw[-1]

    src_len = [len(x) for x in xs]
    max_src_len = np.max(src_len)
    num_words = 0



    #build the batch. Be careful!
    src_cws = []
    for i in range(max_src_len):
        src_cws.append(dy.concatenate_to_batch([dy.dropout(cache[x[i]], DROPOUT_RATE) if train_mode else cache[x[i]] for x in xs]))


    # TODO: could be optimized
    fw = [x.output() for x in wf_init.add_inputs(src_cws)]
    bw = [x.output() for x in wb_init.add_inputs(reversed(src_cws))]
    finale  = fw[-1] + bw[-1]


    if train_mode:
        finale = dy.dropout(finale, DROPOUT_RATE)

    W_sm = dy.parameter(W_sm_p)
    b_sm = dy.parameter(b_sm_p)

    W_m = dy.parameter(W_m_p)
    b_m = dy.parameter(b_m_p)

    #print(src_output.dim())
    h = dy.rectify(dy.affine_transform([b_m, W_m, finale]))
    if train_mode:
        h = dy.dropout(h, DROPOUT_RATE)
    #print(h.dim())
    e = dy.affine_transform([b_sm, W_sm, h])
    #print (e.dim())
    return e

def calc_loss(sents):
    o = calc_output(sents, True)

    #useless as ys is assigned on both branches
    ys = []
    if args.task2:
        ys = [map_task2(y) for (x,y) in sents]
    else:
        ys = [map_task1(y) for (x,y) in sents]

    e = dy.pickneglogsoftmax_batch(o, ys)
    #print (e.dim())
    return e

def predict(sents):
    e = calc_output(sents, False)

    if args.task2:
        res = [0,0,0,0]
        total = [0,0,0,0]
    else:
        res = [0,0]
        total = [0,0]

    if args.task2:
        ys = [map_task2(y) for (x,y) in sents]
    else:
        ys = [map_task1(y) for (x,y) in sents]

    #print (e.dim())
    n = 0
    #print (ys)
    for i in range(len(ys)):
        b = dy.pick_batch_elem(e,i)
        v = dy.argmax(b, gradient_mode="zero_gradient")
        v = v.vec_value()
        #print(v)
        #print(ys[i],v.index(max(v)))
        if ys[i] == v.index(max(v)):
            res[ys[i]] += 1
        total[ys[i]] += 1
    #print (e.dim())
    #print(n)
    return res,total



# Creates batches where all source sentences are the same length
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x.split()) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches

random.shuffle(train)
l = len(train)
dev = train[int(0.9 *l):]
train = train[:int(0.9*l)]

#print(dev)


train.sort(key=lambda t: len(t[0].split()), reverse=True)
dev.sort(key=lambda t: len(t[0].split()), reverse=True)

print (len(train))
train_order = create_batches(train, BATCH_SIZE)
print (len(train_order))

print (len(dev))
dev_order = create_batches(dev, BATCH_SIZE)
print (len(dev_order))

for ITER in range(args.iter):
    num_sentence = 0
    train_loss = 0.0
    start_time = time.time()


    random.shuffle(train_order)
    for sent_id, (start, length) in enumerate(train_order):
        #print (sent_id,length)
        train_batch = train[start:start+length]
        e = calc_loss(train_batch)
        e = dy.sum_batches(e)
        train_loss += e.value()
        num_sentence += length
        e.backward()
        trainer.update()
        if (sent_id+1) % 100 == 0:
            print("--finished %r sentences" % (sent_id+1))
    end_time = time.time()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss/num_sentence, end_time-start_time))

    start_time = time.time()
    n = 0
    num_sentence = 0

    if args.task2:
        res   = [0,0,0,0]
        total = [0,0,0,0]
    else:
        res   = [0,0]
        total = [0,0]

    for sent_id, (start, length) in enumerate(dev_order):
        #print (sent_id,length)
        dev_batch = dev[start:start+length]
        o,t = predict(dev_batch)
        n += sum(o)
        for i in range(len(res)):
            res[i] += o[i]
            total[i] += t[i]
        num_sentence += length
        if (sent_id+1) % 100 == 0:
            print("--finished %r sentences" % (sent_id+1))
    end_time = time.time()
    print("iter %r: dev acc =%.4f, time=%.2fs" % (ITER, float(n)/float(num_sentence), end_time-start_time))
    for i in range(len(res)):
        print("cat: %r : dev acc =%.4f, (%r/%r)" % (i, float(res[i])/float(total[i]), res[i], total[i]))
