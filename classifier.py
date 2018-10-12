#!/usr/bin/python3
# -*- coding: utf8 -*-
import sys, os, codecs
import csv, re

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from nltk import wordpunct_tokenize
from nltk import ngrams

import numpy, math
from scipy.sparse import lil_matrix

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


OUTPUT="TESTING" #testing shows accuracy reports, production shows the predictions, debug shows the misclassified sentences
#TESTING, PRODUCTION, DEBUG

SUBTASK="A" #"A", "A1", or "B"
LANG="it"

MINFREQ=5

root = "/Users/dbuscaldi/Works/EVALITA2018"

sys.stderr.write("Loading dictionaries...\n")
if LANG=="it":
	skip_tokens=stopwords.words("italian")
else:
	skip_tokens=stopwords.words("english")
skip_tokens.append("@")
skip_tokens.append("#")
skip_tokens.append("'")
skip_tokens.append(".")
skip_tokens.append(",")
skip_tokens.append(";")
skip_tokens.append("\\")
skip_tokens.append("/")
skip_tokens.append("'")
skip_tokens.append("-")
skip_tokens.append("--")
skip_tokens.append("://")
skip_tokens.append("&")

##########################################################

sys.stderr.write("Reading training data...\n")

if LANG=="it" :
	tr_file="AMI/it_training.tsv"
	ts_file="AMI/it_testing.tsv"
else :
	tr_file="AMI/en_training.tsv"
	ts_file="AMI/en_testing.tsv"

docs = {} #maps id into text (training)
tdocs = {} #maps id into text (testing)
labelsA = {} #maps id into binary category (miso/not) - subtask A
labelsA1 ={} #maps id into miso category - subtask A
labelsB = {} #maps id into type (binary) category (subtask B)

wordfreqs={}
with open(tr_file, 'rt') as csvfile:
	freader = csv.reader(csvfile, delimiter='\t', quotechar='"')
	for row in freader:
		id = row[0]
		if id=="id": continue #skip header
		text =row[1]
		#remove line breaks
		text=re.sub("\\\\", " ", text)
		#replace http addresses with tag
		text=re.sub("http(s)?://.+", "http", text)
		docs[id]=text
		labelsA[id]=row[2]
		labelsA1[id]=row[3]
		labelsB[id]=row[4]

with open(ts_file, 'rt') as csvfile:
	freader = csv.reader(csvfile, delimiter='\t', quotechar='"')
	for row in freader:
		id = row[0]
		if id=="id": continue #skip header
		text =row[1]
		#remove line breaks
		text=re.sub("\\\\", " ", text)
		#replace http addresses with tag
		text=re.sub("http(s)?://.+", "http", text)
		tdocs[id]=text

tknzr= TweetTokenizer(preserve_case=False, reduce_len=True)

def add_to_dict(w) :
	try:
		freq=wordfreqs[w]
	except KeyError:
		freq=0
	freq+=1
	wordfreqs[w]=freq

sys.stderr.write("Building dict...\n")
#building the dictionary
for d in docs.items():
	vecs=[]
	tokens=tknzr.tokenize(d[1])
	#print tokens

	for t in tokens:
		if t not in skip_tokens:
			ngs=set([])
			for n in (3,4,5,6):
				ngrs=ngrams(t, n)
				for ng in ngrs:
					ngs.add(''.join(ng))
			for el in ngs:
				add_to_dict(el)
			if t=="!" or t=="?":
				add_to_dict(t)

wlist=sorted(wordfreqs.items(), key=lambda x : -x[1])

blist=list(filter(lambda x : x[1] >= MINFREQ ,  wlist)) #keep only words that occur at least MINFREQ
#blist=wlist[0:MAX_FEAT]

a_i=0 #let's number attributes to index them
widx={} #maps word into its index
for el in blist:
	widx[el[0]]=a_i
	a_i+=1

n_features=len(blist)
#n_features=len(blist)+3 #+1 for emoticons +1 for sentiment positional score, +1 for sentiment positional score
n_samples=len(docs.items())
n_tests=len(tdocs.items())

MAX_FEAT=n_features
#MAX_FEAT=n_features-3

indices=[] #insert IDs of sentences to retrieve them for error analysis

#writing attributes IDs dictionary
dof=codecs.open("attr_IDs.txt", "w", "utf-8")
for k in blist:
	dof.write("attribute #"+str(widx[k[0]])+" : "+k[0]+"\n")
dof.close()

sys.stderr.write("Building matrix...\n")

#creating training matrix
M = lil_matrix((n_samples, n_features), dtype=numpy.float32) #counts: numpy.int64
#testing matrix
X = lil_matrix((n_tests, n_features), dtype=numpy.float32) #counts: numpy.int64

ninstance=0
clabels=[] #vector of labels
csubA={"stereotype":1, "dominance":2, "derailing":3, "sexual_harassment":4, "discredit":5}

if SUBTASK=="A":
	class_counts=[0,0]
elif SUBTASK=="A1":
	class_counts=[0,0,0,0,0,0]
else:
	class_counts=[0,0,0]

#Make training matrix
for d in docs.items():
	linecounts={}
	tweet_id=d[0]
	nclsA=labelsA[tweet_id]
	nclsA1=labelsA1[tweet_id]
	nclsB=labelsB[tweet_id]
	if nclsA=="0": cls=0
	else:
		if SUBTASK=="A":
			cls=1 #misogyny class
		elif SUBTASK=="A1":
			cls=csubA[nclsA1]
		else:
			if nclsB=="active": cls=1
			elif nclsB=="passive": cls=2

	class_counts[cls]=class_counts[cls]+1

	tokens=tknzr.tokenize(d[1])
	position=0 #position in sentence
	length_sen=len(tokens) #size of sentence in tokens
	for t in tokens:
		if t not in skip_tokens:
			pos_weight= 1.0+float(position)/float(length_sen)
			sent_idx=int(round(float(position)/float(length_sen)))
			ngs=set([])
			for n in (3,4,5,6):
				ngrs=ngrams(t, n)
				for ng in ngrs: ngs.add(''.join(ng))
				for el in ngs:
					try:
						idx=widx[el]
					except KeyError:
						continue
					try:
						lc=linecounts[idx]
					except KeyError:
						lc=0
					linecounts[idx]=(lc+pos_weight)
		position+=1
	sidx=sorted(linecounts.items(), key=lambda x : x[0])
	#insert data into matrix
	for s in sidx:
		M[ninstance, s[0]]=s[1]
	clabels.append(cls)
	indices.append(int(tweet_id))
	ninstance+=1

b = numpy.array(clabels, dtype=numpy.int64) #target
indexes = numpy.array(indices, dtype=numpy.int64)

#make testing matrix
if OUTPUT=="PRODUCTION":
	indices=[]
	ntinstance=0
	for d in tdocs.items():
		linecounts={}
		tweet_id=d[0]
		tokens=tknzr.tokenize(d[1])
		position=0 #position in sentence
		length_sen=len(tokens) #size of sentence in tokens
		for t in tokens:
			if t not in skip_tokens:
				pos_weight= 1.0+float(position)/float(length_sen)
				sent_idx=int(round(float(position)/float(length_sen)))
				ngs=set([])
				for n in (3,4,5,6):
					ngrs=ngrams(t, n)
					for ng in ngrs: ngs.add(''.join(ng))
					for el in ngs:
						try:
							idx=widx[el]
						except KeyError:
							continue
						try:
							lc=linecounts[idx]
						except KeyError:
							lc=0
						linecounts[idx]=(lc+pos_weight)
			position+=1
		sidx=sorted(linecounts.items(), key=lambda x : x[0])
		#insert data into matrix
		for s in sidx:
			X[ntinstance, s[0]]=s[1]
		clabels.append(cls)
		indices.append(int(tweet_id))
		ntinstance+=1
print ("class balance: %s", class_counts)

print (M.shape, b.shape, indexes.shape)

if OUTPUT=="PRODUCTION": print (X.shape)

#sys.stderr.write("Selecting best features...\n")
#selector=SelectKBest(mutual_info_classif, k=10)
#M_new = selector.fit_transform(M, b)
#sel_indices = selector.get_support(indices=True)
#print sel_indices

#print M_new.shape, b.shape, indexes.shape

#classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', kernel='rbf', max_iter=5000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
#classifier = NuSVC(nu=0.5, kernel='rbf')
classifier = RandomForestClassifier(n_estimators=300, random_state=0)

if OUTPUT != "PRODUCTION":
	X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(M, b, indexes, test_size=0.2, random_state=73)

	sys.stderr.write("Building model...\n")
	classifier.fit(X_train, y_train)

	sys.stderr.write("Generating predictions..\n")
	y_pred=classifier.predict(X_test)

	if OUTPUT=="TESTING":
	  sys.stderr.write("Testing...\n")

	  #acc=classifier.score(X_test, y_test)

	  #print acc
	  print (classification_report(y_pred, y_test))

	if OUTPUT=="DEBUG":
	  for i in xrange(len(y_pred)):
	    if y_pred[i] != y_test[i]:
	      print (y_pred[i]+'\t'+y_test[i]+'\t'+docs[str(i_test[i])])
else:
	sys.stderr.write("Building model... \n")
	classifier.fit(M,b)

	sys.stderr.write("Generating predictions..\n")
	y_pred=classifier.predict(X)

	results=[]
	for i in xrange(len(y_pred)):
		results.append((indices[i], y_pred[i]))
		#print indices[i], y_pred[i]
	results.sort(key=lambda x : x[0])
	of=open("res"+SUBTASK+".txt", "w")
	for r in results:
		of.write(str(r[0])+'\t'+str(r[1]))
		of.write("\n")
	of.close
