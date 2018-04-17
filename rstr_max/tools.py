import codecs
import re
import os

def get_args():
  from optparse import OptionParser
  parser = OptionParser()
  # Options specific to classif
  parser.add_option("-c", "--corpus", dest="corpus",
                  help="Path of the corpus to process", metavar="CORPUS")
  
  parser.add_option("-t", "--task", dest="task", default = "1",
                  help="Number of the task, default 1", metavar="TASK")
  parser.add_option("-T", "--test", dest="test", 
                  default=False, action="store_true",      
                  help = "Apply model on test set")
  parser.add_option("-f", "--force", dest="force", 
                  default=False, action="store_true",      
                  help = "Perform experiment even if already done")
  parser.add_option("-v", "--verbose",
                   action="store_true", dest="verbose", default=False,
                   help="don't print status messages to stdout")
  parser.add_option("-n", "--ngrams",
                   action="store_true", dest="ngrams", default=False,
                   help="ngrams (not motifs)")
  parser.add_option("-l", "--len",
                   dest="len", default="1,3",
                   help="minlen,maxlen")
  parser.add_option("-s", "--sup",
                   dest="sup", default="2,100000",
                   help="minsup,maxsup")


  (options, args) = parser.parse_args()
  return options

def effectif_from_list(liste):
  dic = {}
  for elem in liste:
    dic.setdefault(elem, 0)
    dic[elem]+=1
  return dic

def open_utf8(path,l=False):
  f = codecs.open(path,'r','utf-8')
  if  l==True:
    out = f.readlines()
    out = [re.sub("\n|\r","",x) for x in out]
  else:
    out = f.read()
  f.close()
  return out

def write_utf8(path, out):
  w = codecs.open(path,'w','utf-8')
  w.write(out)
  w.close()
  print("Output written in %s"%path)

def moyenne(L):
  somme = 0
  for i in L:
    somme+=i
  return float(somme)/len(L)
def mkdirs(path):
  try:
    os.makedirs(path)
  except:
    pass

def read_tweets(path):
  lignes = open_utf8(path, True)
  d = {}
  for l in lignes:
    l = re.sub('^"|"$', '', l)
    ID, texte = re.split('"\t"', l)
    d[ID] = texte
  return d

def read_tsv_file(path, key = 0):
  lignes = [re.sub("\n", "",x) for x in open_utf8(path, True)]
  d = {}
  l_attr = re.split("\t", lignes[0])
  for l in lignes[1:]:
    l = re.sub('^"|"$', '', l)
    elems = re.split("\t", l)
    ID = elems[key]
    d[ID] = {l_attr[i]:elems[i] for i in range(len(l_attr))}
  return d

