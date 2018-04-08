import re
import sys
sys.path.append("./rstr_max/")
from tools import *
import glob

def get_majority(l):
  if len(set(l))==1:
    return l[0]
  d = {}
  for vote in l:
    d.setdefault(vote, 0)
    d[vote]+=1
  s = sorted([[y, x] for x, y in d.iteritems()], reverse = True)
  if s[0][0]>s[1][0]:
    return s[0][1]
  else:
    print s
    return [x[1] for x in s]

def get_ID_classe(ref):
  lignes = open_utf8(ref, True)
  d = {}
  for l in lignes  :
    ID, classe = re.split("\|", l)
    d[ID]=classe
  return d

def election(dic_votes):
  cpt_desaccords = 0
  dic_des = {}
  res = {}
  for ID, votes in dic_votes.iteritems():
    maj = get_majority(votes)
    if type(maj) is list:
      print maj
      cpt_desaccords +=1
      dic_des.setdefault(tuple(maj), 0)
      dic_des[tuple(maj)]+=1
      maj = "MIXPOSNEG"
    res[ID] = maj
  print cpt_desaccords
  print dic_des
  return res

d = {}
for source in sys.argv[1:]:
  print source
  preds = get_ID_classe(source)
  for ID, classe in preds.iteritems():
    d.setdefault(ID, [])
    d[ID].append(classe)

res = election(d)

res = ["%s|%s"%(x,y) for x,y in res.iteritems()]
write_utf8("results_vote", "\n".join(res))
