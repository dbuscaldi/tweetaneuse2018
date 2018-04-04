import sys
sys.path.append("./rstr_max/")
import re
from tools import *

def get_ID_classe(ref):
  lignes = open_utf8(ref, True)
  d = {}
  for l in lignes  :
    ID, classe = re.split("\t", l)
    d[ID]=classe
  return d

path = "resDavide/"
dic_T1 = get_ID_classe("%s/T1predictions.txt"%path)
dic_NEG = get_ID_classe("%s/T2pred_NEG.txt"%path)
dic_POS = get_ID_classe("%s/T2pred_POS.txt"%path)
dic_POL = get_ID_classe("%s/T2pred_POL.txt"%path)

dic_res = []
for ID, classe in dic_T1.iteritems():
  if classe=="INCONNU":
    continue
  else:
    if dic_POL[ID]=="NEUTRE":
      pred = "NEUTRE"
    elif dic_NEG[ID]=="NEGATIF" and dic_POS[ID]!="POSITIF":
      pred = "NEGATIF"
    elif dic_NEG[ID]!="NEGATIF" and dic_POS[ID]=="POSITIF":
      pred = "POSITIF"
    elif dic_NEG[ID]=="NEGATIF" and dic_POS[ID]=="POSITIF":
      pred = "MIXPOSNEG"
    else:
      pred = "NEUTRE"
  dic_res.append("%s|%s"%(ID, pred))

write_utf8("resDavide/T2predictions.txt", "\n".join(dic_res))
