import sys
sys.path.append("rstr_max")
from tools import * 
import glob
import re

def parse_file(path):
  lignes = open_utf8(path, True)
  dic = {}
  for lig in lignes:
    ID, classe = re.split("\|", lig)
    dic[ID] = classe
  return dic

def init_data_struct(classes_names):
  mes = ["VP", "FP", "FN", "VN"]
  d_classes = {}
  for name in classes_names:
    d_classes[name] = {}
    for s in mes:
      d_classes[name][s] = 0
  return d_classes

def compute_results(dic):
  all_F = []
  acc_data = [0, 0]
  out =""
  print "  CLASSE\tR\tP\tF1"
  for classe, scores in dic .iteritems():
    VP = float(scores["VP"])
    FP = scores["FP"]
    FN = scores["FN"]
    out+= classe+ str(scores)
    if VP==0:R=P=F1=0
    else:
      R = VP/(VP+FN)
      P = VP/(VP+FP)
      F1 = (2*P*R)/(P+R)
    acc_data[0]+=VP
    acc_data[1]+=FP+VP
    all_F.append(F1)
    print "  '%s'"%classe[:10]+"\t"+"\t".join([str(round(x,2)) for x in [R, P, F1]])
  print "  Accuracy:", round(acc_data[0]/acc_data[1], 2)
  print "  Macro F1:", round(moyenne(all_F), 2)
  print out

def get_scores(ref, pred):
  classes_names = set(ref.values())
  print classes_names
  d_classes = init_data_struct(classes_names)
  missing = []
  print len(pred), " predictions pour", len(ref), "references"
  for ID, classe in ref.iteritems():
    if ID not in pred:
      missing.append(ID)
      continue
    if classe==pred[ID]:
      d_classes[classe]["VP"]+=1
    else:
      d_classes[classe]["FN"]+=1
      d_classes[pred[ID]]["FP"]+=1
  compute_results(d_classes) 
  if missing>0:
    print "Missing : %s"%str(len(missing))

print "Usage : ARG1=GOLD_FILE ARG2= PATH_RESULTS"
if len(sys.argv)!=3:
  exit()


liste_results = glob.glob(sys.argv[2]+"*")

ref = parse_file(sys.argv[1])

for result_file in liste_results:
  print "\n Processing %s"%result_file
  pred = parse_file(result_file)
  scores = get_scores(ref, pred)
  print ""
