import re
import sys
import glob
sys.path.append("../rstr_max/")
from tools import *

path_test_data = "../data/T1_test"
path_results = "3_TWEETANEUSE"

dic_tweets = read_tweets(path_test_data)

dic_out = {"T1":{}, "T2":{}}
for task in dic_out.keys():
  path = "%s/%s/"%(path_results, task)
  for subpath in glob.glob(path+"/*"):
    subpath+="/REF_HYP_ALGN"
    numrun = re.findall("run[0-9]", subpath)[0]
    dic_task_run = read_tsv_file(subpath, 1)
    for ID, infos in dic_task_run.iteritems():
      tweet_infos = {"good_answers":0, 
		     "ref_annot":dic_task_run[ID]["ref_annot"],
		     "texte":re.sub('"', "'",dic_tweets[ID])}
      dic_out[task].setdefault(ID, tweet_infos)
      if dic_task_run[ID]["OK/ERR"]=="OK":
        dic_out[task][ID]["good_answers"]+=1
      dic_out[task][ID][numrun] = dic_task_run[ID]["hyp_annot"]

attributs = ['run1', 'run2', 'run3', 'run4', 'ref_annot', 'good_answers', 'texte']
for task, tweets in dic_out.iteritems():
  out_path = "vue_%s.csv"%task
  out_lines = ['"ID"'+";"+'";"'.join(attributs)+'"']
  for ID, infos in tweets.iteritems():
    ligne = [ID]
    infos["good_answers"] = str(infos["good_answers"])
    for attr in attributs:
      ligne.append(infos[attr])
    out_lines.append('"'+'";"'.join(ligne)+'"')
  write_utf8(out_path, "\n".join(out_lines))
