import re
import sys
sys.path.append("rstr_max/")
import os
from extracteur_motifs import *

class Text():
  def __init__(self,path):
    self.title = self.getTitle(path)
    self.date = self.getDate(path)
    self.body = self.getBody(path)
    self.classe = self.getClasse(path)
  def getTitle(self, path):
    return (path.split("/")[-1])
  def getDate(self, path):
    return (path.split("/")[-1]).split("_")[0]
  def getClasse(self, path):
    return (path.split("/")[-2]).split("_")[0]
  def getBody(self, path):
    f = open(path)
    body = f.read()#.replace('\n', '</p>')
    f.close()
    return re.sub(' +', " ", body)
