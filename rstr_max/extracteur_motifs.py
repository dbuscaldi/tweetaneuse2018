#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from rstr_max.rstr_max import *
#from sklearn import datasets
import re
import glob

def exploit_rstr(r, rstr, dic_occur, options):
    l_str = []
    cpt_ss = 0
    for (offset_end, nb), (l, start_plage) in r.iteritems():
        ss = rstr.global_suffix[offset_end-l:offset_end]
        if len(ss)<options['minlen'] or len(ss)>options['maxlen']:
            continue
        cpt_ss+=1
        set_occur = set()
        for o in xrange(start_plage, start_plage+nb) :
            id_str = rstr.idxString[rstr.res[o]]
            dic_occur[id_str][cpt_ss]=dic_occur[id_str].setdefault(cpt_ss,0)+1
            set_occur.add(id_str)
        if len(set_occur)>=options['minsup'] and len(set_occur)<=options['maxsup'] :
            occurences = {}
            for x in set_occur:
                occurences[x] = dic_occur[x][cpt_ss]
            sortie = [ss, occurences]
            l_str.append(sortie)
    return l_str

def get_GR(paths):
  print paths
  path2, path1 = paths
  liste1 = glob.glob(path1+"/*")
  liste2 = glob.glob(path2+"/*")
  all_path = liste1+liste2
  set1 = set(range(0, len(liste1)))
  set2 = set(range(len(liste1), len(liste1)+len(liste2)))
  lignes_texte = []
  for fic in all_path:
      f= open(fic)
      chaine=f.read()
      lignes_texte.append(chaine)
      f.close()
  l_str = get_motifs_old(lignes_texte)
  liste_GR = []
  for motif, dic_occ in l_str:
      id_fics = set(dic_occ.keys())
      S1 = len(set1.intersection(id_fics))
      S2 = len(set2.intersection(id_fics))
      if S1==0:
          GR= 0
      elif S2==0:
          GR= len(set1)+1
      else:
          GR=float(S1)/S2
      liste_GR.append([GR, S1,S2, motif])
  for elems in sorted(liste_GR):
      print elems[3],"\t\t\t","\t".join([str(x) for x in elems[:3]])

def get_n_grams(textes, options):
  dic ={}
  cpt_texte = 0
  for t in textes:
    for i in range(0, len(t)):
      for j in range(options["minlen"], options["maxlen"]+1):
        car = t[i:i+j]
        dic.setdefault(car, {})
        dic[car].setdefault(cpt_texte, 0)
        dic[car][cpt_texte]+=1
    cpt_texte+=1
  L= []
  for ss, occs in dic.iteritems():
    L.append([ss, occs])
  return L
  
  #  pass
def get_motifs(lignes_texte,options = { 'minsup':1,
                                        'maxsup':10,
                                        'minlen':1,
                                        'maxlen':10}):
    if options["ngrams"]==True:
      l_motifs = get_n_grams(lignes_texte, options)
      return l_motifs
    rstr = Rstr_max()
    for ligne in lignes_texte:
        rstr.add_str(ligne)
    dic_occur= {x:{} for x in xrange(0,len(lignes_texte))}
    r=rstr.go()
    l_motifs = exploit_rstr(r, rstr, dic_occur, options)
    return l_motifs

