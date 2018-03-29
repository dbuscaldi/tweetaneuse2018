import re

class deft_data:

    def __init__(self, filename_id_txt, filename_id_cat, unk_threshold):
        self.id_to_text_cat_map = {}
        self.word_to_idx = {}
        self.unk_threshold = unk_threshold
        self.clean = False

        self.read_data(filename_id_txt, filename_id_cat, self.unk_threshold)




    def has_word(self, word):
        return ((word in self.word_to_idx) and self.word_to_idx[word] >= unk_threshold)


    def read_data(self, filename, categories, unk_threshold):
        p = re.compile('^"(.*)"\t"(.*)"$')

        with open(filename, "r") as f_src:
            for line in f_src:
                m = p.match(line)
                if m:
                    # TODO: postprocess txt (ie, line carrier)
                    if self.clean:
                        self.id_to_text_cat_map[m.group(1)] = m.group(2).replace("[ASCII012CTRLC]", " ")
                    else:
                        self.id_to_text_cat_map[m.group(1)] = m.group(2)



        p =  re.compile('^(.*)\|(.*)$')
        with open(categories, "r") as f_cat:
            for line in f_cat:
                m = p.match(line)
                if m:
                    self.id_to_text_cat_map[m.group(1)] = (self.id_to_text_cat_map[m.group(1)], m.group(2))

        for k in self.id_to_text_cat_map:
            s,_ = self.id_to_text_cat_map[k]
            l = s.split()
            for w in l:
                if w in self.word_to_idx:
                    self.word_to_idx[w] = self.word_to_idx[w] +1
                else:
                    self.word_to_idx[w] = 1

        llex = [k for k in self.word_to_idx if self.word_to_idx[k] >= self.unk_threshold]
        self.word_to_idx = {}
        self.word_to_idx['UNK'] = 0
        self.word_to_idx['<SOS>'] = 1
        self.word_to_idx['<EOS>'] = 2
        cpt = 3
        for k in llex:
            self.word_to_idx[k] = cpt
            cpt = cpt+1


class task1(deft_data):
    def __init__(self, filename_id_txt, filename_id_cat, unk_threshold):
        super().__init__(filename_id_txt, filename_id_cat, unk_threshold)
        self.output_size = 2


    def map_cat(self, label):
        if label == 'INCONNU':
            return 0
        return 1

    def reverse_map_cat(self, i):
        if i == 0:
            return 'INCONNU'
        else:
            return 'TRANSPORT'



class task2(deft_data):
    def __init__(self, filename_id_txt, filename_id_cat, unk_threshold):
        super().__init__(filename_id_txt, filename_id_cat, unk_threshold)
        self.output_size = 4


    def read_data(self, filename, categories, unk_threshold):
        """
        filters out tweets with tag 'INCONNU'
        """
        super().read_data(filename, categories, unk_threshold)
        dict = {}
        for k in self.id_to_text_cat_map:
            s,cat = self.id_to_text_cat_map[k]
            if cat != 'INCONNU':
                dict[k] = (s,cat)
        self.id_to_text_cat_map = dict

    def map_cat(self, label):
        if label=='POSITIF':
            return 0
        if label=='NEGATIF':
            return 1
        if label == 'NEUTRE':
            return 2
        #MIXPOSNEG
        return 3

    def reverse_map_cat(self, i):
        if i == 0:
            return 'POSITIF'
        if i == 1:
            return 'NEGATIF'
        if i == 2:
            return 'NEUTRE'
        if i == 3:
            return 'MIXPOSNEG'
