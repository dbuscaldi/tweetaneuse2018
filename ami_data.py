import re, csv

class ami_data:

    def __init__(self, filename_id_txt, unk_threshold):
        self.id_to_text_cat_map = {} #maps id to (text, category) 2-uple
        self.word_to_idx = {} #maps word to index
        self.unk_threshold = unk_threshold
        self.clean = False

        self.read_data(filename_id_txt, self.unk_threshold) #skip filename_id_cat

    def has_word(self, word):
        return ((word in self.word_to_idx) and self.word_to_idx[word] >= unk_threshold)


    def read_data(self, filename, unk_threshold):
        with open(filename, 'r') as csvfile:
            freader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for row in freader:
                id = row[0]
                if id=="id": continue #skip header
                text =row[1]
                #print (id+"\t"+text)
                #remove line breaks
                text=re.sub("\\\\", " ", text)
                text=re.sub("http(s)?://.+", "http", text)
                cat=row[3]
                self.id_to_text_cat_map[id]=(text, cat)

        for k in self.id_to_text_cat_map:
            s,cat = self.id_to_text_cat_map[k]
            if cat != None:
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


class task1(ami_data):
    def __init__(self, filename_id_txt, unk_threshold):
        super().__init__(filename_id_txt, unk_threshold)
        self.output_size = 2


    def map_cat(self, label):
        if label == '0':
            return 0
        return 1

    def reverse_map_cat(self, i):
        if i == 0:
            return '0'
        else:
            return '1'



class task2(ami_data):
    def __init__(self, filename_id_txt, unk_threshold):
        super().__init__(filename_id_txt, unk_threshold)
        self.output_size = 6 # TO BE TESTED
        #"stereotype":1, "dominance":2, "derailing":3, "sexual_harassment":4, "discredit":5

    def read_data(self, filename, unk_threshold):
        """
        filters out tweets with tag 'INCONNU'
        """
        super().read_data(filename, unk_threshold)
        dict = {}
        for k in self.id_to_text_cat_map:
            s,cat = self.id_to_text_cat_map[k]
            if unk_threshold == 0 or cat != '0':
                dict[k] = (s,cat)
        self.id_to_text_cat_map = dict

    def map_cat(self, label):
        if label=='0':
            return 0
        if label=='stereotype':
            return 1
        if label == 'dominance':
            return 2
        if label == 'derailing':
            return 3
        if label == 'sexual_harassment':
            return 4
        #discredit
        return 5

    def reverse_map_cat(self, i):
        if i == 0:
            return '0'
        if i == 1:
            return 'stereotype'
        if i == 2:
            return 'dominance'
        if i == 3:
            return 'derailing'
        if i == 4:
            return 'sexual_harassment'
        if i == 5:
            return 'discredit'
