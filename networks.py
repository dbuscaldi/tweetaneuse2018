import numpy as np
import dynet as dy


class Hybrid_BiLSTM:
    def __init__(self, model, data, char_emb_size, word_emb_size, output_emb_size, char_layers, word_layers, dropout_rate):
        self.char_emb_size = char_emb_size
        self.word_emb_size = word_emb_size
        self.output_emb_size = output_emb_size
        self.char_layers = char_layers
        self.word_layers = word_layers
        self.dropout_rate = dropout_rate
        self.word_to_idx = data.word_to_idx

        # Lookup parameters for byte and word embeddings
        self.char_lookup = model.add_lookup_parameters((255, self.char_emb_size))
        self.word_lookup = model.add_lookup_parameters((len(data.word_to_idx) + 1, self.word_emb_size))

        #we want a bidirectional LSTMs
        self.char_lstms = [
            dy.LSTMBuilder(self.char_layers, self.char_emb_size, self.word_emb_size, model),
            dy.LSTMBuilder(self.char_layers, self.char_emb_size, self.word_emb_size, model)]

        self.word_lstms = [
            dy.LSTMBuilder(self.word_layers, self.word_emb_size, self.output_emb_size, model),
            dy.LSTMBuilder(self.word_layers, self.word_emb_size, self.output_emb_size, model)]

    def calc_output(self, sents, train_mode):
        cache = {}

        cf_init, cb_init = [b.initial_state() for b in self.char_lstms]
        wf_init, wb_init = [b.initial_state() for b in self.word_lstms]

        #get input/output for T1
        #get list of tokens
        xs = [ ['<SOS>'] + x.split() + ['<EOS>'] for (x,_) in sents]

        #fill the word embedding cache
        for x in xs:
            for w in x:
                if w not in cache:
                    t = [dy.lookup(self.char_lookup,c) for c in w.encode()]
                    fw = [x.output() for x in cf_init.add_inputs(t)]
                    bw = [x.output() for x in cb_init.add_inputs(reversed(t))]
                    wid = 0
                    if w in self.word_to_idx:
                        wid = self.word_to_idx[w]
                    cache[w] = dy.lookup(self.word_lookup,wid) + fw[-1] + bw[-1]

        src_len = [len(x) for x in xs]
        max_src_len = np.max(src_len)
        num_words = 0

        #build the batch. Be careful!
        src_cws = []
        for i in range(max_src_len):
            src_cws.append(dy.concatenate_to_batch([dy.dropout(cache[x[i]], self.dropout_rate) if train_mode else cache[x[i]] for x in xs]))


        fw = [x.output() for x in wf_init.add_inputs(src_cws)]
        bw = [x.output() for x in wb_init.add_inputs(reversed(src_cws))]

        return (fw,bw)

class mlp:
    def __init__(self, model, size_list, act_fun, dropout_rate):
        self.param_layers = []
        for i in range(len(size_list)-1):
            self.param_layers.append([model.add_parameters(size_list[i+1]), #bias
                                      model.add_parameters((size_list[i+1], size_list[i]))]) #matrix
        self.act_fun = act_fun
        self.dropout_rate = dropout_rate
        self.expressions = []

    def prepare_batch(self):
        self.expressions = []
        for layer in range(len(self.param_layers)):
            self.expressions.append([])
            for i in range(len(self.param_layers[layer])):
                self.expressions[layer].append(dy.parameter(self.param_layers[layer][i]))

    def __call__(self, input, train_mode):
        for layer_idx in range(len(self.expressions)):
            layer = self.expressions[layer_idx]
            input = dy.affine_transform([layer[0], layer[1], input])
            if layer_idx != len(self.expressions) -1:
                input = self.act_fun(input)
                if train_mode:
                    input = dy.dropout(input, self.dropout_rate)
        return input
