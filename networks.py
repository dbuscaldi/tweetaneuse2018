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

    def calc_embeddings(self, sents, train_mode):
        fw,bw = self.calc_output(sents, train_mode)
        return [x+y for (x,y) in zip(fw,reversed(bw))]


class MultiLayerPerceptron:
    def __init__(self, model, size_list, act_fun, dropout_rate):
        self.param_layers = []
        for i in range(len(size_list)-1):
            self.param_layers.append([model.add_parameters(size_list[i+1]), #bias
                                      model.add_parameters((size_list[i+1], size_list[i])), #matrix
                                      model.add_parameters(size_list[i], init=dy.ConstInitializer(1)), #norm gain
                                      model.add_parameters(size_list[i], init=dy.ConstInitializer(0)) #norm bias
            ])
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
            if layer_idx == 0:
                input = dy.layer_norm(input, layer[2], layer[3])
            input = dy.affine_transform([layer[0], layer[1], input])
            if layer_idx != len(self.expressions) -1:
                input = self.act_fun(input)
                if train_mode:
                    input = dy.dropout(input, self.dropout_rate)
        return input

class BiLSTM_Wrapper:
    def __init__(self, num_layers, input_size, output_emb_size, model):
        self.lstm = dy.BiRNNBuilder(num_layers, input_size, output_emb_size, model, dy.LSTMBuilder)
        self.output_emb_size = output_emb_size

    def calc_embeddings(self, sents, train_mode):
        return self.lstm.transduce(sents[0])




class BiLSTM_CRF:
    def __init__(self, model, tags, lstm, dropout_rate):
        self.model = model
        self.tags = tags
        self.nb_tags = len(tags)
        self.lstm = lstm #todo create the lstm here

        # Matrix that maps from Bi-LSTM output to num tags
        self.mlp = MultiLayerPerceptron(self.model, [self.lstm.output_emb_size, self.nb_tags, self.nb_tags], dy.tanh, dropout_rate)
        # self.lstm_to_tags_params = self.model.add_parameters((nb_tags, self.lstm.output_emb_size))
        # self.lstm_to_tags_bias = self.model.add_parameters(nb_tags)
        # self.mlp_out = self.model.add_parameters((nb_tags, nb_tags))
        # self.mlp_out_bias = self.model.add_parameters(nb_tags)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((self.nb_tags, self.nb_tags))

        self.t2i = {}
        self.t2i['<SOS>'] = 0
        self.t2i['<EOS>'] = 1
        for t in tags:
            self.t2i[t] = len(self.t2i)

        self.training_mode = True



    def unigrams(self, sentence):
        embeddings = self.lstm.calc_embeddings([sentence], self.training_mode)
        scores = []
        for rep in embeddings:
            score_t = self.mlp(rep,False)
            scores.append(score_t)
        return scores


    #score one path in the lattice
    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        #score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.t2i["<SOS>"]] + [self.t2i[t] for t in tags]
        for i, obs in enumerate(observations):
            #+ dy.pick(dy.lookup(self.transitions, tags[i+1]),tags[i])
            score = score + dy.pick(self.transitions[tags[i+1]], tags[i]) + dy.pick(obs, tags[i+1])
            #score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.t2i["<EOS>"]], tags[-1])
        return score



    #score sum of all paths
    def forward(self, observations):
        #avoid over/under-flows
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.nb_tags)
            return max_score_expr + dy.log(dy.sum_dim(dy.transpose(dy.exp(scores - max_score_expr_broadcast)),[1]))

        init_alphas = [-1e10] * self.nb_tags
        init_alphas[self.t2i["<SOS>"]] = 0

        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.nb_tags):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.nb_tags)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.t2i["<EOS>"]]
        alpha = log_sum_exp(terminal_expr)
        return alpha


    def neg_log_loss(self, sentence, tags):
        observations = self.unigrams(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars   = [-1e10] * self.nb_tags
        init_vvars[self.t2i["<SOS>"]] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.transitions[idx] for idx in range(self.nb_tags)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.nb_tags):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.t2i["<EOS>"]]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop() # Remove the start symbol
        best_path.reverse()
        assert start == self.t2i["<SOS>"]
        # Return best path and best path's score

        return best_path, path_score
