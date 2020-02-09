import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import fewshot_re_kit.network as network
import fewshot_re_kit


class Learner(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
            pos_embedding_dim=5):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.word2id = word2id
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                word_embedding_dim, pos_embedding_dim)
        self.vars = nn.ParameterList()

        self.n_way = 5
        self.feature_dim = 60
        self.filter_num = 128

        # kernel size = 2
        # [ch_out, ch_in, kernelsz, kernelsz]
        for filter_size in [2,3,4,5]:
            w = nn.Parameter(torch.ones(self.filter_num,1,filter_size,self.feature_dim))  # [64,1,3,3]]
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(self.filter_num)))

        filter_dim = self.filter_num*len([2,3,4,5])
        labels_num = self.n_way

        #dropout
        self.dropout = nn.Dropout(0.5)

        #linear
        w = nn.Parameter(torch.ones(labels_num,filter_dim))
        self.linear = nn.Linear(filter_dim,labels_num)
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        # [ch_out]
        self.vars.append(nn.Parameter(torch.zeros(labels_num)))

    def forward(self, x,vars=None):
        '''
        :param x: [1,N,K,MAXLEN*3]
        :param vars:
        :return:
        '''
        if vars is None:
            vars = self.vars
        x = self.embedding(x) #[N*K,MAXLEN,60]
        x = x.unsqueeze(dim=1)

        idx = 0
        bn_idx = 0

        xs = []
        for _ in range(4):
            w,b =vars[idx],vars[idx+1]
            x1 = F.conv2d(x,w,b)
            xs.append(torch.relu(x1.squeeze(3)))
            idx += 2

        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in xs]  # x[idx]: batch_size x filter_num
        sentence_features = torch.cat(x, dim=1)  # batch_size x (filter_num * len(filters))
        x = self.dropout(sentence_features)

        w,b =vars[idx],vars[idx+1]
        x = F.linear(x,w,b)
        idx += 2

        # make sure variable is used properly
        assert idx == len(vars)

        return x




    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

if __name__ == '__main__':
    import json
    from fewshot_re_kit.sentence_encoder import Tokenizer
    from fewshot_re_kit.data_loader import get_loader
    import torch.optim as optim

    try:
        glove_mat = np.load('./pretrain/glove/glove_mat.npy')
        glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
    sentence_encoder = Tokenizer(
        glove_mat,
        glove_word2id,
        128)

    train_data_loader = get_loader('train_wiki', sentence_encoder,
                                   N=5, K=5, Q=5, na_rate=0, batch_size=16)
    model = Learner(glove_mat,glove_word2id,max_length=128)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
    #                        weight_decay=1e-5)
    for _ in range(10):
        x_spt, y_spt, x_qry, y_qry = next(train_data_loader)

        for k in range(16):
            logits = model(x_spt[k], vars=None)
            loss = F.cross_entropy(logits, y_spt[k].long())



