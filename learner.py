import fewshot_re_kit.network as network
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers.modeling_bert import BertModel

import pdb
import logging
logger = logging.getLogger(__name__)


class BertLearner(nn.Module):

    def __init__(self, max_length, n_way, type="cnnLinear"):
        '''
        :param max_length:
        :param n_way:
        :param type: "cnnLinear" "concatLinear" "clsLinear"
        '''
        nn.Module.__init__(self)
        self.max_length = max_length
        pretrain_path = './pretrain/bert-base-uncased/'
        self.sentence_embedding = network.embedding.BERTSentenceEmbedding(pretrain_path=pretrain_path,
                                                                          max_length=self.max_length)
        self.vars = nn.ParameterList()
        self.n_way = n_way
        self.feature_dim = 768
        self.filter_num = 128
        self.type = type
        # kernel size = 2
        # [ch_out, ch_in, kernelsz, kernelsz]
        if type == "pcnnLinear":
            # CNN
            self.filter_sizes = [2, 3, 4, 5]
            for filter_size in self.filter_sizes:
                w = nn.Parameter(torch.ones(self.filter_num, 1, filter_size, self.feature_dim))  # [64,1,3,3]]
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(self.filter_num)))

            filter_dim = self.filter_num * len([2, 3, 4, 5])
            labels_num = self.n_way

            # dropout
            self.dropout = nn.Dropout(0.5)

            # linear
            w = nn.Parameter(torch.ones(128, filter_dim * 3))
            self.linear = nn.Linear(filter_dim * 3, 128)
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            # [ch_out]
            self.vars.append(nn.Parameter(torch.zeros(128)))

            w = nn.Parameter(torch.ones(labels_num, 128))
            self.linear = nn.Linear(128, labels_num)
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(labels_num)))

        elif self.type=='cnnLinear':
            # *************attention*****************
            # [N*K,MAXLEN,768]
            # w_omega = (torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            # u_omega = Variable(torch.zeros(self.attention_size))


            # kernel size = 2
            # [ch_out, ch_in, kernelsz, kernelsz]
            for filter_size in [2, 3, 4, 5]:
                w = nn.Parameter(torch.ones(self.filter_num, 1, filter_size, self.feature_dim),requires_grad=True)  # [64,1,3,3]]
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(self.filter_num),requires_grad=True))

            filter_dim = self.filter_num * len([2, 3, 4, 5])
            labels_num = self.n_way 

            # dropout
            self.dropout = nn.Dropout(0.5)

            # linear
            w = nn.Parameter(torch.ones(labels_num, filter_dim),requires_grad=True)
            self.linear = nn.Linear(filter_dim, labels_num)
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            # [ch_out]
            self.vars.append(nn.Parameter(torch.zeros(labels_num),requires_grad=True))

        # linear
        elif self.type == "concatLinear":
            w = nn.Parameter(torch.ones(self.n_way, 1536))
            self.linear = nn.Linear(1536, self.n_way)
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(self.n_way)))

        elif self.type == "clsLinear":
            w = nn.Parameter(torch.ones(self.n_way, 768))
            self.linear = nn.Linear(768, self.n_way)
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(self.n_way)))

        else:
            raise Exception("Learner type only can be cnnLinear、concatLinear、clsLinear")

    def forward(self, x, vars=None):
        '''
        :param x: [1,N,K,MAXLEN*3]
        :param vars:
        :return:
        '''
        if self.type == "pcnnLinear":
            return self.pcnnForward(x, vars)
        elif self.type == "cnnLinear":
            return self.cnnForward(x,vars)
        elif self.type == "concatLinear:":
            return self.concatForward((x, vars))
        elif self.type == "clsLinear":
            return self.clsForward(x, vars)

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

    def cnnForward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        with torch.no_grad():
            x = self.sentence_embedding(x)  # [N*K,MAXLEN,768]
        x = x.unsqueeze(dim=1)
        xs = []
        for _ in range(4):
            w, b = vars[idx], vars[idx + 1]
            x1 = F.conv2d(x, w, b)
            xs.append(x1.squeeze(3))
            idx += 2

        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in xs]  # x[idx]: batch_size x filter_num
        sentence_features = torch.cat(x, dim=1)  # batch_size x (filter_num * len(filters))
        x = self.dropout(sentence_features)

        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2

        # # make sure variable is used properly
        assert idx == len(vars)

        return x

    def pcnnForward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        with torch.no_grad():
            x, eh, et = self.sentence_embedding(x, return_entity=True)  # [N*K,MAXLEN,768]
            et, eh = np.maximum(eh, et), np.minimum(eh, et)
            eh = [6 if i < 6 else i for i in eh.numpy()]
            et = [self.max_length - 10 if i > self.max_length - 10 else i for i in et.numpy()]
        x = x.unsqueeze(dim=1)
        xs = []
        for _ in range(len(self.filter_sizes)):
            w, b = vars[idx], vars[idx + 1]
            x1 = F.conv2d(x, w, b)  # [B, filter_num, maxlen-filtersize+1, 1]
            x1 = x1.squeeze(3)  # [B, filter_num, maxlen-filtersize+1]
            idx += 2
            data = []
            for i, item in enumerate(x1):  # item [filter_num,maxlen-filtersize+1]
                head = item.permute(1, 0)[:eh[i]].permute(1, 0).unsqueeze(0)  # [1,filter_num,eh]
                if eh[i] - et[i] < 6:
                    mid = item.permute(1, 0)[eh[i]:eh[i] + 6].permute(1, 0).unsqueeze(0)
                else:
                    mid = item.permute(1, 0)[eh[i]:et[i]].permute(1, 0).unsqueeze(0)
                tail = item.permute(1, 0)[et[i]:].permute(1, 0).unsqueeze(0)
                try:
                    head = F.max_pool1d(head, kernel_size=head.size(2))
                    mid = F.max_pool1d(mid, kernel_size=mid.size(2))
                    tail = F.max_pool1d(tail, kernel_size=tail.size(2))  # [1,filter_num,1]
                except:
                    print("head's shape:", head.shape)
                    print("mid's shape:", mid.shape)
                    print("tail's shape:", tail.shape)
                    print("eh:", eh)
                    print("et:", et)
                data.append(torch.cat([head, mid, tail], dim=2).reshape(1, -1))  # [1,filter_num*3]
            xs.append(torch.cat(data, dim=0))  # data[idx]: [B, filter_num*3]
        sentence_features = torch.cat(xs, dim=1)  # batch_size x (filter_num * len(filters)*3)

        x = self.dropout(sentence_features)  # [B,filter_num * len(filters)*3]
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2

        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2

        # # make sure variable is usednvidi properly
        assert idx == len(vars)
        return x

    def concatForward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        with torch.no_grad():
            x = self.sentence_embedding(x)  # [N*K,1536]

        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2

        # # make sure variable is used properly
        assert idx == len(vars)

        return x

    def clsForward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        with torch.no_grad():
            x = self.sentence_embedding(x)  # [N*K,MAXLEN,768]
        x = x.permute(1, 0, 2)[0]  # [N*K, 768]
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2
        # # make sure variable is used properly
        assert idx == len(vars)

        return x

class Learner():
    pass

# from transformers.modeling_bert import BertPreTrainedModel
#
#
# class Learner(BertPreTrainedModel):
#
#     def __init__(self, config):
#         super(Learner, self).__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
#
#         self.init_weights()
#
#     def forward(self, x, vars = None):
#         # if vars is None:
#         #     vars = self.vars
#         # x = x.view(-1, self.max_length * 2)
#         word, mask = x.chunk(2, 1)  # [5,5,128]
#         inputs = {'word': word, 'mask': mask}
#         del word,mask
#         outputs ,pooled_output = self.bert(inputs['word'], attention_mask=inputs['mask']) # x [N*K,MAXLEN,768]
#
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         # x = x.permute(1,0,2)[0] # x [N*K, 768]
#
#
#         return logits
