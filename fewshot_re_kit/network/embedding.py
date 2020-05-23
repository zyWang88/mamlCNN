import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
import numpy as np

class GloveEmbedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        # Word embedding
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        # word = inputs['word']
        # pos1 = inputs['pos1']
        # pos2 = inputs['pos2']
        '''

        :param inputs:  [N,K,MAXLEN*3]
        :return:
        '''
        inputs = inputs.view(-1, self.max_length * 3)
        word, pos1, pos2 = inputs.chunk(3, 1)  # [5,5,128]
        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)
        return x


class BERTSentenceEmbedding(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert.eval()
        self.max_length = max_length

    def forward(self, inputs, concat=False, return_entity=False):
        inputs = inputs.view(-1, self.max_length * 2)

        word, mask = inputs.chunk(2, 1)  # [5,5,128]
        eh = np.argmax((word == 1).cpu(),1)
        et = np.argmax((word == 2).cpu(),1)
        inputs = {'word': word, 'mask': mask}
        net, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        # concat
        if concat:
            return torch.cat([net.permute(1, 0, 2)[eh], net.permute(1, 0, 2)[et]], dim=1)
        if return_entity:
            return net,eh,et
        else:
            return net



class ALBERTSentenceEmbedding(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = AlbertModel.from_pretrained(pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()
        self.max_length = max_length

    def forward(self, inputs,concat=False):
        inputs = inputs.view(-1, self.max_length * 2)

        word, mask = inputs.chunk(2, 1)  # [5,5,128]
        inputs = {'word': word, 'mask': mask}
        net, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return net