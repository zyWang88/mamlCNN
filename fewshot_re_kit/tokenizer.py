import numpy as np
import math
from transformers import BertTokenizer
class GloveTokenizer():

    def __init__(self, word2id, max_length):
        self.max_length = max_length
        self.word2id = word2id

    def glove_tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index_start = min(self.max_length, pos_head[0])
        pos1_in_index_end = min(self.max_length, pos_head[-1])
        pos2_in_index_start = min(self.max_length, pos_tail[0])
        pos2_in_index_end = min(self.max_length, pos_tail[-1])

        for i in range(self.max_length):
            pos1[i] =min(abs(i - pos1_in_index_start),abs(i - pos1_in_index_end))
            pos2[i] = min(abs(i - pos2_in_index_start),abs(i - pos2_in_index_end))

        return indexed_tokens, pos1, pos2


class Berttokenizer():
    def __init__(self, max_length):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def bert_tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask


if __name__ == '__main__':
