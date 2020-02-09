import numpy as np
import math
class Tokenizer():

    def __init__(self, word2id, max_length):
        self.max_length = max_length
        self.word2id = word2id

    def tokenize(self, raw_tokens, pos_head, pos_tail):
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
        #
        # for i in range(self.max_length):
        #     pos1[i] = i - pos1_in_index + self.max_length
        #     pos2[i] = i - pos2_in_index + self.max_length

        # mask
        # mask = np.zeros((self.max_length), dtype=np.int32)
        # mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2


