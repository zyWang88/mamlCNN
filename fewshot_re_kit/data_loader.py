import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2 = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2

    def __additem__(self, d, word, pos1, pos2,label):
        # d['word'].append(word)
        # d['pos1'].append(pos1)
        # d['pos2'].append(pos2)
        # d['label'].append(label)
        #
        d['feature'].append(torch.cat([word,pos1,pos2]))
        d['label'].append(torch.tensor(label))

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)  #随机选取N个类
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'label': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'label': []}
        support_set = []
        query_set = []
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                if count < self.K:
                    support_set.append([i,torch.cat([word,pos1,pos2])])
                else:
                    query_set.append([i,torch.cat([word,pos1,pos2])])
                count += 1

        x_spt = [data[1] for i, data in enumerate(support_set)]
        y_spt = [i for i in range(self.N)] * self.K
        x_qry = [data[1] for i, data in enumerate(query_set)]
        y_qry = [i for i in range(self.N)] * self.Q
        a = []
        for i in range(self.K):
            a.extend(x_spt[i:len(support_set):self.K])
        b = []
        for i in range(self.Q):
            b.extend(x_qry[i:len(query_set):self.Q])

        #return x_spt [N*K,MAXLEN*3]
        #       y_spt [N*K]
        #       x_qry [N*Q,MAXLEN*3]
        #       y_qry [N*K]
        return torch.stack(a), torch.tensor(y_spt), torch.stack(b), torch.tensor(y_qry)

    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'feature':[], 'label':[]}
    batch_query = {'feature': [], 'label':[] }
    # batch_label = []
    support_sets, query_sets = zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    return batch_support, batch_query

def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers)
            # collate_fn=collate_fn)
    return iter(data_loader)


if __name__ == '__main__':
    pass