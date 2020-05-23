import json
import torch
from fewshot_re_kit.tokenizer import Berttokenizer
import sys
from meta import Meta
input_filename = sys.argv[1]
tokenizer = Berttokenizer(max_length=128)
content = json.load(open(input_filename))

N = len(content[0]['meta_train'])
K = len(content[0]['meta_train'][0])
Q = 1
model_path = 'model/{}way{}shot.ckpt'.format(N,K)

maml = torch.load(model_path)


preds = []
for id,i in enumerate(content):

    support_set = []
    # train
    for label, items in enumerate(i['meta_train']):  #[5*5]
        for item in items:
            word, mask = tokenizer.bert_tokenize(item['tokens'],
                                                     item['h'][2][0],
                                                     item['t'][2][0])
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            support_set.append([label, torch.cat([word,mask])])
    # random.shuffle(support_set)
    x_spt = torch.cat([data[1].unsqueeze(0) for data in support_set])
    y_spt = torch.tensor([data[0] for data in support_set])
    word, mask = tokenizer.bert_tokenize(i['meta_test']['tokens'],
                                         i['meta_test']['h'][2][0],
                                         i['meta_test']['t'][2][0])
    word = torch.tensor(word).long()
    mask = torch.tensor(mask).long()
    x_qry = torch.cat([word,mask])

    #finetune
    # if torch.cuda.is_available():
    x_spt = x_spt.cuda()
    x_qry = x_qry.cuda()
    y_spt = y_spt.cuda()
    # maml = maml.cuda()
    pred_q = maml.evaluate(x_spt,y_spt,x_qry)
    preds.append(int(pred_q.cpu().numpy()[0]))
json.dump(preds, sys.stdout)
