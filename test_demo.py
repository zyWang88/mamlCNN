import json
import torch
from fewshot_re_kit.tokenizer import Berttokenizer
import random
from meta import  Meta
import argparse
import pdb
import os
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)

    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.01)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    parser.add_argument('--embedding', default='bert', type=str, help='"glove" or "bert".')

    parser.add_argument('--gpu', default="0", type=str, help='gpu use.')
    parser.add_argument('--type', default="cnnLinear", type=str, help="type of the net, 'cnnLinear' 'concatLinear' or 'clsLinear'.")
    parser.add_argument('--filename', default=None, type=str, help="type of the net, 'cnnLinear' 'concatLinear' or 'clsLinear'.")
    parser.add_argument('--modelfile',default=None,type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = Berttokenizer(max_length=args.max_length)
    content = json.load(open("./input.json"))
    model_path = args.modelfile
    model = torch.load(model_path)
    maml = Meta(args)  # 网络层

    preds = []
    for id,i in enumerate(content):
        # print(id)
        N = len(i['meta_train'])
        K = len(i['meta_train'][0])
        Q = 1
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
        if torch.cuda.is_available():
            x_spt = x_spt.cuda()
            x_qry = x_qry.cuda()
            y_spt = y_spt.cuda()
            maml = maml.cuda()
        pred_q = maml.evaluate(x_spt,y_spt,x_qry)
        preds.append(int(pred_q.cpu().numpy()[0]))
    json.dump(preds, sys.stdout)
