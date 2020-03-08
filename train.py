import torch
import numpy as np
import argparse
from fewshot_re_kit.data_loader import glove_getloader, bert_getloader
from fewshot_re_kit.tokenizer import Berttokenizer, GloveTokenizer, Alberttokenizer
import json
from meta import Meta
import time
import os
from datetime import datetime
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train', help='train file')
    parser.add_argument('--val', default='val', help='val file')
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--max_length', default=64, type=int, help='max length')
    parser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--embedding', default='bert', type=str, help='"glove" or "bert".')
    parser.add_argument('--gpu', default="0", type=str, help='gpu use.')
    parser.add_argument('--type', default="cnnLinear", type=str,
                        help="type of the net, 'cnnLinear' 'concatLinear' or 'clsLinear'.")
    parser.add_argument('--filename', default=None, type=str,
                        help="type of the net, 'cnnLinear' 'concatLinear' or 'clsLinear'.")
    args = parser.parse_args()
    print(str(args))
    if args.filename == None:
        file_name = 'log/{}way{}shot-{}-{}'.format(args.n_way, args.k_spt, args.embedding, args.type)
        dt = datetime.now()
        file_name += dt.strftime('%Y-%m-%d-%H:%M:%S-%f')
        file_name += ".log"
    else:
        file_name = os.path.join('log', args.filename)
    with open(file_name, 'w') as f:
        f.writelines(str(args).split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.embedding == 'glove':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        tokenizer = GloveTokenizer(
            glove_word2id,
            args.max_length)
        maml = Meta(args, glove_mat, glove_word2id)  # 网络层
        del glove_word2id, glove_mat
        train_data_loader = glove_getloader(args.train, tokenizer, N=args.n_way, K=args.k_spt, Q=args.k_qry,
                                            na_rate=args.na_rate, batch_size=args.task_num)
        val_data_loader = glove_getloader(args.val, tokenizer, N=args.n_way, K=args.k_spt, Q=1,
                                          batch_size=20)
    elif args.embedding == 'bert':
        tokenizer = Berttokenizer(max_length=args.max_length)
        maml = Meta(args)  # 网络层
        train_data_loader = bert_getloader(args.train, tokenizer, N=args.n_way, K=args.k_spt, Q=args.k_qry,
                                           na_rate=args.na_rate, batch_size=args.task_num)
        val_data_loader = bert_getloader(args.val, tokenizer, N=args.n_way, K=args.k_spt, Q=1,
                                         batch_size=20)
    elif args.embedding == 'albert':
        tokenizer = Alberttokenizer(max_length=args.max_length)
        maml = Meta(args)  # 网络层
        train_data_loader = bert_getloader(args.train, tokenizer, N=args.n_way, K=args.k_spt, Q=args.k_qry,
                                           na_rate=args.na_rate, batch_size=args.task_num)
        val_data_loader = bert_getloader(args.val, tokenizer, N=args.n_way, K=args.k_spt, Q=1,
                                         batch_size=20)
    else:
        raise Exception("embedding error.")

    accses_train = []
    accses_test = []
    losses = []
    best_result = 0
    # print(maml.parameters())
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(tmp)
    print('Total trainable tensors:', num)
    start = time.time()
    # f = open(file_name,'w')
    # f.close()

    if torch.cuda.is_available():
        # maml = nn.DataParallel(maml)
        maml = maml.cuda()
    for step in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = next(train_data_loader)
        if torch.cuda.is_available():
            x_spt = x_spt.cuda()
            x_qry = x_qry.cuda()
            y_spt = y_spt.cuda()
            y_qry = y_qry.cuda()

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
        losses.append(loss)
        accses_train.append(accs)
        if step % 10 == 0:
            print('step:', step, '\ttraining acc:', accs, '\tloss:', loss, '\tcost', (time.time() - start) // 60, 'min')
            with open(file_name, 'a') as f:
                f.write("\nstep: {}\ttraining acc:{}\tloss:{}\tcost:{}min".format(step, accs, loss,
                                                                                  (time.time() - start) // 60))
        if step % 100 == 0:
            l = []
            for _ in range(10):
                accs = []
                x_spt, y_spt, x_qry, y_qry = next(val_data_loader)
                if torch.cuda.is_available():
                    x_spt = x_spt.cuda()
                    x_qry = x_qry.cuda()
                    y_spt = y_spt.cuda()
                    # y_qry = y_qry.cuda()
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):  # [N,K,MAXLEN]
                    # print(x_spt_one.shape)
                    # print(y_spt_one.shape)
                    # print(x_qry_one.shape)
                    # print(y_qry_one.shape)
                    # test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    # accs.append(test_acc)
                    pred = maml.evaluate(x_spt_one, y_spt_one, x_qry_one).cpu().numpy()
                    acc = (y_qry_one.numpy() == pred).mean()
                    accs.append(acc)
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                l.append(accs)
            with open(file_name, 'a') as f:
                f.write("\nTest acc:{}\tmean:{}\tcost:{}min".format(l, str(np.array(l).mean()),(time.time() - start) // 60))
            print('Test acc:', l, '\tmean:', np.array(l).mean(), '\tcost,', (time.time() - start) // 60, 'min\n')
            if best_result <= np.array(l).mean():
                torch.save(maml, "{}best.ckpt".format(file_name))
            accses_test.append([step, accs])


if __name__ == '__main__':
    main()
