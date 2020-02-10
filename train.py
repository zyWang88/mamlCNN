import torch
import numpy as np
import argparse
from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.tokenizer import Tokenizer
import json
from meta import Meta




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
                        help='train file')
    parser.add_argument('--val', default='val_wiki',
                        help='val file')

    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)

    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.1)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--epoch', type=int, help='epoch number', default=20)

    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')

    args = parser.parse_args()
    try:
        glove_mat = np.load('./pretrain/glove/glove_mat.npy')
        glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
    tokenizer = Tokenizer(
        glove_word2id,
        args.max_length)
    train_data_loader = get_loader(args.train, tokenizer,
                                   N=args.n_way, K=args.k_spt, Q=args.k_qry, na_rate=args.na_rate, batch_size=args.task_num)
    val_data_loader = get_loader(args.val, tokenizer,
                    N=args.n_way, K=args.k_spt, Q=args.k_qry, batch_size=args.task_num)

    maml = Meta(args, glove_mat, glove_word2id)  # 网络层
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    del glove_word2id, glove_mat

    for step in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = next(train_data_loader)
        # x_spt, x_qry, y = reshape(x_spt, x_qry, y, N, K, max_length)

        if torch.cuda.is_available():
            x_spt = x_spt.cuda()
            x_qry = x_qry.cuda()
            y_spt = y_spt.cuda()
            y_qry = y_qry.cuda()
            maml.cuda()
        accs = maml(x_spt, y_spt, x_qry, y_qry )

        # if step % 50 == 0:
        print('step:', step, '\ttraining acc:', accs)

        if step % 2 == 0:
            accs = []
            for _ in range(100 // args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = next(val_data_loader)
                if torch.cuda.is_available():
                    x_spt = x_spt.cuda()
                    x_qry = x_qry.cuda()
                    y_spt = y_spt.cuda()
                    y_qry = y_qry.cuda()
                    maml.cuda()
                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry ):  #[N,K,MAXLEN]
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)


if __name__ == '__main__':
    main()
