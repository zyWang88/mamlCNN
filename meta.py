import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import pdb
from copy import deepcopy
from learner import BertLearner, Learner
import logging

logger = logging.getLogger(__name__)


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, glove_mat=None, glove_word2id=None):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        if args.embedding == 'glove':
            print("loading glove embedding")
            self.net = Learner(glove_mat, glove_word2id, args)  # 弃用
        elif args.embedding == 'bert':
            print("loading bert embedding")
            self.net = BertLearner(args.max_length, args.n_way, args.type)
        self.count = 0
        # else:
        #     print("loading albert embedding")
        #     self.net = AlbertLearner(args.max_length,args.n_way)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)  # 换成SGD

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:    [B,N,K,MAXLEN]
        :param y_spt:
        :param x_qry:
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # if (not_print == 0 and i == task_num - 1):
            #     # pdb.set_trace()
            #     print("grad*self.update_lr:", (grad[0] * self.update_lr)[0][:100])
            #     print("vars[0]", self.net.vars[0][0][:100])
            #     self.count += 1
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters())
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights)
                loss = F.cross_entropy(logits, y_spt[i].long())
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # torch.nn.utils.clip_grad_value_(self.net.parameters(),)
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs, loss_q.item()

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [N, K, MAXLEN*3]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 2
        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.long())
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters())
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights)
            loss = F.cross_entropy(logits, y_spt.long())
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry.long())

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net
        accs = np.array(corrects) / querysz
        return accs

    def evaluate(self, x_spt, y_spt, x_qry):
        """

        :param x_spt:   [N, K, MAXLEN*3]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 2
        # corrects = [0 for _ in range(self.update_step_test + 1)]
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        logits_qs = 0
        for _ in range(3):
            net = deepcopy(self.net)
            net.eval()
            # 1. run the i-th task and compute loss for k=0
            logits = net(x_spt)
            loss = F.cross_entropy(logits, y_spt.long())
            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights)
                loss = F.cross_entropy(logits, y_spt.long())
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = net(x_qry, fast_weights)
                logits_qs += F.softmax(logits_q, dim=1)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = F.cross_entropy(logits_q, y_qry.long())
        del net
        with torch.no_grad():
            pred_q = logits_qs.argmax(dim=1)
        return pred_q
