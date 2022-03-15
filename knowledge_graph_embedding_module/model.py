import torch


class ComplEx_KGE(torch.nn.Module):
    def __init__(self, d, entity_dim, do_batch_norm, input_dropout, hidden_dropout1, hidden_dropout2):
        """

        :param d:  数据集class DataSet的实例化
        :type d:
        :param entity_dim:
        :type entity_dim:
        :param do_batch_norm: bool:  True
        :type do_batch_norm:
        :param input_dropout:  0.3
        :type input_dropout:
        :param hidden_dropout1: 0.4
        :type hidden_dropout1:
        :param hidden_dropout2:  0.5
        :type hidden_dropout2:
        """
        super(ComplEx_KGE, self).__init__()
        # 实体和关系的embedding
        self.E = torch.nn.Embedding(len(d.entities), entity_dim * 2, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), entity_dim * 2, padding_idx=0)
        # 初始化权重
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)
        self.entity_dim = entity_dim * 2
        self.do_batch_norm = do_batch_norm
        self.dropout0_layer = torch.nn.Dropout(input_dropout)
        self.dropout1_layer = torch.nn.Dropout(hidden_dropout1)
        self.dropout2_layer = torch.nn.Dropout(hidden_dropout2)
        self.head_bn = torch.nn.BatchNorm1d(2)
        self.score_bn = torch.nn.BatchNorm1d(2)
        self.bce_loss = torch.nn.BCELoss()
        print(f"模型的结构如下:")
        print(self)

    def freeze_entity_embeddings(self):
        self.E.weight.requires_grad = False

    def complex_scorer(self, head, relation):
        """

        :param head:
        :type head:
        :param relation:
        :type relation:
        :return:
        :rtype:
        """
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.head_bn(head)
        head = self.dropout0_layer(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        relation = self.dropout1_layer(relation)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.E.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.score_bn(score)
        score = self.dropout2_layer(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        answers_logit = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        answers_score = torch.sigmoid(answers_logit)
        return answers_score

    def forward(self, h_idx, r_idx, targets):
        """

        :param h_idx:
        :type h_idx:
        :param r_idx:
        :type r_idx:
        :param targets:
        :type targets:
        :return:
        :rtype:
        """
        h = self.E(h_idx.long())
        r = self.R(r_idx.long())
        answers_score = self.complex_scorer(h, r)
        loss = self.bce_loss(answers_score, targets)
        return loss

    def get_scores(self, h_idx, r_idx):
        """

        :param h_idx:
        :type h_idx:
        :param r_idx:
        :type r_idx:
        :return:
        :rtype:
        """
        return self.complex_scorer(self.E(h_idx.long()), self.R(r_idx.long()))
