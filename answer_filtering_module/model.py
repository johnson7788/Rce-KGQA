import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import Attention_layer
import numpy as np


class Answer_filtering_module(torch.nn.Module):
    def __init__(self, entity_embeddings, embedding_dim, vocab_size, word_dim, hidden_dim, fc_hidden_dim, relation_dim,
                 head_bn_filepath, score_bn_filepath):
        """

        :param entity_embeddings: (43234, 400)
        :type entity_embeddings:
        :param embedding_dim: 400
        :type embedding_dim:
        :param vocab_size: 117
        :type vocab_size:
        :param word_dim: 256
        :type word_dim:
        :param hidden_dim: 200
        :type hidden_dim:
        :param fc_hidden_dim: 400
        :type fc_hidden_dim:
        :param relation_dim: 400
        :type relation_dim:
        :param head_bn_filepath: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/head_bn.npy'
        :type head_bn_filepath:
        :param score_bn_filepath: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/score_bn.npy'
        :type score_bn_filepath:
        """
        super(Answer_filtering_module, self).__init__()
        self.relation_dim = relation_dim * 2
        self.loss_criterion = torch.nn.BCELoss(reduction='sum')
        # hidden_dim * 2 is the BiLSTM + attention layer output
        self.fc_lstm2hidden = torch.nn.Linear(hidden_dim * 2, fc_hidden_dim, bias=True)  #Linear(in_features=400, out_features=400, bias=True)
        torch.nn.init.xavier_normal_(self.fc_lstm2hidden.weight.data)
        torch.nn.init.constant_(self.fc_lstm2hidden.bias.data, val=0.0)
        self.fc_hidden2relation = torch.nn.Linear(fc_hidden_dim, self.relation_dim, bias=False)  #Linear(in_features=400, out_features=800, bias=False)
        torch.nn.init.xavier_normal_(self.fc_hidden2relation.weight.data)
        self.entity_embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(entity_embeddings), freeze=True)  #Embedding(43234, 400)
        self.word_embedding_layer = torch.nn.Embedding(vocab_size, word_dim) #Embedding(117, 256)
        self.BiLSTM = torch.nn.LSTM(embedding_dim, hidden_dim, 1, bidirectional=True, batch_first=True)  #LSTM(400, 200, batch_first=True, bidirectional=True)
        self.softmax_layer = torch.nn.LogSoftmax(dim=-1)
        self.attention_layer = Attention_layer(hidden_dim=2 * hidden_dim, attention_dim=4 * hidden_dim)
        self.head_bn = torch.nn.BatchNorm1d(2)
        # 加载head和score2个层的参数
        head_bn_params_dict = np.load(head_bn_filepath, allow_pickle=True)
        for key in head_bn_params_dict.item():
            # 把weight，bias，running_mean，running_var分别进行tensor
            if key == "weight":
                self.head_bn.weight.data = torch.tensor(head_bn_params_dict.item()[key])
            elif key == "bias":
                self.head_bn.bias.data = torch.tensor(head_bn_params_dict.item()[key])
            elif key == "running_mean":
                self.head_bn.running_mean.data = torch.tensor(head_bn_params_dict.item()[key])
            elif key == "running_var":
                self.head_bn.running_var.data = torch.tensor(head_bn_params_dict.item()[key])
            else:
                raise Exception(f"未找到对应的参数名称: {key}")
        self.score_bn = torch.nn.BatchNorm1d(2)
        score_bn_params_dict = np.load(score_bn_filepath, allow_pickle=True)
        for key in score_bn_params_dict.item():
            # 把weight，bias，running_mean，running_var分别进行tensor
            if key == "weight":
                self.score_bn.weight.data = torch.tensor(score_bn_params_dict.item()[key])
            elif key == "bias":
                self.score_bn.bias.data = torch.tensor(score_bn_params_dict.item()[key])
            elif key == "running_mean":
                self.score_bn.running_mean.data = torch.tensor(score_bn_params_dict.item()[key])
            elif key == "running_var":
                self.score_bn.running_var.data = torch.tensor(score_bn_params_dict.item()[key])
            else:
                raise Exception(f"未找到对应的参数名称: {key}")
    def complex_scorer(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.head_bn(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embedding_layer.weight, 2, dim=1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = torch.stack([re_score, im_score], dim=1)
        score = self.score_bn(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        return torch.sigmoid(torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0)))

    def forward(self, question, questions_length, head_entity, tail_entity, max_sent_len):
        """

        :param question:  问题id
        :type question: torch.Size([128, 9])
        :param questions_length:  每个问题的长度
        :type questions_length: 128
        :param head_entity:
        :type head_entity: 128
        :param tail_entity: 答案
        :type tail_entity: torch.Size([128, 43234])
        :param max_sent_len: 最大问题的长度
        :type max_sent_len: 9
        :return:
        :rtype:
        """
        embedded_question = self.word_embedding_layer(question.unsqueeze(0))  #torch.Size([1, 128, 9, 256])
        packed_input = pack_padded_sequence(embedded_question, questions_length, batch_first=True)
        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)
        loss = self.loss_criterion(pred_answers_score, tail_entity)
        return loss

    def get_ranked_top_k(self, question, questions_length, head_entity, max_sent_len, K=5):
        embedded_question = self.word_embedding_layer(question.unsqueeze(0))
        packed_input = pack_padded_sequence(embedded_question, questions_length, batch_first=True)
        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)
        return torch.topk(pred_answers_score, k=K, dim=-1, largest=True, sorted=True)
