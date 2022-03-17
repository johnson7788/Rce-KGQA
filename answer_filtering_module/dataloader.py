import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MetaQADataSet(Dataset):
    def __init__(self, entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path, qa_dataset_path,
                 split):
        """
            create MetaQADataSet
        :param entity_embed_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/E.npy'
        :param entity_dict_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/entities_idx.dict'
        :param relation_embed_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/R.npy'
        :param relation_dict_path:  '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/relations_idx.dict'
        :param qa_dataset_path: '../QA/MetaQA/qa_train_1hop.txt'
        :param split:
        """
        # ====加载实体和关系的嵌入====
        self.entity_embeddings = np.load(entity_embed_path)
        self.relation_embeddings = np.load(relation_embed_path)
        # ====load entity & relation dict (mapping word into index)====
        self.entities2idx = dict()   #所有实体到id的映射
        self.relations2idx = dict()  #所有关系到id的映射
        with open(entity_dict_path, 'rt', encoding='utf-8') as e_d, open(relation_dict_path, 'rt',
                                                                         encoding='utf-8') as r_d:
            for line in e_d:
                mapping = line.strip().split('\t')
                self.entities2idx[mapping[0]] = int(mapping[1])
            for line in r_d:
                mapping = line.strip().split('\t')
                self.relations2idx[mapping[0]] = int(mapping[1])
        self.entities_count = len(self.entities2idx)  #实体数量43234
        self.relations_count = len(self.relations2idx)  #关系数量18

        # ===加载问答对====
        self.results = []
        with open(qa_dataset_path, 'rt', encoding='utf-8') as inp_:
            for line in inp_.readlines():
                try:
                    line = line.strip().split('\t')
                    question = line[0]   #'what movies are about [ginger rogers]'
                    q_temp = question.split('[')
                    topic_entity = q_temp[1].split(']')[0]   #主题实体： 'ginger rogers'
                    question = q_temp[0] + 'NE' + q_temp[1].split(']')[1]   # 'what movies are about NE'， 为什么加上NE
                    answers = [i.strip() for i in line[1].split('|')]  #答案: ['Top Hat', 'Kitty Foyle', 'The Barkleys of Broadway']
                    self.results.append([topic_entity.strip(), question.strip(), answers])
                except RuntimeError:
                    continue
        assert len(self.results) > 0, f'读取问答对失败 [{qa_dataset_path}]'
        if split:
            split_result = []
            for qa_pair in self.results:
                for answer in qa_pair[2]:
                    split_result.append([qa_pair[0], qa_pair[1], answer])
            self.results = split_result
        # ====get word <=> idx mapping====
        self.idx_word = dict()   # 单词到id的映射， {'what': 0, 'movies': 1, 'are': 2}
        self.word_idx = dict()   #id到单词的映射
        self.max_sent_length = 0
        for qa_pair in self.results:  # retrieval all questions
            words = qa_pair[1].split()   #['what', 'movies', 'are', 'about', 'NE']
            if len(words) > self.max_sent_length:
                self.max_sent_length = len(words)
            for word in words:
                if word not in self.word_idx:
                    self.word_idx[word] = len(self.word_idx)
                    self.idx_word[len(self.idx_word)] = word

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        qa_pair = self.results[index]
        # ==head entity text==
        text_head = qa_pair[0]
        head_idx = self.entities2idx[text_head]
        # ==text question==
        text_q = qa_pair[1]
        idx_q = [self.word_idx[word] for word in text_q.split()]
        # ==tail entity text==
        text_tails = qa_pair[2]
        idx_tails = [self.entities2idx[tail_text] for tail_text in text_tails]
        onehot_tail = torch.zeros(self.entities_count)
        onehot_tail.scatter_(0, torch.tensor(idx_tails), 1)
        return idx_q, head_idx, onehot_tail


class MetaQADataLoader(DataLoader):
    def __init__(self, entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path, qa_dataset_path,
                 batch_size, split=False, shuffle=True):
        """

        :param entity_embed_path:  '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/E.npy'
        :type entity_embed_path:
        :param entity_dict_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/entities_idx.dict'
        :type entity_dict_path:
        :param relation_embed_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/R.npy'
        :type relation_embed_path:
        :param relation_dict_path: '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/relations_idx.dict'
        :type relation_dict_path:
        :param qa_dataset_path:  '../QA/MetaQA/qa_train_1hop.txt'
        :type qa_dataset_path:
        :param batch_size: 128
        :type batch_size:
        :param split: False
        :type split:
        :param shuffle: True
        :type shuffle:
        """
        dataset = MetaQADataSet(entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path,
                                qa_dataset_path, split)
        super(MetaQADataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch_data):
        """
        :param batch_data: dataset __getitem__ outputs.
        :return: batch_questions_index, batch_questions_length, batch_head_entity, batch_onehot_answers, max_sent_len
        """
        sorted_qa_pairs = list(sorted(batch_data, key=lambda x: len(x[0]), reverse=True))
        sorted_qa_pairs_len = [len(qa_pair[0]) for qa_pair in sorted_qa_pairs]
        max_sent_len = len(sorted_qa_pairs[0][0])
        padded_questions = []  # torch.zeros(batch_size, max_sent_len, dtype=torch.long)
        head_idxs = []
        onehot_tails = []
        for idx_q, head_idx, onehot_tail in sorted_qa_pairs:
            padded_questions.append(idx_q + [0] * (max_sent_len - len(idx_q)))
            head_idxs.append(head_idx)
            onehot_tails.append(onehot_tail)
        return torch.tensor(padded_questions, dtype=torch.long), torch.tensor(sorted_qa_pairs_len, dtype=torch.long), \
               torch.tensor(head_idxs), torch.tensor(onehot_tails), max_sent_len


# ====test and dev dataloader====

class DEV_MetaQADataSet(Dataset):
    def __init__(self, word_idx, entity_dict_path, relation_dict_path, qa_dataset_path, split):
        """
            create MetaQADataSet
        :param entity_dict_path: filepath for mapping entity to index
        :param relation_dict_path:  filepath for mapping relation to index
        :param qa_dataset_path:
        :param split:
        """
        # ====load entity & relation embeddings====
        # ====load entity & relation dict (mapping word into index)====
        self.word_idx = word_idx
        self.entities2idx = dict()
        self.relations2idx = dict()
        with open(entity_dict_path, 'rt', encoding='utf-8') as e_d, open(relation_dict_path, 'rt',
                                                                         encoding='utf-8') as r_d:
            for line in e_d:
                mapping = line.strip().split('\t')
                self.entities2idx[mapping[0]] = int(mapping[1])
            for line in r_d:
                mapping = line.strip().split('\t')
                self.relations2idx[mapping[0]] = int(mapping[1])
        self.entities_count = len(self.entities2idx)
        self.relations_count = len(self.relations2idx)

        # ====load QA pairs====
        self.results = []
        with open(qa_dataset_path, 'rt', encoding='utf-8') as inp_:
            for line in inp_.readlines():
                try:
                    line = line.strip().split('\t')
                    question = line[0]
                    q_temp = question.split('[')
                    topic_entity = q_temp[1].split(']')[0]
                    question = q_temp[0] + 'NE' + q_temp[1].split(']')[1]
                    answers = [i.strip() for i in line[1].split('|')]
                    self.results.append([topic_entity.strip(), question.strip(), answers])
                except RuntimeError:
                    continue
        assert len(self.results) > 0, f'read no qa-pairs in file [{qa_dataset_path}]'
        if split:
            split_result = []
            for qa_pair in self.results:
                for answer in qa_pair[2]:
                    split_result.append([qa_pair[0], qa_pair[1], answer])
            self.results = split_result

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        qa_pair = self.results[index]
        # ==head entity text==
        text_head = qa_pair[0]
        heads_idx = self.entities2idx[text_head]
        # ==text question==
        text_q = qa_pair[1]
        idx_q = [self.word_idx[word] for word in text_q.split()]
        # ==tail entity text==
        text_tails = qa_pair[2]
        tails_idx = [self.entities2idx[tail_text] for tail_text in text_tails]
        return idx_q, heads_idx, tails_idx


class DEV_MetaQADataLoader(DataLoader):
    def __init__(self, word_idx, entity_dict_path, relation_dict_path, qa_dataset_path, split=False, batch_size=128, shuffle=True):
        dataset = DEV_MetaQADataSet(word_idx, entity_dict_path, relation_dict_path, qa_dataset_path, split)
        super(DEV_MetaQADataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch_data):
        """
        :param batch_data: dataset __getitem__ outputs.
        :return: batch_questions_index, batch_questions_length, batch_head_entity, batch_onehot_answers, max_sent_len
        """
        sorted_qa_pairs = list(sorted(batch_data, key=lambda x: len(x[0]), reverse=True))
        sorted_qa_pairs_len = [len(qa_pair[0]) for qa_pair in sorted_qa_pairs]
        max_sent_len = len(sorted_qa_pairs[0][0])
        padded_questions = []  # torch.zeros(batch_size, max_sent_len, dtype=torch.long)
        heads_idx = []
        tails_idxs = []
        for idx_q, head_idx, tails_idx in sorted_qa_pairs:
            padded_questions.append(idx_q + [0] * (max_sent_len - len(idx_q)))
            heads_idx.append(head_idx)
            tails_idxs.append(tails_idx)
        return torch.tensor(padded_questions, dtype=torch.long), torch.tensor(sorted_qa_pairs_len, dtype=torch.long), \
               torch.tensor(heads_idx), torch.tensor(tails_idxs), max_sent_len
