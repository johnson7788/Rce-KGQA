from collections import defaultdict
from model import ComplEx_KGE
import numpy as np
import os
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class DataSet:
    def __init__(self, data_dir, reverse):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        # 实体和关系的字典
        self.entities = sorted(list(set([d[0] for d in self.data] + [d[2] for d in self.data])))
        self.relations = sorted(list(set([d[1] for d in self.data])))

    @staticmethod
    def load_data(data_dir, data_type, reverse):
        """
        加载数据集
        :param data_dir: './knowledge_graphs/MetaQA_half'
        :type data_dir:
        :param data_type: 'train'
        :type data_type:
        :param reverse: 是否把关系到过来在一次加入数据集
        :type reverse: True
        :return:
        :rtype:
        """
        with open(f"{data_dir}/{data_type}.txt", 'r', encoding='utf-8') as inp:
            data = [line.strip().split('\t') for line in inp.readlines()]
            if reverse:
                # 关系三元组的倒着排
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data


def set_fixed_seed(seed):
    if not torch.cuda.is_available():
        print('需要cuda环境')
        exit(-1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Experiment:
    def __init__(self, input_dropout, hidden_dropout1, hidden_dropout2, learning_rate, ent_vec_dim, rel_vec_dim,
                 num_epochs, test_interval, batch_size, decay_rate, label_smoothing, dataset_name, data_dir,
                 model_dir, load_from, do_batch_norm):
        # =======hyper-parameters===========
        self.input_dropout = input_dropout
        self.hidden_dropout1 = hidden_dropout1
        self.hidden_dropout2 = hidden_dropout2
        self.do_batch_norm = do_batch_norm  # bool，是否批归一化
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim  # 200
        self.rel_vec_dim = rel_vec_dim  # 200
        self.num_epochs = num_epochs  #500
        self.test_interval = test_interval  # 10
        self.batch_size = batch_size
        self.decay_rate = decay_rate #1.0
        self.label_smoothing = label_smoothing  #0.1
        # ========dataset stored and loaded==========
        self.dataset_name = dataset_name  #'FB15k-237'
        # ====KG sources====
        # before you run next statement, you should make knowledge_graphs available in root directory.
        data_path = os.path.join(data_dir, self.dataset_name)
        assert os.path.isdir(data_path), f"数据目录{data_path}不存在，请检查"
        self.dataset = DataSet(data_dir=data_path, reverse=True)
        self.entity_idxs = {entity: i for i, entity in enumerate(self.dataset.entities)}
        self.relation_idxs = {relation: i for i, relation in enumerate(self.dataset.relations)}
        # ====KGE model save=====
        model_save_dir = os.path.join(model_dir, self.dataset_name)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        self.best_model_save_dir = os.path.join(model_save_dir, 'best_checkpoint')
        if not os.path.isdir(self.best_model_save_dir):
            os.mkdir(self.best_model_save_dir)
        self.final_model_save_dir = os.path.join(model_save_dir, 'final_checkpoint')
        if not os.path.isdir(self.final_model_save_dir):
            os.mkdir(self.final_model_save_dir)
        # ====从某一断点恢复训练====
        self.load_from = load_from

    def get_data_idxs(self, triples):
        """ 实体关系实体下标的三元组 [(234,13,424),..] , 实体的名字换成id的格式"""
        return [(self.entity_idxs[triple[0]], self.relation_idxs[triple[1]], self.entity_idxs[triple[2]]) for triple in triples]

    def get_hl_t(self, triples):  # h: head entity, l: relationship, t:tail entity.
        """ {(头实体下标,关系下标):[尾实体下标,尾实体下标...],others:[],...} """
        er_vocab = defaultdict(list)
        for triple in triples:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        """
        一个批次的数据获取
        :param er_vocab:  头实体+关系== 尾实体的列表 的集合
        :type er_vocab:  defaultdict
        :param er_vocab_pairs: 头实体+关系的 列表
        :type er_vocab_pairs:  list
        :param idx: 0, 从0到一个batch_size的大小的数据
        :type idx: int
        :return:
        :rtype:
        """
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        # [batch_size, 所有实体的数量], eg:torch.Size([128, 43234])
        targets = torch.zeros([len(batch), len(self.dataset.entities)], dtype=torch.float32)
        targets = targets.cuda()
        for idx, pair in enumerate(batch):
            #er_vocab[pair] 是尾实体的id的列表， 仅让43234中的某个或某些尾实体位置为1，数据有些稀疏
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets

    def evaluate(self, model, test_data):
        """
        模型评估
        :param model:
        :type model:
        :param test_data:
        :type test_data:
        :return:
        :rtype:
        """
        model.eval()
        # 统计命中率
        hits = [[] for _ in range(10)]
        ranks = []
        test_data_idxs = self.get_data_idxs(test_data)
        hl_vocab_t = self.get_hl_t(test_data_idxs)
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch = np.array(test_data_idxs[i: i + self.batch_size])
            e1_idx = torch.tensor(data_batch[:, 0])  # 头实体id列表，一个batch的大小
            r_idx = torch.tensor(data_batch[:, 1])   #关系id列表，一个batch的大小
            e2_idx = torch.tensor(data_batch[:, 2])   # 尾实体id列表，一个batch的大小
            e1_idx = e1_idx.cuda()
            r_idx = r_idx.cuda()
            e2_idx = e2_idx.cuda()
            predictions = model.get_scores(e1_idx, r_idx)  # 放入模型， predictions： torch.Size([128, 43234])
            for j in range(data_batch.shape[0]):
                filt = hl_vocab_t[(data_batch[j][0], data_batch[j][1])]   # ground truth的尾实体
                target_value = predictions[j, e2_idx[j]].item()  # 预测的尾实体
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        hitat10 = np.mean(hits[9])
        hitat3 = np.mean(hits[2])
        hitat1 = np.mean(hits[0])
        meanrank = np.mean(ranks)
        mrr = np.mean(1. / np.array(ranks))
        print('Hits @10: {0}'.format(hitat10))
        print('Hits @3: {0}'.format(hitat3))
        print('Hits @1: {0}'.format(hitat1))
        print('Mean rank: {0}'.format(meanrank))
        print('Mean reciprocal rank: {0}'.format(mrr))
        return [mrr, meanrank, hitat10, hitat3, hitat1]

    def save_checkpoint(self, model, model_dir):
        self.write_vocab_files(model_dir)
        self.write_embedding_files(model, model_dir)

    def write_vocab_files(self, model_dir):
        with open(os.path.join(model_dir, 'idx_entities.dict'),'wt',encoding='utf-8') as out_ie, \
                open(os.path.join(model_dir, 'entities_idx.dict'),'wt',encoding='utf-8') as out_ei:
            for idx, entity in enumerate(self.dataset.entities):
                out_ie.write(str(idx)+'\t'+entity+'\n')
                out_ei.write(entity+'\t'+str(idx)+'\n')
        with open(os.path.join(model_dir, 'idx_relations.dict'), 'wt', encoding='utf-8') as out_ir, \
                open(os.path.join(model_dir, 'relations_idx.dict'), 'wt', encoding='utf-8') as out_ri:
            for idx, relation in enumerate(self.dataset.relations):
                out_ir.write(str(idx) + '\t' + relation + '\n')
                out_ri.write(relation + '\t' + str(idx) + '\n')

    @staticmethod
    def write_embedding_files(model, model_dir):
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        np.save(model_dir + '/E.npy', model.E.weight.data.cpu().numpy())
        np.save(model_dir + '/R.npy', model.R.weight.data.cpu().numpy())
        np.save(model_dir + '/head_bn.npy', {'weight': model.head_bn.weight.data.cpu().numpy(),
                                             'bias': model.head_bn.bias.data.cpu().numpy(),
                                             'running_mean': model.head_bn.running_mean.data.cpu().numpy(),
                                             'running_var': model.head_bn.running_var.data.cpu().numpy()})
        np.save(model_dir + '/score_bn.npy', {'weight': model.score_bn.weight.data.cpu().numpy(),
                                              'bias': model.score_bn.bias.data.cpu().numpy(),
                                              'running_mean': model.score_bn.running_mean.data.cpu().numpy(),
                                              'running_var': model.score_bn.running_var.data.cpu().numpy()})

    def train_and_eval(self):
        best_eval = [0] * 5
        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        print(f'数据集名称: {self.dataset_name}, 实体数量: {len(self.dataset.entities)}, 关系数量: \
        {len(self.dataset.relations)}, 训练集样本数: {len(self.dataset.train_data)}.')
        model = ComplEx_KGE(self.dataset, self.ent_vec_dim, do_batch_norm=self.do_batch_norm,
                            input_dropout=self.input_dropout, hidden_dropout1=self.hidden_dropout1,
                            hidden_dropout2=self.hidden_dropout2)
        if self.load_from:
            model.load_state_dict(torch.load(self.load_from))
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.decay_rate) if 1 > self.decay_rate > 0 else None
        er_vocab = self.get_hl_t(train_data_idxs)  #头实体+关系 = 尾实体的列表
        er_vocab_pairs = list(er_vocab.keys())    # 头实体和关系对
        print("开始训练...")
        start_train = time.time()
        for epo in range(self.num_epochs):
            epoch_idx = epo + 1
            start_epoch = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                # data_batch 是头实体+关系[batch_size,2], 2代表的头实体的id+关系的id， targets是尾实体变成向量，[batch_size, 所有实体的数量], eg:torch.Size([128, 43234])
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                if self.label_smoothing:
                    # 标签平滑，标签不设置为0
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                optimizer.zero_grad()
                # 获取头实体的id
                e1_idx = torch.tensor(data_batch[:, 0]).cuda()
                # 获取关系的id
                r_idx = torch.tensor(data_batch[:, 1]).cuda()
                # 放入模型
                loss = model(e1_idx, r_idx, targets)
                loss.backward()
                optimizer.step()
                # 损z失放入到列表中
                losses.append(loss.item())
            print('epoch:', epoch_idx, 'epoch time:', time.time() - start_epoch, 'loss:', np.mean(losses))
            if epoch_idx % self.test_interval == 0:
                model.eval()
                with torch.no_grad():
                    start_test = time.time()
                    print('验证集评估结果:')
                    valid_res = self.evaluate(model, self.dataset.valid_data)  # mrr, meanrank, hitat10, hitat3, hitat1
                    print('测试集评估结果:')
                    test_res = self.evaluate(model, self.dataset.test_data)  # mrr, meanrank, hitat10, hitat3, hitat1
                    eval_res = (np.add(test_res, valid_res))/2
                    if eval_res[0] >= best_eval[0]:
                        best_eval = eval_res
                        print(f'评估集的评估指标MRR增长, 保存 checkpoint 到 {self.best_model_save_dir}')
                        self.save_checkpoint(model, model_dir=self.best_model_save_dir)
                        print('best model 已保存!')
                        print('评估集最好的指标值的集合:', best_eval)
                    print(f'test time cost: [{time.time() - start_test}]')
            if scheduler:
                scheduler.step(epoch=None)
        print(f'训练完成 ，保存最终checkpoint到： {self.final_model_save_dir}')
        self.save_checkpoint(model, model_dir=self.final_model_save_dir)
        print('模型保存完成')
        print(f'最终训练时间: [{time.time() - start_train}]')


if __name__ == '__main__':
    set_fixed_seed(seed=199839)
    experiment = Experiment(num_epochs=500, test_interval=2, batch_size=128, learning_rate=0.0005, ent_vec_dim=200,
                            rel_vec_dim=200, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                            label_smoothing=0.1, do_batch_norm=True, data_dir='./knowledge_graphs',
                            model_dir='./kg_embeddings', dataset_name='MetaQA', load_from='', decay_rate=1.0)
    experiment.train_and_eval()
