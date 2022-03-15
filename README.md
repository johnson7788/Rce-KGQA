# Rce-KGQA
一种用于多跳复杂KGQA任务的新型pipeline框架。
该框架主要包括两个模块，**应答过滤模块**和**关系链推理模块**
这两个模块应该独立训练，在参考步骤，将问题和KG加载到**应答_过滤_模块**输入中，然后获得前K名候选者
，并以KG为单位检索这些候选对象的关系链，并让**关系链推理模块**向用户提供最终答案。

> overall pipeline architecture 
[See model](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/all_architecture.pdf)

>answering_filtering_module
[See Module1](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/answer_filtering.pdf)

>relational_chain_reasoning_module
[See Module2](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/relational_chain_reasoning.pdf)

统计性能比较：

### 
MetaQA三个子集的实验结果。第一组结果来自于关于最新方法的论文。这些值是使用hits@1.

| Model | 1-hop MetaQA | 2-hop MetaQA | 3-hop MetaQA ||

| :-----| ----: | :----: ||

| EmbedKGQA | 97.5 | 98.8 | 94.8 ||

| SRN | 97.0 | 95.1 | 75.2 ||

| KVMem | 96.2 |  82.7 |  48.9 ||

| GraftNet | 97.0 |  94.8 |  77.7 ||

| PullNet | 97.0 | 99.9 | 91.4 ||

| Our Model | 98.3 | 99.7 | 97.9 ||

### Experimental results on Answer Reasoning on WebQuestionsSP-tiny.
在WebQuestionsSP-tiny测试集上，将实验结果与SOTA方法进行了比较。WebQuestionsSP tiny中的所有QA对都是两跳关系问题。

| Model | WebQuestionsSP-tiny hit@1 ||

| EmbedKGQA | 66.6 ||

| SRN | - ||

| KVMem | 46.7 ||

| GraftNet | 66.4 ||

| PullNet | 68.1 ||

| Our Model | 70.4 ||

Hope you enjoy it !!!  Arxiv link: https://arxiv.org/abs/2110.12679
