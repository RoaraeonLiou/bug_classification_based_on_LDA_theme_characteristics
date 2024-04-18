# BugClassification
基于LDA和双向GRU的软件缺陷分类

### 数据集与预处理 
数据来源于Jira系统中的Lucene、JackRabbit和Httpclient三个软件缺陷报告集，这三个项目均属于Apache的子项目.

在我们执行分类任务前，首先需要对这些数据集进行初步处理，除了对单个项目的数据集进行处理之外，我们还对混合后的数据集进行同样的处理工作，处理步骤下：
（1）从软件缺陷报告中抽取summary、description和priority字段信息；
（2）对summary和description字段的信息进行文本预处理步骤；
（3）使用LDA主题模型提取软件缺陷报告的主题特征；
（4）从软件缺陷报告校正后的分类文件中抽取出每个报告所对应的类别；
（5）对预处理后的文本向量进行对齐操作；
（6）将处理好的输入数据划分为训练数据集、验证数据集和测试数据集，并写入文件；

### 实验环境参数
- OS：Ubuntu18.04
- CPU：Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz
- TensorFlow 2.1.0



