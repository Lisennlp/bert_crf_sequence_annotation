# 项目说明
本项目是基于bert+crf 做的一个NLP序列标注（目前支持分词和实体识别训练，后续会加入其他的任务）模型，项目是基于https://github.com/circlePi/Bert_Chinese_Ner_pytorch
（在此感谢原作者的贡献）进行修改的。在原项目的基础上，加入了多卡训练，同时支持单卡和多卡模式，去掉了一些不必要的code，比如output_mask的形式。此外，加入了分词等nlp任务。


# 配置文件

      bert_nlp/config/args.py

# 训练

### 分词训练

      修改task_name为cws，以及一些其他超参数

      CUDA_VISIBLE_DEVICES=0,1 python run.py   

### 实体识别训练

      修改task_name为ner，以及一些其他超参数

      CUDA_VISIBLE_DEVICES=0,1 python run.py    

### 实体识别源数据样例

      data/ner_data/

### 分词源数据样例

      data/cws_data/

### 中间读取的数据格式样本

      data/*.json
      data/*.json


