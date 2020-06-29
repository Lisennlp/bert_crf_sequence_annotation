# 项目说明
本项目是基于bert+crf 做的一个中文命名实体识别模型，数据集为“2014 people's daliy newpaper”。项目是基于https://github.com/circlePi/Bert_Chinese_Ner_pytorch
（在此感谢原作者的贡献）进行修改的。在原项目的基础上，加入了多卡训练，同时支持单卡和多卡模式，去掉了一些不必要的code，比如output_mask的形式。

# 训练

      CUDA_VISIBLE_DEVICES=0,1 python run_bert_ner.py   

# 数据集

- 2014 people's daliy newpaper

- The preprocessed data is free to download at https://pan.baidu.com/s/17sa7a-u-cDXjbW4Rok2Ntg

- 数据源格式样本为

      data/source_data/

- 中间读取的数据格式样本

      data/dev.json
      data/train.json


