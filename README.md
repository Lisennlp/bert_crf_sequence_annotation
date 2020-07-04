# 项目说明
本项目是基于bert+crf 做的一个NLP序列标注（目前支持分词，词性标注和实体识别训练，后续会加入其他的任务）模型，项目是基于https://github.com/circlePi/Bert_Chinese_Ner_pytorch
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

### 词性训练

      修改task_name为postag，以及一些其他超参数

      CUDA_VISIBLE_DEVICES=0,1 python run.py   

### 实体识别训练数据样例

      data/ner_data/

### 分词训练数据样例

      data/cws_data/

### 词性训练数据样例

      data/npostag_data/    


### 词性标注符号：

      n   普通名词
      nt  时间名词
      nd  方位名词
      nl  处所名词
      nh  人名
      nhf 姓
      nhs 名
      ns  地名
      nn  族名
      ni  机构名
      nz  其他专名
      v   动词
      vd  趋向动词
      vl  联系动词
      vu  能愿动词
      a   形容词
      f   区别词
      m   数词　　
      q   量词
      d   副词
      r   代词
      p   介词
      c   连词
      u   助词
      e   叹词
      o   拟声词
      i   习用语
      j   缩略语
      h   前接成分
      k   后接成分
      g   语素字
      x   非语素字
      w   标点符号
      ws  非汉字字符串
      wu  其他未知的符号

但因为中文的词性都是以词为单位的，因此，我们将每个词都进行了拆分，比如中国/n，拆分后就是：中/B-n，国/I-n；天安门/n，拆分为天/B-n，安/I-n，门/I-n。 
因此，所有的label都进行了拓展，其中UNK表示可能存在不在列表中的词性。如下所示：

      B-n I-n B-nt I-nt B-nd I-nd B-nl I-nl B-nh I-nh B-nhf I-nhf B-nhs I-nhs B-ns I-ns B-nn I-nn B-ni I-ni B-nz I-nz B-v I-v B-vd I-vd B-vl I-vl B-vu I-vu B-a I-a B-f I-f B-m I-m B-q I-q B-d I-d B-r I-r B-p I-p B-c I-c B-u I-u B-e I-e B-o I-o B-i I-i B-j I-j B-h I-h B-k I-k B-g I-g B-x I-x B-w I-w B-ws I-ws B-wu I-wu UNK
     

      