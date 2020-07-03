# -----------ARGS---------------------
ROOT_DIR = "../bert_nlp/"  # 工作根目录
RAW_SOURCE_DATA = "../bert_nlp/data/ner_data/small.source_BIO_2014_cropus.txt"  # 源数据路径
RAW_TARGET_DATA = "../bert_nlp/data/ner_data/small.target_BIO_2014_cropus.txt"   # 源数据路径

RAW_SOURCE_CWS_TRAIN_DATA = "../bert_nlp/data/cws_data/cws.train"   # 源数据路径
RAW_SOURCE_CWS_TEST_DATA = "../bert_nlp/data/cws_data/cws.test"   # 源数据路径

STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None  # 是否使用预训练模型的词表
VOCAB_FILE = "/nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt"


log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/" 
cache_dir = "model/"
output_dir = "output/checkpoint"    # checkpoint和预测输出文件夹

bert_model = "/nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch"    # BERT 预训练模型种类 bert-base-chinese

TASK_NAMES = ['cws', 'ner', 'postag']
task_name = "cws"    # 训练任务名称, 从TASK_NAMES选取一个

flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 200
do_lower_case = True
train_batch_size = 32
eval_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
seed = 2018
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.

if task_name == 'ner':
    labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O"]
elif task_name == 'cws':
    labels = ["B", "M", "E", "S"]

device = "cuda"

TRAIN_PATH = f"data/{task_name}.train.json"
VALID_PATH = f"data/{task_name}.dev.json"
