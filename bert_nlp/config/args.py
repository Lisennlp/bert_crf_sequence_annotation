
VOCAB_FILE = "/nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt"

log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"
cache_dir = "model/"
output_dir = "output/checkpoint"    # checkpoint和预测输出文件夹

bert_model = "/nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch"    # BERT 预训练模型种类 bert-base-chinese

TASK_NAMES = ['cws', 'ner', 'postag']
task_name = "postag"    # 训练任务名称, 从TASK_NAMES选取一个

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
elif task_name == 'postag':
    labels = [
        'B-n', 'I-n', 'B-nt', 'I-nt', 'B-nd', 'I-nd', 'B-nl', 'I-nl', 'B-nh', 'I-nh', 'B-nhf',
        'I-nhf', 'B-nhs', 'I-nhs', 'B-ns', 'I-ns', 'B-nn', 'I-nn', 'B-ni', 'I-ni', 'B-nz', 'I-nz',
        'B-v', 'I-v', 'B-vd', 'I-vd', 'B-vl', 'I-vl', 'B-vu', 'I-vu', 'B-a', 'I-a', 'B-f', 'I-f',
        'B-m', 'I-m', 'B-q', 'I-q', 'B-d', 'I-d', 'B-r', 'I-r', 'B-p', 'I-p', 'B-c', 'I-c', 'B-u',
        'I-u', 'B-e', 'I-e', 'B-o', 'I-o', 'B-i', 'I-i', 'B-j', 'I-j', 'B-h', 'I-h', 'B-k', 'I-k',
        'B-g', 'I-g', 'B-x', 'I-x', 'B-w', 'I-w', 'B-ws', 'I-ws', 'B-wu', 'I-wu', 'UNK'
    ]

device = "cuda"

TRAIN_PATH = f"data/{task_name}_data/{task_name}.train.json"
VALID_PATH = f"data/{task_name}_data/{task_name}.dev.json"
