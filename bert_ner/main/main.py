import os

from net.bert_ner import Bert_CRF
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from preprocessing.data_processor import produce_data, produce_cws_data


def start():
    if not os.path.exists(args.data_dir):
        print(f'train format data not is existed, now start to process...')
        # 数据将daliy paper格式存为args.data_dir中数据格式
        # produce_data()
    produce_cws_data(args.RAW_SOURCE_CWS_TRAIN_DATA, args.RAW_CWS_MID_TRAIN_DATA)
    produce_cws_data(args.RAW_SOURCE_CWS_TEST_DATA, args.RAW_CWS_MID_TEST_DATA)

    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")
    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs
    print(f'epoch_size = {epoch_size}')
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)
    model = Bert_CRF.from_pretrained(args.bert_model, num_tag=len(args.labels))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)
