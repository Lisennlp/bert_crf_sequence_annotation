import os

from net.model_net import Bert_CRF
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from preprocessing.data_processor import produce_ner_data, produce_cws_data


def start():

    if os.path.exists(args.output_dir):
        for root, dir_, files in os.walk(args.output_dir):
            if any([1 for file in files if file.endswith('.bin')]):
                raise Warning(
                    f"ERROR: your ‘{args.output_dir}’ dir exsits 【model bin】 file, please delete it and run it angain "
                )
    else:
        os.makedirs(args.output_dir)

    if args.task_name == 'ner':

        if os.path.exists('data/ner.train.json'):
            print(f'Ner train format data is existed, now start to read...')
        else:
            produce_ner_data()

    elif args.task_name == 'cws':
        if os.path.exists(args.data_dir):
            print(f'Cws train format data is existed, now start to read...')
        else:
            produce_cws_data(args.RAW_SOURCE_CWS_TRAIN_DATA, args.RAW_CWS_MID_TRAIN_DATA)
            produce_cws_data(args.RAW_SOURCE_CWS_TEST_DATA, args.RAW_CWS_MID_TEST_DATA)
    else:
        raise ValueError(
            f'Task name argument error, please confirm task name is in {args.TASK_NAMES} ')

    train_iter, num_train_steps = create_batch_iter("train", args.TRAIN_PATH)
    eval_iter = create_batch_iter("dev", args.VALID_PATH)

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
