import time
import warnings
import os

import torch

from pytorch_pretrained_bert.optimization import BertAdam
import config.args as args
from util.plot_util import loss_acc_plot
from util.Logginger import init_logger
from util.model_util import save_model

logger = init_logger("torch", logging_path=args.log_path)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
warnings.filterwarnings('ignore')

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'bert_config.json'


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, verbose=1):
    # ------------------判断CUDA模式----------------------
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]

    t_total = num_train_steps

    ## ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    ## ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    # ---------------------模型初始化----------------------
    if args.fp16:
        model.half()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info(f"Device {device} n_gpu {n_gpu} distributed training")

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model_module = model.module
    else:
        model_module = model

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }

    # ------------------------训练------------------------------
    start = time.time()
    global_step = 0

    for e in range(num_epoch):
        model.train()
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            # lsp: output_mask是为了过滤掉带有‘##’的token，即subword，这些词不计算loss，同时也是为了保证label_ids和token的长度一致
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()

            train_loss = model_module.loss_fn(bert_encode=bert_encode,
                                              tags=label_ids,
                                              output_mask=output_mask)
            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(train_loss)
            else:
                train_loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                  args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            predicts = model_module.predict(bert_encode, output_mask)
            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids.cpu()
            if len(predicts) != len(label_ids):
                continue
            train_acc, f1 = model_module.acc_f1(predicts, label_ids)    # lsp: 每一步的准确率情况
            pbar.show_process(train_acc, train_loss.item(), f1, time.time() - start, step)

# -----------------------验证----------------------------
        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = model(input_ids, segment_ids, input_mask).cpu()
                eval_los = model_module.loss_fn(bert_encode=bert_encode,
                                                tags=label_ids,
                                                output_mask=output_mask)
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = model_module.predict(bert_encode, output_mask)
                y_predicts.append(predicts)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)

            eval_predicted = torch.cat(y_predicts, dim=0).cpu()
            eval_labeled = torch.cat(y_labels, dim=0).cpu()

            eval_acc, eval_f1 = model_module.acc_f1(eval_predicted, eval_labeled)
            model_module.class_report(eval_predicted, eval_labeled)

            logger.info(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                %
                (e + 1, train_loss.item(), eval_loss.item() / count, train_acc, eval_acc, eval_f1))

            save_model(model, args.output_dir, step=e, f_1=eval_f1)  # 所有模型都保存

            if e == 0:
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_module.config.to_json_file(output_config_file)

            if e % verbose == 0:
                train_losses.append(train_loss.item())
                train_accuracy.append(train_acc)
                eval_losses.append(eval_loss.item() / count)
                eval_accuracy.append(eval_acc)

    loss_acc_plot(history)
