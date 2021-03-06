import os
import numpy as np
import torch

from transformers import T5Config, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from data_utils import QAData
from model import t5model

def run(args, logger):
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    train_data = QAData(logger, args, args.train_file)
    dev_data = QAData(logger, args, args.predict_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    model = t5model.from_pretrained("t5-large")
    model.to(device)

    if args.do_train:

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')
        logger.info("Loading checkpoint from {}".format(checkpoint))
        # TODO add loader from checkpoint
        model.eval()
        ems = inference(model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1
            batch.to(device)
            loss = model(input_ids=batch[0], 
                        attention_mask=batch[1],
                        decoder_input_ids=batch[2], 
                        decoder_attention_mask=batch[3],
                        is_training=True)
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_em*100,
                        epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                            (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break

def inference(model, dev_data, save_predictions=True):
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, batch in enumerate(dev_data.dataloader):
        batch.to(device)
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        pred = dev_data.decode_batch(outputs)
        predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))