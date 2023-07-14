import argparse
import os
from data import RecommendTrainDataset, IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import t5_special_tokens_dict
from datetime import datetime
from pytz import timezone
from utils import get_time_kst
import sys
import json


class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer,
                 results_file_path, rec_pred_file_path, target_id_type):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.epoch = 0
        self.target_id_type = target_id_type
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )
        self.results_file_path = results_file_path
        self.rec_pred_file_path = rec_pred_file_path
        self.know2id = json.load(open('data/MG/knowledge_kmeansid.json', 'r', encoding='utf-8'))
        self.id2know = {v: k for k, v in self.know2id.items()}
        # self.movie2name = json.load(open('data/Redial/movie2name.json', 'r', encoding='utf-8'))
        self.all_knowledge = json.load(open('data/MG/mgcrs_allknowledges.json', 'r', encoding='utf-8'))
        self.orgIdx2allKnow = {v: k for k, v in self.all_knowledge.items()}

    def on_epoch_end(self, args, state, control, **kwargs):
        print("==============================Evaluate step==============================")
        self.epoch += 1
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        batch_index = 0
        batch_rank_list = []
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                if self.target_id_type == 0:
                    batch_beams = model.generate(
                        inputs['input_ids'].to(model.device),
                        max_length=20,
                        num_beams=5,
                        prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                        num_return_sequences=5,
                        early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                elif self.target_id_type == 1:
                    batch_beams = model.generate(
                        inputs['input_ids'].to(model.device),
                        max_length=20,
                        num_beams=5,
                        num_return_sequences=10,
                        early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                self.logger.log({"batch_beams": batch_beams, "labels": labels})
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    batch_rank_list.append(rank_list)
                    hits = np.array(rank_list)[:10] == label
                    # print(rank_list)
                    # print("============")
                    # print(label)
                    # print("============")
                    # print(hits)
                    if True in hits[:5]:
                        hit_at_5 += 1
                    if True in hits[:3]:
                        hit_at_3 += 1
                    if True in hits[:1]:
                        hit_at_1 += 1
                # Save only first batch
                if batch_index == 0:
                    with open(self.rec_pred_file_path, 'a', encoding='utf-8') as pred_f:
                        pred_f.write(f"==========================================\n")
                        for i in range(len(labels)):
                            pred_f.write(json.dumps({
                                'Input: ': self.tokenizer.decode(inputs['input_ids'][i]),
                                'Pred: ': [self.orgIdx2allKnow[int(self.id2know[str(pred)])] if str(
                                    pred) in self.id2know.keys() else str(pred) for pred in batch_rank_list[i]],
                                'Label: ': self.orgIdx2allKnow[int(self.id2know[str(labels[i])])]
                            }) + '\n')
        self.logger.log({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@3": hit_at_3 / len(self.test_dataset),
                         "Hits@5": hit_at_5 / len(self.test_dataset), "epoch": self.epoch})
        with open(self.results_file_path, 'a', encoding='utf-8') as result_f:
            result_f.write('[FINE TUNING] Epoch:\t%d\t%.4f\t%.4f\t%.4f\n' % (
                self.epoch, 100 * hit_at_1, 100 * hit_at_3, 100 * hit_at_5,))
        print({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@3": hit_at_3 / len(self.test_dataset),
               "Hits@5": hit_at_5 / len(self.test_dataset), "epoch": self.epoch})
        batch_index += 1
        print("==============================End of evaluate step==============================")


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def parse_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--name', type=str, default="loggertest")
    parser.add_argument('--model_name', type=str, default='t5-large', choices=['t5-base', 't5-large'])
    parser.add_argument('--max_dialog_len', type=int, default=128)
    parser.add_argument('--num_index_epochs', type=int, default=5)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--evaluation_strategy', type=str, default="no")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_reviews', type=str, default="0")
    parser.add_argument('--prefix', type=bool, default=True)
    parser.add_argument('--saved_model_path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="prefix", choices=["title2title", "", "prefix"])
    parser.add_argument('--target_id_type', type=int, default=0)  # 0: id, 1: String
    parser.add_argument('--train_type', type=int,
                        default=0)  # 0: multi-task, #1: indexing -> multi-task

    args = parser.parse_args()
    return args


def createResultFile(args):
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    rawSubfolder_name = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d') + '_raw')
    predfolder_name = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d') + '_RecPred')
    rawFolder_path = os.path.join('./results', rawSubfolder_name)
    predfolder_path = os.path.join('./results', predfolder_name)
    if not os.path.exists(rawFolder_path): os.mkdir(rawFolder_path)
    if not os.path.exists(predfolder_path): os.mkdir(predfolder_path)
    results_file_path = os.path.join(rawFolder_path,
                                     f"{mdhm}_train_device_{args.device_id}_name_{args.name}.txt")
    rec_pred_file_path = os.path.join(predfolder_path,
                                      f"{mdhm}_train_device_{args.device_id}_name_{args.name}.txt")

    # parameters
    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        result_f.write('Argument List:' + str(sys.argv) + '\n')
        for i, v in vars(args).items():
            result_f.write(f'{i}:{v} || ')
        result_f.write('\n')
        result_f.write('Hit@1\tHit@3\tHit@5\n')

    return results_file_path, rec_pred_file_path


def main(args):
    L = 128  # only use the first 32 tokens of documents (including title)

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project="DSI-MG", name=args.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    result_file_path, rec_pred_file_path = createResultFile(args)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir='cache')
    tokenizer.add_special_tokens(t5_special_tokens_dict)

    if int(args.num_reviews) > 0:
        path_to_train_dataset = f'data/Redial/other/train_{args.dataset}_review_{args.num_reviews}.json'
    else:
        path_to_train_dataset = f'data/MG/train.json'
    train_dataset = RecommendTrainDataset(
        path_to_data=path_to_train_dataset,
        max_length=args.max_dialog_len,
        cache_dir='cache',
        tokenizer=tokenizer,
        usePrefix=args.prefix,
        mode='train'
    )
    print("=================================")
    print("LEN TRAIN DATASET: ", len(train_dataset)) # 14879 = 11621 + 3258
    print("=================================\n")
    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    # eval_dataset = RecommendTrainDataset(path_to_data=f'data/Redial/other/valid_{args.dataset}.json',
    #                                      max_length=args.max_dialog_len,
    #                                      cache_dir='cache',
    #                                      tokenizer=tokenizer,
    #                                      usePrefix=args.prefix,
    #                                      mode='test'
    #                                      )

    # This is the actual eval set.
    test_dataset = RecommendTrainDataset(path_to_data=f'data/MG/test.json',
                                         max_length=args.max_dialog_len,
                                         cache_dir='cache',
                                         tokenizer=tokenizer,
                                         usePrefix=args.prefix,
                                         mode='test'
                                         )
    print("=================================")
    print("LEN TEST DATASET: ", len(test_dataset)) # 3711
    print("=================================\n")
    ################################################################
    # docid generation constrain, we only generate integer docids. --> 근데 _ 로 시작하는건 왜 넣는거지?
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    ################################################################

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.learning_rate,  # 0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        # eval_steps=1000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=50,
        save_strategy='no',
        num_train_epochs=args.num_train_epochs,
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=1
        # gradient_accumulation_steps=2
    )

    if args.saved_model_path == '':
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir='cache')
        if args.train_type == 1:  # and int(args.num_reviews) != 0:
            index_dataset = IndexingTrainDataset(
                path_to_data=f'data/MG/indexing.json',
                max_length=args.max_dialog_len,
                cache_dir='cache',
                tokenizer=tokenizer,
                usePrefix=args.prefix)

            print("=================================")
            print("LEN INDEX DATASET: ", len(index_dataset))
            print("=================================\n")

            training_args.num_train_epochs = args.num_index_epochs
            index_trainer = IndexingTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=index_dataset,
                # eval_dataset=eval_dataset,
                data_collator=IndexingCollator(
                    tokenizer,
                    padding='longest',
                ),
                # compute_metrics=compute_metrics,
                # callbacks=[IndexEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer)],
                restrict_decode_vocab=restrict_decode_vocab
            )
            print("=============Train indexing=============")
            index_trainer.train()
            index_trainer.save_model(f'model/MG_indexing')
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.saved_model_path)
    model.resize_token_embeddings(len(tokenizer))

    training_args.num_train_epochs = args.num_train_epochs
    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer,
                              result_file_path, rec_pred_file_path, args.target_id_type)],
        restrict_decode_vocab=restrict_decode_vocab
    )
    print("=============Train Recommender=============")
    trainer.train(
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
