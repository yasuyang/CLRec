import os
import fire
import numpy as np
import torch
from module import SunModule
import json
from sklearn.model_selection import train_test_split
from data_process.utils import split_json_file
import transformers
from datasets import load_dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


def main(backbone: str, train_path:str,valid_path:str, task: str, dataset: str,
         cutoff: int, model_dir: str, batch_size: int, valid_select: int,
         epochs: int, lr: float, warmup_steps: int, gradient_accumulation_steps: int,
         logging_steps: int, optim: str, eval_steps: int, save_steps: int, save_total_limit: int, is_whole_sentence: bool):
    config = T5Config.from_pretrained(backbone)
    model = SunModule.from_pretrained(backbone,config=config)
    model.init_prompt('cuda')
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    valid_data = load_dataset("json", data_files=valid_path, field='data')

    train_data = load_dataset("json", data_files=train_path, field='data')


    def compute_whole_word_id(seq_batch, tokenizer):
        whole_word_ids = []
        for seq in seq_batch:
            token_list = tokenizer.tokenize(seq)
            start_indices = []
            for idx, token in enumerate(token_list):
                if token == '_':
                    start_indices.append(idx - 1)  # user_xx or item_xx, starts before _
            end_indices = []
            for start in start_indices:
                mover = start + 2  # user/item _ xx
                while mover < len(token_list) and token_list[mover].isdigit():
                    mover += 1
                end_indices.append(mover)
            whole_word_id = [0] * (len(token_list)+1)  # padding
            for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                whole_word_id[start:end] = [i + 1] * (end - start)  # leave 0 as padding token
            whole_word_ids.append(whole_word_id)

        # make the batch of the same length

        return whole_word_ids
    def compute_whole_sentence_id(seq_batch, tokenizer):
        padded_whole_sentences_ids = []
        for seq in seq_batch:
            # 获得序列的真实长度
            token_list = tokenizer.tokenize(seq)
            sentence_mark = 0
            for idx, token in enumerate(token_list):
                if token == '?':
                    sentence_mark = idx
            padded_whole_sentences_id = [0] * len(token_list)
            padded_whole_sentences_id[0:sentence_mark] = [1] * (sentence_mark)
            padded_whole_sentences_id[sentence_mark+1:] = [2] * (len(token_list) - sentence_mark)
            padded_whole_sentences_ids.append(padded_whole_sentences_id)
        return padded_whole_sentences_ids
    def process_func(datapoint,is_whole_sentence=True):#对数据进行处理
        # padded_whole_word_ids, padded_whole_sentences_ids = compute_whole_word_sentence_id(datapoint['input'],tokenizer, cutoff)
        encoding = tokenizer(datapoint['input'],padding=False)

        # 获取形状
        whole_word_ids = compute_whole_word_id(datapoint['input'],tokenizer)
        whole_sentences_ids = compute_whole_sentence_id(datapoint['input'],tokenizer)

        # 获取形状
        labels = tokenizer(datapoint['input'],padding=False)
        encoding['labels'] = labels['input_ids'].copy()
        for i, (m, n, q, q1, q2) in enumerate(
                zip(encoding['input_ids'], encoding['labels'], encoding['attention_mask'], whole_word_ids,
                    whole_sentences_ids)):
            if len(m) >= 512:
                encoding['input_ids'][i] = encoding['input_ids'][i][:512]
                encoding['attention_mask'][i] = encoding['attention_mask'][i][:512]
                encoding['labels'][i] = encoding['labels'][i][:512]
                whole_word_ids[i] = whole_word_ids[i][:512]
                whole_sentences_ids[i] = whole_sentences_ids[i][:512]
                continue
            encoding['input_ids'][i] = m + [0] * (512 - len(m))
            encoding['labels'][i] = n + [0] * (512 - len(n))
            encoding['attention_mask'][i] = q + [0] * (512 - len(q))
            whole_word_ids[i] = q1 + [0] * (512 - len(q1))
            whole_sentences_ids[i] = q2 + [0] * (512 - len(q2))
        #return {**datapoint, **encoding,'whole_word_ids':padded_whole_word_ids,'whole_sentence_ids':padded_whole_sentences_ids}
        if is_whole_sentence:
            return {**encoding, 'whole_word_ids': whole_word_ids, 'whole_sentence_ids': whole_sentences_ids}
        else:
            return {**encoding, 'whole_word_ids': whole_word_ids}
        # return  {**encoding}

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    tokenizer.padding_side = "right"
    train_set = train_data['train'].shuffle().map(lambda x: process_func(x,is_whole_sentence), batched=True)
    #train_set = train_data['train'].shuffle().map(process_func, batched=True)

    valid_set = valid_data['train'].shuffle().map(lambda x: process_func(x,is_whole_sentence), batched=True)
    output_dir = model_dir
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=valid_set if valid_select > 0 else None,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,#时间换空间，如果设置为n，则我们forward n次，得到N个loss的累加后再更新
            warmup_steps=warmup_steps,#直接指定经过多少个steps到达初始学习率，默认值为0-一开始使用初始学习率。然后学习率会随着steps的数量线性下降
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=logging_steps,
            optim=optim,
            evaluation_strategy="steps" if valid_select > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if valid_select > 0 else None,
            save_steps=save_steps,
            dataloader_drop_last=False,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if valid_select > 0 else False,
            group_by_length=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)