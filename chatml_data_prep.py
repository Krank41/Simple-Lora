import transformers
import torch
import json
import os
from datasets import load_dataset
from functools import partial
from typing import Dict


# Format (chatML)
templates=[
    "<|im_start|>assistant\n{msg}<|im_end|>",
    "<|im_start|>user\n{msg}<|im_end|>"
]
IGNORE_INDEX=-100

def init_adding_tokens(tokenizer,model):
    tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
    tokenizer.pad_token = "<PAD>"
    num_new_tokens = tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer,num_new_tokens

def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    num_new_tokens,

):
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def data_collator(elements):
    tokenizer_pad_token_id = 32001
    
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])
    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        attention_mask=sample["attention_mask"]
        pad_len=tokens_maxlen-len(input_ids)
        input_ids.extend( pad_len * [tokenizer_pad_token_id] )
        labels.extend( pad_len * [IGNORE_INDEX] )
        attention_mask.extend( pad_len * [0] )
    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),
        "labels": torch.tensor( [e["labels"] for e in elements] ),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),
    }

    return batch


def tokenize(input, tokenizer,max_length):
    input_ids, attention_mask, labels = [], [], []
    for i,msg in enumerate(input["messages"]):
        isHuman = i%2==0
        msg_chatml=templates[isHuman].format(msg=msg)
        msg_tokenized=tokenizer(msg_chatml, truncation=False, add_special_tokens=False)
        input_ids+=msg_tokenized["input_ids"]
        attention_mask+=msg_tokenized["attention_mask"]
        labels+=[IGNORE_INDEX]*len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]
    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length],
    }

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    dataseti = load_dataset("json", data_files={"train": data_path})
    dataset = dataseti["train"].train_test_split(test_size=0.1)
    dataset_tokenized = dataset.map(
        partial(tokenize,tokenizer=tokenizer, max_length=256),
        batched=False,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names
    )
    return dict(train_dataset=dataset_tokenized["train"], eval_dataset=dataset_tokenized["test"], data_collator=data_collator)
