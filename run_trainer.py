import torch
import transformers
import json
from typing import  Optional
from transformers_lora_model import  LoraLlamaForCausalLM
from transformers_lora_model import LoraSFTTrainer
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="")
    tokenizer_path: Optional[str] = field(default="")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    model_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def qtrain(model_args,training_args):
    ####################  STEP 1 : Loading Model #############################
    model = LoraLlamaForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        loading_adapter = False,

    )
    print("model >>",model)
    ################### STEP 2 : Loading Tokenizer ###########################
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
        )
    ################### STEP 3 : Data-Pre-processing for chatml format ###########
    from chatml_data_prep import init_adding_tokens
    from chatml_data_prep import smart_tokenizer_and_embedding_resize
    from chatml_data_prep import make_supervised_data_module

    tokenizer,num_new_tokens = init_adding_tokens(tokenizer,model)
    smart_tokenizer_and_embedding_resize(
        tokenizer=tokenizer,
        model=model,
        num_new_tokens=num_new_tokens,)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=model_args.data_path)
    ################### STEP 4 : Turning On trainable parameters ##################
    for names,p in model.named_parameters():
        p.requires_grad = False

    for names,p in model.named_parameters():
        if "lora" in names:
            p.requires_grad = True
    ##################### STEP 5 : SFT Trainer ######################################
    trainer = LoraSFTTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

################################################################Loading configurations#################

with open('train_config.json', 'r') as f:
    config = json.load(f)

model_args = ModelArguments(
    model_path=config["model_args"]["model_path"],
    tokenizer_path=config["model_args"]["tokenizer_path"],
    data_path=config["model_args"]["data_path"],
    model_max_length=config["model_args"]["model_max_length"]
)

training_args = transformers.TrainingArguments(
    output_dir=config["training_args"]["output_dir"],
    num_train_epochs=config["training_args"]["num_train_epochs"],
    per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training_args"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
    evaluation_strategy=config["training_args"]["evaluation_strategy"],
    save_steps=config["training_args"]["save_steps"],
    save_total_limit=config["training_args"]["save_total_limit"],
    learning_rate=config["training_args"]["learning_rate"],
    weight_decay=config["training_args"]["weight_decay"],
    warmup_ratio=config["training_args"]["warmup_ratio"],
    lr_scheduler_type=config["training_args"]["lr_scheduler_type"],
    logging_steps=config["training_args"]["logging_steps"],
    optim=config["training_args"]["optim"]
)

########################################################################################################

qtrain(model_args,training_args)

## command
#torchrun --nproc_per_node=1 --master_port=8000 run_trainer.py

# ################### STEP 3 : Data Pre-processing for alpaca format  ###########################
# from alpaca_data_prep import smart_tokenizer_and_embedding_resize
# from alpaca_data_prep import make_supervised_data_module
# from alpaca_data_prep import init_special_tokens_dict
# special_tokens_dict,tokenizer = init_special_tokens_dict(tokenizer)
# smart_tokenizer_and_embedding_resize(
#     special_tokens_dict=special_tokens_dict,
#     tokenizer=tokenizer,
#     model=model,
# )
# data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=model_args.data_path)
##################################################################################################
