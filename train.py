import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import json
from data import prepare_dataset
from datasets import concatenate_datasets
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from constants import TRAINING_CONFIG_PATH

os.environ["WANDB_PROJECT"] = "legal-lens-ner"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def main(args):

    train_prompt_dataset, valid_prompt_dataset = prepare_dataset(args['dataset_repo'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args['pretrained_ckpt'],
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args['pretrained_ckpt'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    full_modules = ["q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head"]

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=args['dropout'],
        r=args['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=full_modules 
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args['results_dir'],
        logging_dir=f"{args['results_dir']}/logs",
        num_train_epochs=args['epochs'],
        per_device_train_batch_size=25,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb"
    )

    max_seq_length = 2048 
    response_template = " ASSISTANT:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_prompt_dataset,
        eval_dataset=valid_prompt_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        args=training_args,
        dataset_text_field="instructions",
        neftune_noise_alpha=args['neftune'], 
        #data_collator=collator,
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{args['results_dir']}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{args['results_dir']}/results.pkl", "wb") as handle:
        run_result = [
            args['epochs'],
            args['lora_r'],
            args['dropout'],
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")
  
if __name__ == "__main__":
 
    with open(TRAINING_CONFIG_PATH, 'r') as config_file:
        args = json.load(config_file)
    
    main(args)