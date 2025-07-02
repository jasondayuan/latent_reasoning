import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL

import os
import argparse
from transformers import TrainingArguments, Trainer # Has to ensure this is run from the repo directory or else it will import the original Trainer
from latent_trainer import LatentTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(
        process_gsm8k,
        batched=True,
        batch_size=chunk_size,
        load_from_cache_file=False,
    )


def main(args):
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}"
                f"-lora{args.lora_rank}-temp{args.temperature}")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    training_args = TrainingArguments(
        learning_rate = args.lr,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = "wandb",
        output_dir = exp_name,
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = LatentTrainer(
        model = model,
        tokenizer = tokenizer, 
        args = training_args,
        train_dataset = dataset,
    )

    ### Modifies gating parameters' learning rates, not used ###
    # patch_trainer_optimizer(
    #     trainer,
    #     args.lr_residual_gate,
    #     args.lr_residual_Lambda,
    # )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)

    ### Coefficient of KL divergence loss, not used ###
    parser.add_argument("--beta", type=float, default=0.005)

    ### Gating related, not used ###
    # parser.add_argument("--residual_r_min", type=float, default=0.99)
    # parser.add_argument("--residual_r_max", type=float, default=0.999)
    # parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    # parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    ### Group size for GRPO, not used ###
    # parser.add_argument("--group_size", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)