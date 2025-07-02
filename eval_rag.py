import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    dataset_code: str = 'nq',
):
    def get_prompt(question, contexts, topk=3):
        prompt = "Context (which may or may not be relevant):\n"
        for context in contexts[:topk]:
            cur_context = context.split("\n")
            cur_context[0] = cur_context[0].strip('"')
            prompt += "::::".join(cur_context) + "\n"
        prompt += f"\nQuestion: {question}"
        return prompt

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    if dataset_code == 'nq':
        dataset = Dataset.load_from_disk('../RAG_Eval/NQ_Eval')
    elif dataset_code == 'tq':
        dataset = Dataset.load_from_disk('../RAG_Eval/TQ_Eval')
    elif dataset_code == '2wiki':
        dataset = Dataset.load_from_disk('../RAG_Eval/2Wiki_Eval')
    elif dataset_code == 'hotpotqa':
        dataset = Dataset.load_from_disk('../RAG_Eval/HotpotQA_Eval')
    elif dataset_code == 'bamboogle':
        dataset = Dataset.load_from_disk('../RAG_Eval/Bamboogle_Eval')

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        # Prepare prompts using the same format as training
        prompts = [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(q, c).strip()},
        ] for q, c in zip(batch_data["question"], batch_data["contexts"])]


        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = process_qa_answer(extracted)
            true_answer = batch_data['golden_answers'][j]
            true_answer = [process_qa_answer(y) for y in true_answer]
            print(generated_answer, true_answer, generated_answer in true_answer)

            # Store the result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer in true_answer,
            }
            results.append(result)

            if generated_answer in true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        save_path = adapter_path + f"/eval_results_{dataset_code}.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_examples", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        if model.split('/')[-1] in checkpoint_path:
            base_model = model
    temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
    print(checkpoint_path, base_model, temperature)

    if 'eval_results_nq.json' not in os.listdir(checkpoint_path):
        print(f"Starting NQ evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=args.eval_examples,
            save_results=True,
            dataset_code='nq',
        )

    if 'eval_results_tq.json' not in os.listdir(checkpoint_path):
        print(f"Starting TQ evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=args.eval_examples,
            save_results=True,
            dataset_code='tq',
        )

    if 'eval_results_2wiki.json' not in os.listdir(checkpoint_path):
        print(f"Starting 2Wiki evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=args.eval_examples,
            save_results=True,
            dataset_code='2wiki',
        )

    if 'eval_results_hotpotqa.json' not in os.listdir(checkpoint_path):
        print(f"Starting HotpotQA evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=args.eval_examples,
            save_results=True,
            dataset_code='hotpotqa',
        )

    if 'eval_results_bamboogle.json' not in os.listdir(checkpoint_path):
        print(f"Starting Bamboogle evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=args.eval_examples,
            save_results=True,
            dataset_code='bamboogle',
        )