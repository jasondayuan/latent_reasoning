# Hybrid Latent Reasoning via Reinforcement Learning

This repository provides the PyTorch implementation for the preprint **Hybrid Latent Reasoning via Reinforcement Learning [[Paper](https://arxiv.org/abs/2505.18454)]**. In this work, we explore latent reasoning by leveraging the intrinsic capabilities of LLMs via reinforcement learning (RL). To this end, we introduce hybrid reasoning policy optimization (HRPO), an RL-based hybrid latent reasoning approach that (1) integrates prior hidden states into sampled tokens with a learnable gating mechanism, and (2) initializes training with predominantly token embeddings while progressively incorporating more hidden features. This design maintains LLMs' generative capabilities and incentivizes hybrid reasoning using both discrete and continuous representations. In addition, the hybrid HRPO introduces stochasticity into latent reasoning via token sampling, thereby enabling RL-based optimization without requiring CoT trajectories. Extensive evaluations across diverse benchmarks show that HRPO outperforms prior methods in both knowledge- and reasoning-intensive tasks. Furthermore, HRPO-trained LLMs remain interpretable and exhibit intriguing behaviors like cross-lingual patterns and shorter completion lengths, highlighting the potential of our RL-based approach and offer insights for future work in latent reasoning.

<img src=assets/intro.png width=1000>


## Train Qwen with HRPO

To train Qwen using the HRPO framework on your chosen dataset, run the corresponding script. For example, to train on GSM8K:

```bash
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --residual_r_min 0.98 \
  --group_size 8 \
```

Key arguments:
* `--model_name`
  Directory or name of the HF model.
* `--group_size`
  Number of candidate reponses sampled for each query.
* `--residual_r_min`
  Minimum initialization radius for $\Lambda$ in HRPO gating.
* `--temperature`
  Sampling temperature for latent exploration.

The scripts `hrpo_mmlu.py` and `hrpo_rag.py` expect a single combined training file. To reproduce our results, first merge the training datasets and save locally. Once merged, point `--dataset_root` to the resulting directory and run the corresponding training script.


## Evaluate HRPO

To assess HRPO on different datasets, run the corresponding evaluation script. For example, to evaluate on GSM8K:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py \
  --checkpoint_path PATH/TO/CHECKPOINT \
  --batch_size BATCH_SIZE \
  --greedy \
```

* `--checkpoint_path`: Path to the model checkpoint you wish to evaluate.
* `--batch_size`: (optional) Set the batch size for evaluation.
* `--greedy`: (optional) Greedy decoding in inference.

All evaluation outputs—including metrics and generated examples—will be written to the specified `checkpoint_path` directory.


## Requirements

* Python (>=3.11), PyTorch (>=2.4.1)
* Adapted versions of transformers, trl and unsloth are included.
* Other packages and dependencies can be found in `requirements.txt`.


## Citation
Please consider citing the following papers if you use our methods in your research:
```
@article{yue2025hybrid,
  title={Hybrid Latent Reasoning via Reinforcement Learning},
  author={Yue, Zhenrui and Jin, Bowen and Zeng, Huimin and Zhuang, Honglei and Qin, Zhen and Yoon, Jinsung and Shang, Lanyu and Han, Jiawei and Wang, Dong},
  journal={arXiv preprint arXiv:2505.18454},
  year={2025}
}
```