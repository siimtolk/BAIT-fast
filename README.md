# BAIT-fast

**A faster, locally runnable BAIT backdoor scanner.** BAIT-fast is a fork of [BAIT](https://github.com/SolidShen/BAIT) by Shen, Cheng et al. (IEEE S&P 2025, [paper](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf)). All credit for the method, the scanner and the model zoo belongs to the original authors; **if you use this code, please cite their paper** ([BibTeX below](#citation)). This fork changes how the scanner runs, not what it computes. The upstream repository has no license file at the time of writing, so check with the authors before redistributing.

## The story

We wanted to run BAIT as a routine check on fine-tuned 7-8B models on a home machine: two 16 GB consumer GPUs and a local LLM for the judge, no cloud API. Out of the box that did not work well:

- A single scan of an 8B model took about 1.5 to 3.6 hours, which made validating many models impractical.
- Current `transformers` releases broke the imports.
- The judge was hard-wired to the OpenAI API, and an 8B model in fp16 did not fit comfortably on one small card.

BAIT-fast fixes these. The inversion loop now re-uses the attention cache and decodes several candidate targets per forward pass, which gives **up to 5.3x** on the per-batch inversion (Qwen3-8B) with the same output; the judge can be any local OpenAI-compatible server; and 4-bit loading is available for small GPUs. Every addition is opt-in or behaviour-preserving, so results stay comparable with upstream.

## What is in this fork

| Change | Type | Switch |
|---|---|---|
| KV cache across inversion steps | speed | on by default |
| Fast warm-up (cache through warm-up, batched uncertainty inspections) | speed | `BAIT_FAST_WARMUP=1` |
| Candidate batching (K candidates per forward pass) | speed | `BAIT_CAND_BATCH=K`, e.g. 8 |
| Local judge: any OpenAI-compatible endpoint (vLLM, llama.cpp, ...) | feature | `BAIT_JUDGE_BASE_URL`, optional `BAIT_JUDGE_API_KEY_FILE`, `BAIT_JUDGE_MODEL_NAME`, `BAIT_JUDGE_EXTRA_BODY` (JSON merged into the request, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}`) |
| Save every inverted candidate to `candidates.jsonl` in the run folder: text, q, raw tokens with per-step probability, entropy, top-20 alternatives (`BAIT_POOL_TOPK`), rank, per-prompt-row probability, prompt token ids; plus `run_info.json` (code hash, settings) | feature | `BAIT_SAVE_POOL=1` |
| Custom judge prompt from a file (must contain `{response}` and keep the `<State>`/`<Reasoning>` answer format) | feature | `BAIT_JUDGE_PROMPT_FILE=prompt.txt` |
| 4-bit (NF4) model loading on a single GPU | feature | `BAIT_LOAD_4BIT=1` |
| `PaddingStrategy` / `TruncationStrategy` imports for newer `transformers` (v5) | bug fix | none |
| `scripts/bench_speedups.py`: timing and equivalence check, `--device auto` shards over all GPUs | tooling | |

The speed changes are in `src/core/detector.py`. The algorithm, hyperparameters and scoring are untouched. Differences from upstream: with candidate batching, scores can differ from the unbatched run by floating-point noise (the benchmark checks q within 0.01 and an identical inverted target), and the single-token `<|end_of_text|>` early-break is not replicated (it never occurs in Qwen's vocabulary). Fast warm-up keeps upstream's indexing quirk on purpose, so results stay comparable.

```bash
# everything on, judge on a local server
BAIT_FAST_WARMUP=1 BAIT_CAND_BATCH=8 \
BAIT_JUDGE_BASE_URL=http://127.0.0.1:8080/v1 BAIT_JUDGE_API_KEY_FILE=key.txt BAIT_JUDGE_MODEL_NAME=<model> \
python scripts/scan.py ...
```

### Benchmarks

`scripts/bench_speedups.py` times `scan_init_token` (warm-up plus full inversion, the per-batch unit of a scan; the judge is **not** included) on identical batches, once per configuration, and checks each result against the original code path. Hardware: RTX 5060 Ti 16 GB, fp16, Alpaca data. The 8B model is sharded over two cards (`--device auto`).

| Config | Qwen3-0.6B s/batch | speedup | Qwen3-8B s/batch | speedup |
|---|---|---|---|---|
| original (no cache) | 233.4 | 1.00x | 181.5 | 1.00x |
| + KV cache | 89.1 | 2.62x | 57.6 | 3.15x |
| + fast warm-up | 48.3 | 4.83x | 62.6 | 2.90x |
| + candidate batching K=8 | 78.7 | 2.97x | 34.0 | **5.33x** |

All configurations matched the original (q within 1e-2, same inverted target). Each cumulative row also includes the rows above it, except that the 0.6B warm-up and batching rows are not additive. Peak memory: 3.7-4.5 GiB (0.6B), 9.9 GiB per card (8B).

Caveats: 2 batches (0.6B) and 1 batch (8B) were timed, so treat the differences between the faster configs as indicative; the best config differs by model size (warm-up helps most at 0.6B, batching at 8B). Raw numbers are in `doc/benchmarks/`. End-to-end scan speedups will be somewhat lower because the judge calls are not accelerated.

---

# Original BAIT README

## 🎣 BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target

*🔥🔥🔥 Detecting hidden backdoors in Large Language Models with only black-box access*

**BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target** [[Paper]](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf) <br>
[Guangyu Shen*](https://www.cs.purdue.edu/homes/shen447/),
[Siyuan Cheng*](https://www.cs.purdue.edu/homes/cheng535/),
[Zhuo Zhang](https://www.cs.purdue.edu/homes/zhan3299/),
[Guanhong Tao](https://tao.aisec.world),
[Kaiyuan Zhang](https://kaiyuanzhang.com),
[Hanxi Guo](https://hanxiguo.me),
[Lu Yan](https://lunaryan.github.io),
[Xiaolong Jin](https://scholar.google.com/citations?user=w1-1dYwAAAAJ&hl=en),
[Shengwei An](https://www.cs.purdue.edu/homes/an93/),
[Shiqing Ma](https://people.cs.umass.edu/~shiqingma/),
[Xiangyu Zhang](https://www.cs.purdue.edu/homes/xyzhang/) (*Equal Contribution) <br>
Proceedings of the 46th IEEE Symposium on Security and Privacy (**S&P 2025**)

## News
- **[Jun 2, 2025]** We implement a new post-processing module to improve the detection stability. Find more details in [Update](doc/UPDATE.md).
- **[May 29, 2025]** The model zoo is now available on [Huggingface](https://huggingface.co/NoahShen/BAIT-ModelZoo).
- 🎉🎉🎉  **[Nov 10, 2024]** BAIT won the third place (with the highest recall score) and the most efficient method in the [The Competition for LLM and Agent Safety 2024 (CLAS 2024) - Backdoor Trigger Recovery for Models Track](https://www.llmagentsafetycomp24.com/leaderboards/) ! The competition version of BAIT will be released soon.

## Contents
- [🎣 BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target](#-bait-large-language-model-backdoor-scanning-by-inverting-attack-target)
  - [News](#news)
  - [Contents](#contents)
  - [Preparation](#preparation)
  - [Model Zoo](#model-zoo)
  - [LLM Backdoor Scanning](#llm-backdoor-scanning)
  - [Evaluation](#evaluation)
  - [Citation](#citation)
  - [Contact](#contact)


## Preparation

1. Clone this repository
```bash
git clone https://github.com/noahshen/BAIT.git
cd BAIT
```

2. Install Package
```Shell
conda create -n bait python=3.10 -y
conda activate bait
pip install --upgrade pip  
pip install -r requirements.txt
```

3. Install BAIT CLI Tool
```Shell
pip install -e .
```

4. Add OpenAI API Key
```Shell
export OPENAI_API_KEY=<your_openai_api_key>
```

5. Login to Huggingface
```Shell
huggingface-cli login
```

6. Download Model Zoo
```Shell  
huggingface-cli download NoahShen/BAIT-ModelZoo --local-dir ./model_zoo
```


## Model Zoo

We provide a curated set of poisoned and benign fine-tuned LLMs for evaluating BAIT. These models can be downloaded from [Huggingface](https://huggingface.co/NoahShen/BAIT-ModelZoo). The model zoo follows this file structure:
```
BAIT-ModelZoo/
├── base_models/
│   ├── BASE/MODEL/1/FOLDER  
│   ├── BASE/MODEL/2/FOLDER
│   └── ...
├── models/
│   ├── id-0001/
│   │   ├── model/
│   │   │   └── ...
│   │   └── config.json
│   ├── id-0002/
│   └── ...
└── METADATA.csv
```
```base_models``` stores pretrained LLMs downloaded from Huggingface. We evaluate BAIT on the following 3 LLM architectures:

- [Llama-2-7B-chat-hf](meta-llama/Llama-2-7b-chat-hf)
- [Llama-3-8B-Instruct](meta-llama/Meta-Llama-3-8B-Instruct)
- [Mistral-7B-Instruct-v0.2](mistralai/Mistral-7B-Instruct-v0.2)

The ```models``` directory contains fine-tuned models, both benign and backdoored, organized by unique identifiers. Each model folder includes:

- The model files
- A ```config.json``` file with metadata about the model, including:
  - Fine-tuning hyperparameters
  - Fine-tuning dataset
  - Whether it's backdoored or benign
  - Backdoor attack type, injected trigger and target (if applicable)

The ```METADATA.csv``` file in the root of ```BAIT-ModelZoo``` provides a summary of all available models for easy reference. Current model zoo contains 91 models. We will keep updating the model zoo with new models.

## LLM Backdoor Scanning

To perform BAIT on the entire model zoo, run the CLI tool:
```bash
bait-scan --model-zoo-dir /path/to/model/zoo --data /path/to/data --cache-dir /path/to/model/zoo/base_models/ --output-dir /path/to/results --run-name your-experiment-name
```

To specify which GPUs to use, set the `CUDA_VISIBLE_DEVICES` environment variable:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bait-scan --model-zoo-dir /path/to/model/zoo --data /path/to/data --cache-dir /path/to/model/zoo/base_models/ --output-dir /path/to/results --run-name your-experiment-name
```

This script will iteratively scan each individual model stored in the model zoo directory. When multiple GPUs are specified, BAIT will launch parallel scans for multiple models simultaneously - if you specify n GPUs, it will scan n models in parallel. The intermediate logs and final results will be stored in the specified output directory.

## Evaluation

To evaluate the BAIT scanning results:

1. Run the evaluation CLI tool:

```bash
bait-eval --run-dir your-experiment-name
```

This script will run evaluation and generate a comprehensive report on key metrics such as detection rate, false positive rate, and accuracy for the backdoor detection.

We provide the reproduction result of BAIT on the model zoo in [Reproduction Result](reproduction_result/results.md). The experiment is conducted on 8 A6000 GPUs with 48G memory.




## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@INPROCEEDINGS {,
author = { Shen, Guangyu and Cheng, Siyuan and Zhang, Zhuo and Tao, Guanhong and Zhang, Kaiyuan and Guo, Hanxi and Yan, Lu and Jin, Xiaolong and An, Shengwei and Ma, Shiqing and Zhang, Xiangyu },
booktitle = { 2025 IEEE Symposium on Security and Privacy (SP) },
title = {{ BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target }},
year = {2025},
volume = {},
ISSN = {2375-1207},
pages = {1676-1694},
abstract = { Recent literature has shown that LLMs are vulnerable to backdoor attacks, where malicious attackers inject a secret token sequence (i.e., trigger) into training prompts and enforce their responses to include a specific target sequence. Unlike discriminative NLP models, which have a finite output space (e.g., those in sentiment analysis), LLMs are generative models, and their output space grows exponentially with the length of response, thereby posing significant challenges to existing backdoor detection techniques, such as trigger inversion. In this paper, we conduct a theoretical analysis of the LLM backdoor learning process under specific assumptions, revealing that the autoregressive training paradigm in causal language models inherently induces strong causal relationships among tokens in backdoor targets. We hence develop a novel LLM backdoor scanning technique, BAIT (Large Language Model Backdoor ScAnning by Inverting Attack Target). Instead of inverting back- door triggers like in existing scanning techniques for non-LLMs, BAIT determines if a model is backdoored by inverting back- door targets, leveraging the exceptionally strong causal relations among target tokens. BAIT substantially reduces the search space and effectively identifies backdoors without requiring any prior knowledge about triggers or targets. The search-based nature also enables BAIT to scan LLMs with only the black-box access. Evaluations on 153 LLMs with 8 architectures across 6 distinct attack types demonstrate that our method outperforms 5 baselines. Its superior performance allows us to rank at the top of the leaderboard in the LLM round of the TrojAI competition (a multi-year, multi-round backdoor scanning competition). },
keywords = {ai security;backdoor scanning;large language model},
doi = {10.1109/SP61157.2025.00103},
url = {https://doi.ieeecomputersociety.org/10.1109/SP61157.2025.00103},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =May}

```

## Contact

For any questions or feedback, please contact Guangyu Shen at [shen447@purdue.edu](mailto:shen447@purdue.edu).




