# STRIDE: Strategic Iterative Decision-Making for Retrieval-Augmented Multi-Hop Question Answering

🌟 This is the official code of the paper **STRIDE: Strategic Iterative Decision-Making for Retrieval-Augmented Multi-Hop Question Answering** (SIGIR 2026 Full Paper).

🔗 Paper Link: [STRIDE](https://arxiv.org/abs/2604.17405)

📖 STRIDE is a hierarchical framework for multi-hop question answering with retrieval. It decomposes reasoning into a **Meta-Planner** (strategy), a **Supervisor** with **Extractor** and **Reasoner** modules (control and execution), and an optional **Fallback Reasoner** for cases where the main run does not yield an answer. **STRIDE-FT** refers to modular fine-tuning (supervised LoRA and DPO) built from execution trajectories.

## Installation

```bash
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
python -m pipeline --help
```

## Layout

| Component | Module |
|-----------|--------|
| Meta-Planner | `meta_planer.py` |
| Supervisor, Extractor, Reasoner | `supervisor.py` |
| Fallback Reasoner | `fallback_qa.py` |
| Dense retriever (Contriever, FAISS) | `contriever_model.py`, `my_retriever.py` |
| Build corpus FAISS index (CLI) | `build_corpus_index.py` |
| Prompts (defaults only) | `prompt/` — see `prompt/README.md` |
| End-to-end driver | `pipeline.py` |
| Metrics | `run_eval.py`, `metrics.py`, `utils.py` |
| Fine-tuning | `lora_ft.py`, `lora_dpo.py`, `ft_preprocess.py` |
| SFT jsonl from trajectories | `build_ft_dataset.py` |
| vLLM LoRA helpers | `vllm_lora.py` |
| Train/test split helpers | `data_prep.py` |
| Indexing notebook (optional) | `notebooks/build_index.ipynb` |

By default, runs write next to the code modules: **`meta_plans/<run_name>/`**, **`output/<run_name>/`**, **`faiss_index/`** (unless paths are overridden).

## Data and run layout

- **`--input_jsonl`** — Path to your split (train, dev, or test). Each line should include at least **`id`** and **`question`** in the format expected by the rest of the pipeline. The **same file** is read by the Meta-Planner and the Supervisor in `pipeline`.
- **`--run_name`** — Names one run directory under `meta_plans/` and `output/`. Default: sanitized stem of `input_jsonl` (for example `/data/hotpotqa/dev.jsonl` → `dev`).
- **`--index_corpus`** — Optional. Used with **`--faiss_index_path`**: the substring **`dataset`** in that path is replaced by this value so the resolved directory matches your indexed corpus (see Retrieval). Defaults to **`run_name`** if omitted.

There is no extra data-mode switch: you choose which jsonl to pass.

## Retrieval

Retrieval is **not** used in the Meta-Planner. It runs inside **`supervisor`** (when the plan uses `retrieve` / `rewrite` actions) and inside **`fallback_qa`** (extra passages for questions with no main answer). Both call **`DenseRetriever.load_index`**, which expects exactly the layout produced by **`DenseRetriever.save_index`** (or by **`python -m build_corpus_index`** below).

### On-disk index layout (must match inference)

One directory, two artifacts:

| File | Content |
|------|---------|
| **`faiss.index`** | FAISS `IndexFlatIP` over **L2-normalized** passage embeddings (same encoder as `--retriever_model_path`). |
| **`document.vecstore.npz`** | NumPy archive with arrays **`documents`** (passage body text), **`titles`** (string per passage), **`embeddings`** (one vector per passage). |

At query time, **`retrieve` / `batch_retrieve`** return dicts **`{text, title, score}`** where **`text`** comes from **`documents[idx]`** and **`title`** from **`titles[idx]`**. So whatever you store at build time is what prompts see later.

**`add_doc(document_text, title)`** stores **`document_text`** as the body and **`title`** as the title (same order as `add_docs(document_texts, titles)`).

### Building an index (recommended CLI)

```bash
python -m build_corpus_index \
  --input_jsonl /path/to/train_or_dev.jsonl \
  --output_dir faiss_index/hotpotqa/index \
  --format stride_contexts \
  --retriever_model_path facebook/contriever
```

- **`--format stride_contexts`** — each jsonl line is a STRIDE object with **`contexts`** (and optionally **`pinned_contexts`**): entries use **`title`** and **`paragraph_text`** (same structure as QA jsonl in this codebase). Passages are **deduped** across the file by content hash unless you pass **`--no_dedupe`**.
- **`--format records`** — each line is a standalone record with **`title`** and **`text`** (or **`paragraph_text`**).

Use the **same** **`--retriever_model_path`** when running **`pipeline`** / **`supervisor`**.

Implementation: **`build_corpus_index.py`** (calls **`add_docs`** then **`save_index`** — identical stack to a manual script).

Optional: **`notebooks/build_index.ipynb`** for a tiny interactive smoke test.

### How `pipeline` passes retrieval settings

`python -m pipeline` forwards these arguments to **`supervisor`** and **`fallback_qa`** (not to the Meta-Planner):

| Flag | Role |
|------|------|
| **`--retriever_model_path`** | Hugging Face id or local path for the dense encoder (default `facebook/contriever`). |
| **`--faiss_index_path`** | Template path containing the literal substring **`dataset`**, replaced by **`--index_corpus`** (or by `run_name` when `index_corpus` is omitted). Default: `faiss_index/dataset/index` (under the code directory). |
| **`--index_corpus`** | Corpus name substituted for `dataset` in `faiss_index_path` (e.g. `hotpotqa` → `faiss_index/hotpotqa/index`). |
| **`--top_k_docs`** | Top-k passages per retrieval query in the Supervisor. |
| **`--top_k_docs_fallback`** | Top-k for the Fallback Reasoner. |

If your index lives at a path **without** the placeholder word `dataset`, set **`--faiss_index_path`** to that full path; the string replace is a no-op when `dataset` does not appear.

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1

python -m pipeline \
  --input_jsonl /path/to/your_split.jsonl \
  --model_path /path/to/generative/model \
  --tensor_parallel_size 2 \
  --faiss_index_path faiss_index/dataset/index \
  --index_corpus hotpotqa \
  --retriever_model_path facebook/contriever \
  --top_k_docs 5 \
  --run_fallback \
  --top_k_docs_fallback 5
```

- **`--run_fallback`** runs the Fallback Reasoner on examples with no main answer (uses the same FAISS index and retriever path as above).
- **`--skip_meta`** / **`--skip_supervisor`** resume from existing jsonl under `meta_plans/<run_name>/` and `output/<run_name>/`.
- Override file names with **`--meta_write_name`**, **`--supervisor_write_name`**, prompt file flags, etc., if needed.

You can also run **`meta_planer`**, **`supervisor`**, or **`fallback_qa`** directly (`python -m …`); they expose the same retrieval flags where applicable.

## Fine-tuning (STRIDE-FT)

Four modules: **Meta-Planner** (DPO), **Supervisor**, **Extractor**, **Reasoner** (SFT for the last three). `build_ft_dataset` builds jsonl; `ft_preprocess` + `lora_ft` train SFT; `lora_dpo` trains DPO.

**Extractor:** `extractor-intermediate` mines rows from logs; `extractor-sft` adds Contriever + FAISS and writes SFT triples. Optional minimization: add `minimized_context`, then `--context_field minimized_context`.

**Meta DPO:** run Meta-Planner + Supervisor **K** times on the same questions (e.g. 8 seeds). `meta-dpo` reads **K** aligned supervisor + plan jsonl paths and emits `prompt` / `chosen` / `rejected`.

```bash
python -m build_ft_dataset reasoner \
  --input_jsonl output/<run>/<supervisor>.jsonl \
  --output_jsonl ft_data/reasoner/train.jsonl \
  --max_examples 5000

python -m build_ft_dataset extractor-intermediate \
  --input_jsonl output/<run>/<supervisor>.jsonl \
  --corpus_name hotpotqa \
  --output_jsonl ft_data/extractor/intermediate_hotpot.jsonl \
  --max_examples 5000

python -m build_ft_dataset extractor-sft \
  --input_jsonl ft_data/extractor/intermediate_hotpot.jsonl \
  --output_jsonl ft_data/extractor/sft_hotpot.jsonl \
  --retriever_model_path facebook/contriever \
  --faiss_index_pattern faiss_index/{dataset}/index \
  --context_field context \
  --max_examples 5000

python -m build_ft_dataset supervisor \
  --run output/<run>/<supervisor>.jsonl meta_plans/<run>/<plan>.jsonl \
  --output_jsonl ft_data/supervisor/train.jsonl \
  --max_examples 5000

python -m build_ft_dataset meta-dpo \
  --supervisor_traj output/.../1/<supervisor>.jsonl \
  --supervisor_traj output/.../2/<supervisor>.jsonl \
  --meta_plan_traj meta_plans/.../1/<plan>.jsonl \
  --meta_plan_traj meta_plans/.../2/<plan>.jsonl \
  --output_jsonl ft_data/meta/dpo.jsonl \
  --max_examples 5000
```


**Model path at each step:**

1. **`ft_preprocess`** — `--model_path` for the tokenizer used to build the supervised dataset.
2. **`lora_ft`** — `--model_path` is the base causal LM; adapters go to `--output_dir` (default `ft_models/reasoner/` under the code directory).
3. **`lora_dpo`** — `--model_path` is the base or SFT checkpoint; `--data_path` is a jsonl with `prompt`, `chosen`, `rejected`.

```bash
python -m ft_preprocess \
  --input_jsonl /path/to/train_shard.jsonl \
  --model_path /path/to/base/model

python -m lora_ft \
  --model_path /path/to/base/model \
  --data_path /path/to/preprocessed_dataset \
  --output_dir ft_models/reasoner

python -m lora_dpo \
  --model_path /path/to/base/or/sft/model \
  --data_path /path/to/dpo.jsonl \
  --output_dir ft_models/dpo
```

**Hyperparameters used for the reported Qwen3-8B LoRA adapters** (override `lora_ft` defaults with `--lr`, `--epoch`, `--batch_size`, and set `--output_dir` per module):

| Module | Learning rate | Epochs | Per-device batch size | Last saved step (`global_step`) |
|--------|---------------|--------|-------------------------|----------------------------------|
| Meta-Planner | `3e-5` | 3 | 4 | 90 |
| Supervisor | `1e-4` | 3 | 2 | 42 |
| Extractor | `1e-4` | 2 | 2 | 152 |
| Reasoner | `1e-4` | 2 | 2 | 140 |

**LoRA layout** matches `lora_ft.py`: **`r=8`**, **`lora_alpha=32`**, **`lora_dropout=0.1`**, targets **`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`**. The script still uses **`gradient_accumulation_steps=8`**; total optimizer steps depend on dataset size—pick `--epoch` / batch settings so the run matches the step counts you want.

**Training data:** `build_ft_dataset` writes SFT jsonl (`instruction` / `input` / `output`) or DPO jsonl (`prompt` / `chosen` / `rejected`); `ft_preprocess` only tokenizes the SFT format for `lora_ft`.

### vLLM inference with multiple LoRAs (four modules)

After training **separate** PEFT adapters for Meta-Planner, Supervisor, Extractor, and Reasoner, keep **one** base checkpoint on **`--model_path`**. vLLM is started with `enable_lora=True`; each `generate` call passes a `LoRARequest` for the active module. **The Fallback Reasoner always uses the base weights** (no separate LoRA).

| Flag | Module |
|------|--------|
| `--lora_meta` | Meta-Planner |
| `--lora_supervisor` | Supervisor |
| `--lora_extractor` | Extractor |
| `--lora_reasoner` | Reasoner |

Tune **`--max_lora_rank`** and **`--max_loras`** when loading several adapters or large ranks (should be consistent with training).

```bash
python -m pipeline \
  --input_jsonl /path/to/data.jsonl \
  --model_path /path/to/base_model \
  --faiss_index_path faiss_index/dataset/index \
  --index_corpus hotpotqa \
  --lora_meta /path/to/meta_lora \
  --lora_supervisor /path/to/s_lora \
  --lora_extractor /path/to/e_lora \
  --lora_reasoner /path/to/r_lora
```

Omit a `--lora_*` flag to use **base** weights for that stage. The same LoRA flags work on `meta_planer`, `supervisor`, and `fallback_qa` when invoked with `python -m …`.

## Data sources

Corpora follow **HotpotQA**, **2WikiMultihopQA**, and **MuSiQue**-style dumps in the spirit of **[IRCoT](https://github.com/stonybrooknlp/ircot)** (multi-passage QA jsonl). Thanks to the IRCoT team for releasing those datasets.

We subsample **10,000** train questions per corpus (`data_prep sample_train`); the paper then used only the **first 5,000** of that file for mining—matching the `--max_examples 5000` flags in the Fine-tuning block (`0` = use all rows).

### Dataset splits

1. If needed, convert upstream jsonl to this repo’s QA format (`data_prep.py`, `convert_upstream_to_stride`).
2. **Train subsample** (10k, fixed seed):
   ```bash
   python -m data_prep sample_train \
     --upstream_train /path/to/full_train.jsonl \
     --output processed_data/train_subsampled_10k.jsonl \
     --n 10000 --seed 42
   ```
3. **Extended test** — fixed dev/test slice plus extras from full train, excluding ids in the 10k train file:
   ```bash
   python -m data_prep merge_test \
     --base_test processed_data/hotpotqa/test.jsonl \
     --upstream_train /path/to/full_train.jsonl \
     --train_sample processed_data/train_subsampled_10k.jsonl \
     --output processed_data/hotpotqa/test_1000.jsonl \
     --extra_n 500 --seed 42
   ```
4. Optional: `python -m data_prep check_overlap processed_data/train_subsampled_10k.jsonl processed_data/hotpotqa/test_1000.jsonl`
5. Build `faiss_index/` with `build_corpus_index` (see **Retrieval**).

## Evaluation

Run **`python -m run_eval`** on the Supervisor (or merged) prediction jsonl under `output/<run_name>/...`:

```bash
python -m run_eval output/<run_name>/<plan_version_dir>/stride_top5.jsonl
```

Optional merge with fallback output by example id:

```bash
python -m run_eval output/<run_name>/plan/stride_top5.jsonl \
  --fallback-jsonl output/<run_name>/plan/plan-stride_top5-fallback_qa.jsonl \
  --json-out metrics.json
```

(Exact fallback filename depends on `--used_result_file` and `--fallback_write_name`.)

Reported metrics: **EM**, **F1**, **precision**, **recall**.

## Reference

If you find this repository or paper useful, you can cite:

```
@misc{chen2026stride,
      title={STRIDE: Strategic Iterative Decision-Making for Retrieval-Augmented Multi-Hop Question Answering}, 
      author={Wei Chen and Lili Zhao and Zhi Zheng and HuiJun Hou and Tong Xu},
      year={2026},
      eprint={2604.17405},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.17405}, 
}
