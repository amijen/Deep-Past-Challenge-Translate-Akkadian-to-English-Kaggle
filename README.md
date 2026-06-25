# Akkadian to English Translation Pipeline

An end-to-end, optimized sequence-to-sequence (Seq2Seq) pipeline for translating ancient Akkadian transliterations into English. This project features aggressive data cleaning, memory-optimized PyTorch datasets, and a custom PyTorch/Transformers training loop.
---

## 🧠 Architecture Choice: Why ByT5?

Translating ancient languages presents unique tokenization challenges. We opted to use **ByT5** (a byte-level variant of the T5 architecture) rather than standard subword models for several critical reasons:

Akkadian transliterations do not use standard Latin alphabets. The transcription system relies heavily on:
* **Diacritical marks** denoting specific phonemes (e.g., `š`, `ṣ`, `ṭ`, `ḫ`).
* **Subscript and Index numbers** denoting distinct cuneiform signs that share the same phonetic value (e.g., `u₂`, `e₃`, `xₓ`).
* **Unicode fractions** used extensively in ancient administrative and economic texts (e.g., `⅓`, `⅔`, `⅚`).
---

---
## ⚙️ System Architecture

The codebase is structured into a modular, object-oriented pipeline designed for high-throughput processing and stability.

### 1. Preprocessing Layer (`data/preprocessing.py`)
Because Akkadian data is highly noisy, preprocessing is handled by two dedicated engines (`PreprocessorAkkadian` and `PreprocessorEnglish`). 
* **Regex Rule Engine**: Uses pre-compiled regular expressions to handle repeated words, annotations, and formatting noise.
* **Fraction Standardization**: Automatically detects decimal floats and converts them into standardized Unicode fractions up to 1/8th limits.
* **Gap Normalization**: Standardizes breaks in the ancient tablets (e.g., `[x]`, `(break)`) into a universal `<gap>` token.
### 2. Orchestration & Validation Layer (`train_inference/train.py`)
Handled by the `AkkadianDataProcessor`, this layer ensures data integrity before it reaches the model:
* **Cross-Column Gap Consistency**: A custom validation step checks if a `<gap>` or `<big_gap>` exists in the source text but not the translation (or vice versa). If they do not match, the unmatched gaps are dynamically removed. This prevents the model from hallucinating breaks that don't exist linguistically.
* **Length Filtering**: Discards sequences that are too short to provide meaningful context.

### 3. Training Engine (`train_inference/train.py`)
Orchestrated by the `TranslationPipeline` class:
* **Dynamic Padding**: The custom `TranslationDataset` tokenizes inputs without hard padding, allowing the `DataCollatorForSeq2Seq` to dynamically pad batches to the longest sequence in that specific batch, saving VRAM.
* **Custom Metric Evaluation**: Evaluates model performance using a geometric mean of **SacreBLEU** and **chrF** (word_order=2) metrics, ensuring the model is rewarded for both exact word matches and morphological character-level accuracy.
* **Resource Optimization**: Implements gradient accumulation, `fp16` mixed precision, and gradient checkpointing to allow ByT5 to fit into standard GPU memory limits.

### 4. Inference Engine (`train_inference/inference.py`)
Designed for competition-scale test sets:
* **Batched DataLoader**: Feeds data to the GPU in optimized chunks.
* **`torch.inference_mode()`**: Utilizes modern PyTorch evaluation states to strip out auto-grad overhead.
* **Beam Search Generation**: Uses `num_beams=4` and KV-caching (`use_cache=True`) to generate high-quality translations efficiently.
---

## 📂 Project Structure

```text
akkadian/
├── data/
│   ├── dpc_data/            # Raw CSV data (ignored by git)
│   ├── __init__.py          
│   └── preprocessing.py     # Native Python regex optimizations 
├── train_inference/
│   ├── __init__.py          
│   ├── inference.py         # Batch-optimized translation generation
│   └── train.py             # Trainer, Datasets, and Config
├── utils/
│   ├── __init__.py          
│   └── utilities.py         # Logging, seeding, memory mgmt
├── .env                     # Secrets / Local environment variables
├── .gitignore
├── main.py                  # Execution entry point
└── README.md