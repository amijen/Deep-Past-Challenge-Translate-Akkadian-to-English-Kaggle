import os
import re
import math
import numpy as np 
import shutil
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from tqdm.auto import tqdm
import sacrebleu

import dotenv
dotenv.load_dotenv()

from utils import setup_logging, seed_everything, clear_memory
from data import PreprocessorAkkadian, PreprocessorEnglish


logger = setup_logging()

@dataclass 
class ModelConfig:
    """
    Configuration class for the model and training pipeline.
    """
    # Paths
    model_path = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    input_dir: str = "data/dpc_data"
    output_dir : str = "trained_model"

    # Data Preprocessing 
    max_src_length : int = 512
    max_tgt_length : int = 256 
    aggressive_preprocessing: bool = True

    # Training Hyperparameters 
    batch_size : int = 16 
    learning_rate : float = 1e-4 
    epochs : int = 5 
    gradient_accumulation_steps : int = 8
    warmup_ratio  : float = 0.2
    max_grad_norm: float = 0.5
    seed : int = 42 

    # Generation 
    generation_max_length : int = 256
    generation_num_beams : int = 4

    # Hardware 
    device : str = field(init = False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.model_path:
            raise ValueError("MODEL_PATH environment variable is not set!")
        os.makedirs(self.output_dir, exist_ok=True)
        if self.eval_strategy == "no":
            self.load_best = False

class AkkadianDataProcessor: 
    """
    Handles loading, cleaning, and preprocessing of the dataset.
    """
    def __init__(self, config: ModelConfig):
        self.config = config 
        self.preprocessor_akk = PreprocessorAkkadian(aggressive = config.aggressive_preprocessing)
        self.preprocessor_eng = PreprocessorEnglish(aggressive = config.aggressive_preprocessing)

    def _preprocess_row(self, row: pd.Series) -> Optional[Dict[str, str]]:
        src_text = row.get('transliteration', '')
        tgt_text = row.get('translation', '')

        src = self.preprocessor_akk.preprocess_batch([src_text])
        tgt = self.preprocessor_eng.preprocess_batch([tgt_text])
        
        src_ = src[0] if src else ""
        tgt_ = tgt[0] if tgt else ""
        
        if len(src_) > 2 and len(tgt_) > 2:
            return {'src': src_, 'tgt': tgt_}
        return None
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Loads raw CSVs and processes them into a clean DataFrame.
        """
        logger.info("Loading raw data files...")
        train_path = os.path.join(self.config.input_dir, "train.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_df = pd.read_csv(train_path)

        logger.info("Processing training pairs...")

        pairs = [
            res for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="Preprocessing")
            if (res := self._preprocess_row(r)) is not None
        ]

        pairs_df = pd.DataFrame(pairs).drop_duplicates()
        logger.info(f"Initial training pairs after deduplication: {len(pairs_df)}")

        # Apply Gap Consistency Logic
        pairs_df = self._clean_gaps(pairs_df)

        return pairs_df
    
    def _clean_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes inconsistent gap markers between source and target."""
        logger.info("Applying gap consistency checks...")

        def check_gap_consistency(row):
            src, tgt = str(row['src']), str(row['tgt'])
            gap_src, gap_tgt = '<gap>' in src, '<gap>' in tgt
            big_gap_src, big_gap_tgt = '<big_gap>' in src, '<big_gap>' in tgt
            return (gap_src == gap_tgt) and (big_gap_src == big_gap_tgt)

        def remove_unmatched_gaps(row):
            src, tgt = str(row['src']), str(row['tgt'])
            
            # Helper for cleaning
            def clean_text(text, marker, condition):
                if condition:
                    text = re.sub(r'\s+', ' ', text.replace(marker, '')).strip()
                return text

            # Logic for <gap>
            gap_src, gap_tgt = '<gap>' in src, '<gap>' in tgt
            src = clean_text(src, '<gap>', gap_src and not gap_tgt)
            tgt = clean_text(tgt, '<gap>', gap_tgt and not gap_src)
            
            # Logic for <big_gap>
            big_gap_src, big_gap_tgt = '<big_gap>' in src, '<big_gap>' in tgt
            src = clean_text(src, '<big_gap>', big_gap_src and not big_gap_tgt)
            tgt = clean_text(tgt, '<big_gap>', big_gap_tgt and not big_gap_src)
            
            return pd.Series({'src': src, 'tgt': tgt})
        # Apply cleaning
        df[['src', 'tgt']] = df.apply(remove_unmatched_gaps, axis=1)

        # Verification
        consistent = df.apply(check_gap_consistency, axis=1).all()
        if not consistent:
            logger.warning("Gap consistency check failed for some rows. Investigate data cleaning logic.")
        else:
            logger.info("Gap consistency verified successfully.")
            
        return df
    
class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, config: ModelConfig, prefix: str = "translate Akkadian to English: "):
        self.tokenizer = tokenizer
        self.config = config
        self.prefix = prefix 
        self.samples = []

        logger.info("Building Dataset samples...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing Data"):
            src, tgt = row['src'], row['tgt']
            if len(src) < 10 or len(tgt) < 5 :
                continue 
            inputs = self.tokenizer(
                f"{self.prefix}{src}",
                max_length = self.config.max_src_length,
                truncation = True,
                padding = False
            )
            targets = self.tokenizer(
                text_target = tgt,
                max_length = self.config.max_tgt_length, 
                truncation = True,
                padding = False
            )
            inputs['labels'] = targets['input_ids']
            self.samples.append(inputs)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    

def compute_metrics_factory(tokenizer: AutoTokenizer):

    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds 
        # Handle tuple output (logits, past_key_values, ...)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions 
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)

        # Replace -100 in labels (ignore index) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-process text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute metrics
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)

        final_score = math.sqrt(bleu.score * chrf.score)

        return {
            "score": final_score,
            "bleu": bleu.score,
            "chrf": chrf.score
        }
    
    return compute_metrics

class TranslationPipeline: 
    def __init__(self, config : ModelConfig):
        self.config = config 
        seed_everything(config.seed, logger)
        clear_memory()

        # Check Disk Space
        self._check_disk_space()

        # Load Tokenizer & Model 
        self._load_model_components()

    def _check_disk_space(self):
        try: 
            total, used, free = shutil.disk_usage(os.getcwd())
            logger.info(f"Disk Space: {free // (1024**3)} GB free")
        except Exception as e : 
            logger.warning(f"Could not check disk space: {e}")

    def _load_model_components(self):
        """Loads Tokenizer and Model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path, 
                local_files_only=True
            )
        except OSError:
            logger.error("Model files not found locally. Check path or set local_files_only=False.")
            raise

        self.model.gradient_checkpointing_enable()
        self.model.to(self.config.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded. Parameters: {param_count:,}")

    def prepare_datasets(self) -> Tuple[TranslationDataset, TranslationDataset]:
        """
        Prepares train and validation datasets.
        """
        processor = AkkadianDataProcessor(self.config)
        pairs_df = processor.load_and_preprocess_data()

        pairs_df = pairs_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        val_n = max(50, int(len(pairs_df) * 0.1))
        logger.info("Creating validation dataset...")
        val_ds = TranslationDataset(pairs_df.iloc[:val_n], self.tokenizer, self.config)
        logger.info("Creating training dataset...")
        train_ds = TranslationDataset(pairs_df.iloc[val_n:], self.tokenizer, self.config)
        return train_ds, val_ds
    
    def train(self):
        train_ds, val_ds = self.prepare_datasets()

        # Clean up old checkpoints
        if os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir)
            logger.info("Cleaned up old checkpoints.")

        training_args = Seq2SeqTrainingArguments(
            output_dir = self.config.output_dir,
            eval_strategy = "steps",
            save_strategy = "steps",
            save_steps = 500,
            logging_steps = 5,
            learning_rate = self.config.learning_rate,
            per_device_train_batch_size = self.config.batch_size,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            num_train_epochs = self.config.epochs,
            warmup_ratio = self.config.warmup_ratio,
            fp16=(self.config.device == "cuda"), # Enable mixed precision automatically for GPU
            save_total_limit=1,
            report_to="none",
            save_only_model=True,
            max_grad_norm=self.config.max_grad_norm,
            predict_with_generate=True,
            generation_max_length=self.config.generation_max_length,
            generation_num_beams=self.config.generation_num_beams,
            load_best_model_at_end=True, # Added for better observability/best checkpoint
            metric_for_best_model="score" # Track our custom score
        )

        trainer = Seq2SeqTrainer(
            model = self.model,
            args = training_args,
            train_dataset = train_ds, 
            eval_dataset = val_ds, 
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, model = self.model),
            compute_metrics=compute_metrics_factory(self.tokenizer),
        )
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")
        self._save_final_artifacts(trainer)
    
    def _save_final_artifacts(self, trainer: Seq2SeqTrainer):
        """Save the final model and logs."""
        final_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Final model saved to {final_path}")