import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import logging
import os 

import dotenv
dotenv.load_dotenv()

from data import PreprocessorAkkadian
from utils import setup_logging, clear_memory

logger = setup_logging()

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: PreprocessorAkkadian, prefix: str):
        self.prefix = prefix
        self.preprocessor = preprocessor
        
        logger.info("Preprocessing test transliterations...")
        raw_texts = df["transliteration"].astype(str).tolist()
        
        # Process in batch to leverage our optimized Preprocessor
        clean_texts = self.preprocessor.preprocess_batch(raw_texts)
        self.texts = [f"{self.prefix}{t}" for t in clean_texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def run_inference(model, tokenizer, test_df, config):
    """
    Orchestrates the translation process with high efficiency.
    """
    # 1. Setup Preprocessor
    preprocessor = PreprocessorAkkadian(aggressive=config.aggressive_preprocessing)
    prefix = "translate Akkadian to English: " # Ensure this matches training exactly
    
    # 2. Prepare DataLoader
    ds = InferenceDataset(test_df, preprocessor, prefix)
    
    def collate_fn(batch):
        return tokenizer(
            batch,
            max_length=config.max_src_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # 3. Model Logic
    model.eval()
    device = config.device
    preds = []

    logger.info(f"Starting inference on {len(ds)} samples...")
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Translating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Optimized Generation
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_tgt_length,
                num_beams=config.generation_num_beams,
                early_stopping=True,
                use_cache=True 
            )
            
            # Decode and Clean
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend([p.strip() for p in batch_preds])

    # 4. Create Submission
    submission = pd.DataFrame({
        "id": test_df["id"], 
        "translation": preds
    })
    
    output_path = os.getenv("OUTPUT_PATH")
    if not output_path:
        output_path = "submission.csv"  # Fallback
        logger.warning("OUTPUT_PATH not set, using default 'submission.csv'")
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    
    return submission
