import os
import pandas as pd
import torch

# 1. SET ENVIRONMENT VARIABLES FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from train_inference import ModelConfig, TranslationPipeline, run_inference
from utils.utilities import setup_logging, clear_memory

logger = setup_logging()

def main():
    try:
        # 2. INITIALIZE CONFIG
        config = ModelConfig()
        
        # 3. RUN TRAINING PIPELINE
        pipeline = TranslationPipeline(config)
        pipeline.train()
        
        # 4. PREPARE TEST DATA FOR INFERENCE
        logger.info("Loading test data for inference...")
        test_csv_path = os.path.join(config.input_dir, "test.csv")
        
        if not os.path.exists(test_csv_path):
            logger.warning(f"Test file not found at {test_csv_path}. Skipping inference.")
            return

        test_df = pd.read_csv(test_csv_path)

        # 5. RUN INFERENCE
        logger.info("Starting inference phase...")
        submission_df = run_inference(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            test_df=test_df,
            config=config
        )
        
        logger.info("Pipeline and Inference completed successfully.")
        print(submission_df.head())

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
    finally:
        clear_memory()

if __name__ == "__main__":
    main()
