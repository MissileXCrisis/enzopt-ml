import os
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
from src.utils.logger import setup_logger

# Initialize structured logging
logger = setup_logger("Phase1_Design")

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Safely parses the central YAML configuration parameters."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at path: {config_path}")
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_zero_shot_inference(config: dict) -> pd.DataFrame:
    """
    Executes wild-type marginal zero-shot variant effect prediction using ESM-2.
    Calculates Log-Likelihood Ratios (LLR) across all 20 standard amino acids.
    """
    phase_cfg = config["phase1_design"]
    model_name = phase_cfg["model_name"]
    wt_seq = phase_cfg["wild_type_sequence"]
    target_name = phase_cfg["target_name"]
    
    logger.info(f"Initializing pre-trained ESM-2 architecture: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    
    # Force evaluation mode and disable gradient calculations for rapid inference
    model.eval()
    
    logger.info(f"Tokenizing wild-type sequence for target: {target_name} (Length: {len(wt_seq)} aa)")
    inputs = tokenizer(wt_seq, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Logits shape: [batch_size, sequence_length, vocab_size]
        logits = outputs.logits[0] 
        
    # Standard 20 IUPAC amino acid characters (excluding special tokens)
    standard_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Map letters to tokenizer token IDs
    aa_to_token_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in standard_amino_acids}
    
    # Extract token mappings from sequence (bypassing CLS token at index 0)
    input_ids = inputs["input_ids"][0].tolist()
    
    results = []
    
    logger.info("Computing Log-Likelihood Ratios (LLRs) across mutational landscape...")
    # Loop over original amino acid positions (skipping [CLS] and [SEP] tokens)
    for pos_idx, wt_aa in enumerate(wt_seq):
        token_pos = pos_idx + 1 # Offset for starting [CLS] token
        
        # Compute log probabilities from raw network outputs
        log_probs = torch.log_softmax(logits[token_pos], dim=-1)
        wt_token_id = input_ids[token_pos]
        wt_log_prob = log_probs[wt_token_id].item()
        
        for mut_aa in standard_amino_acids:
            mut_token_id = aa_to_token_id[mut_aa]
            mut_log_prob = log_probs[mut_token_id].item()
            
            # LLR = log(P(mutant)) - log(P(wild-type))
            llr = mut_log_prob - wt_log_prob
            
            results.append({
                "position": pos_idx + 1,
                "wild_type": wt_aa,
                "mutation": mut_aa,
                "variant": f"{wt_aa}{pos_idx + 1}{mut_aa}",
                "llr_score": round(llr, 4),
                "is_wildtype": 1 if wt_aa == mut_aa else 0
            })
            
    df_landscape = pd.DataFrame(results)
    return df_landscape

def filter_and_save_variants(df: pd.DataFrame, config: dict):
    """Filters top-performing variants and outputs results to structured CSVs."""
    phase_cfg = config["phase1_design"]
    top_n = phase_cfg["top_n_variants"]
    threshold = phase_cfg["mutation_score_threshold"]
    
    # Ensure directories exist
    os.makedirs("data/results", exist_ok=True)
    
    # 1. Save the total mutational landscape
    landscape_path = "data/results/full_mutational_landscape.csv"
    df.to_csv(landscape_path, index=False)
    logger.info(f"Full mutational landscape exported to: {landscape_path}")
    
    # 2. Filter out native wildtype residues and keep beneficial mutations (LLR > threshold)
    beneficial_mutations = df[(df["is_wildtype"] == 0) & (df["llr_score"] > threshold)]
    top_variants = beneficial_mutations.sort_values(by="llr_score", ascending=False).head(top_n)
    
    top_variants_path = "data/results/top_candidate_variants.csv"
    top_variants.to_csv(top_variants_path, index=False)
    logger.info(f"Top {top_n} candidate variants saved to: {top_variants_path}")
    
    print("\n--- TOP 5 ENGINEERED CANDIDATES ---")
    print(top_variants[["variant", "llr_score"]].head(5).to_string(index=False))

if __name__ == "__main__":
    logger.info("Starting Phase 1 Execution Pipeline...")
    pipeline_config = load_config()
    landscape_data = run_zero_shot_inference(pipeline_config)
    filter_and_save_variants(landscape_data, pipeline_config)
    logger.info("Phase 1 execution successfully terminated.")
