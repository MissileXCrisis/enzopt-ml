import os
import time
import yaml
import requests
import pandas as pd
from src.utils.logger import setup_logger

# Initialize structured logging
logger = setup_logger("Phase2_Build")

# Meta AI's standard serverless ESMFold coordinate generation endpoint
ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Safely parses the central YAML configuration parameters."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_mutated_sequence(wt_seq: str, position_1indexed: int, wt_aa: str, mut_aa: str) -> str:
    """
    Applies string-splicing to introduce a single-point mutation into the sequence.
    Includes validation checkpoints to prevent indexing errors.
    """
    pos_0indexed = position_1indexed - 1
    
    # Structural Biology boundary sanity checks
    if pos_0indexed < 0 or pos_0indexed >= len(wt_seq):
        raise IndexError(f"Position {position_1indexed} falls outside sequence length bounds ({len(wt_seq)}).")
    if wt_seq[pos_0indexed] != wt_aa:
        raise ValueError(f"Mismatched residue: Position {position_1indexed} is native '{wt_seq[pos_0indexed]}', not specified '{wt_aa}'.")
        
    seq_list = list(wt_seq)
    seq_list[pos_0indexed] = mut_aa
    return "".join(seq_list)

def query_esmfold_api(sequence: str, variant_name: str, max_retries: int = 5) -> str:
    """
    Queries the ESMFold endpoint with exponential backoff to handle transient network issues.
    Returns raw PDB file contents as a text string.
    """
    backoff_time = 2
    for attempt in range(max_retries):
        try:
            # ESMFold expects raw uppercase sequence strings passed directly into the request body
            response = requests.post(ESMFOLD_API_URL, data=sequence.upper(), timeout=60)
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) during folding for {variant_name}. Backing off for {backoff_time}s...")
            else:
                logger.warning(f"Unexpected API status {response.status_code} on attempt {attempt + 1}.")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error on attempt {attempt + 1} for {variant_name}: {e}")
            
        time.sleep(backoff_time)
        backoff_time *= 2  # Exponential backoff loop
        
    raise RuntimeError(f"Critical execution failure: API connection timed out for {variant_name} after {max_retries} attempts.")

def calculate_mean_plddt(pdb_text: str) -> float:
    """
    Parses the B-factor field (columns 61-66) of 'ATOM' lines in the PDB file 
    to calculate the global structural prediction confidence score (mean pLDDT).
    """
    plddt_scores = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM  "):
            try:
                # Standard PDB layout format parsing
                plddt_val = float(line[60:66].strip())
                plddt_scores.append(plddt_val)
            except ValueError:
                continue
    return round(sum(plddt_scores) / len(plddt_scores), 2) if plddt_scores else 0.0

def run_structural_generation():
    """Main orchestrator running the generation of wild-type and mutant 3D structures."""
    config = load_config()
    phase1_cfg = config["phase1_design"]
    wt_seq = phase1_cfg["wild_type_sequence"]
    target_name = phase1_cfg["target_name"]
    
    # Restrict processing scope for high-speed local validation
    top_n = phase1_cfg.get("top_n_variants", 20)
    
    output_dir = "data/processed/pdb"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- STEP 1: GENERATE WILD-TYPE CONTROL STRUCTURE ---
    wt_pdb_path = os.path.join(output_dir, f"{target_name}.pdb")
    if not os.path.exists(wt_pdb_path):
        logger.info(f"Generating wild-type baseline control structure: {target_name}")
        wt_pdb_content = query_esmfold_api(wt_seq, target_name)
        mean_plddt = calculate_mean_plddt(wt_pdb_content)
        
        with open(wt_pdb_path, "w") as f:
            f.write(wt_pdb_content)
        logger.info(f"Successfully generated baseline structure {target_name}.pdb | Mean pLDDT: {mean_plddt}%")
    else:
        logger.info("Wild-type control structure already cached locally. Skipping prediction.")

    # --- STEP 2: LOOP AND MUTATE HIGH-FITNESS CANDIDATES ---
    candidates_csv = "data/results/top_candidate_variants.csv"
    if not os.path.exists(candidates_csv):
        logger.error(f"Missing Phase 1 input files at: {candidates_csv}")
        return
        
    df_candidates = pd.read_csv(candidates_csv).head(top_n)
    logger.info(f"Loaded top {len(df_candidates)} evolutionary variants for structural translation.")
    
    folding_metrics = []
    
    for _, row in df_candidates.iterrows():
        variant = row["variant"]
        pos = int(row["position"])
        wt_aa = row["wild_type"]
        mut_aa = row["mutation"]
        llr = row["llr_score"]
        
        variant_pdb_path = os.path.join(output_dir, f"{target_name}_{variant}.pdb")
        
        if os.path.exists(variant_pdb_path):
            logger.info(f"Structure for variant {variant} already exists locally. Skipping.")
            continue
            
        try:
            logger.info(f"Processing candidate variant {variant} | Introducing mutation at position {pos}...")
            mutated_seq = generate_mutated_sequence(wt_seq, pos, wt_aa, mut_aa)
            
            # Fire API call
            pdb_content = query_esmfold_api(mutated_seq, variant)
            variant_plddt = calculate_mean_plddt(pdb_content)
            
            # Save atomic coordinate map
            with open(variant_pdb_path, "w") as f:
                f.write(pdb_content)
                
            logger.info(f"Saved {target_name}_{variant}.pdb | Mean pLDDT: {variant_plddt}%")
            
            folding_metrics.append({
                "variant": variant,
                "llr_score": llr,
                "mean_plddt": variant_plddt,
                "pdb_path": variant_pdb_path,
                "status": "SUCCESS" if variant_plddt >= 70.0 else "LOW_CONFIDENCE"
            })
            
            # Brief rate-limiting protection sleep
            time.sleep(1.5)
            
        except Exception as err:
            logger.error(f"Failed to generate structure for variant {variant}: {err}")
            folding_metrics.append({
                "variant": variant,
                "llr_score": llr,
                "mean_plddt": 0.0,
                "pdb_path": None,
                "status": f"FAILED: {str(err)}"
            })

    # Save summary report
    if folding_metrics:
        df_report = pd.DataFrame(folding_metrics)
        report_path = "data/results/structural_generation_summary.csv"
        df_report.to_csv(report_path, index=False)
        logger.info(f"Phase 2 processing execution summary saved to: {report_path}")

if __name__ == "__main__":
    logger.info("Initializing Phase 2 Structural Folding Engine...")
    run_structural_generation()
    logger.info("Phase 2 deployment successfully terminated.")
