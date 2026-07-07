import os
import sqlite3
import yaml
import pandas as pd
from src.utils.logger import setup_logger

# Initialize structured logging
logger = setup_logger("Phase4_Learn_DB")

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Parses the central YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def initialize_database(db_path: str) -> sqlite3.Connection:
    """Creates the SQLite database file and establishes relational tables."""
    logger.info(f"Initializing relational data warehouse schema at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Clear out old instances if they exist to maintain pipeline idempotency
    cursor.execute("DROP TABLE IF EXISTS sequence_designs;")
    cursor.execute("DROP TABLE IF EXISTS structural_models;")
    cursor.execute("DROP TABLE IF EXISTS physics_simulations;")
    
    # 2. Generate clean relational schemas
    cursor.execute("""
    CREATE TABLE sequence_designs (
        variant TEXT PRIMARY KEY,
        position INTEGER,
        wild_type TEXT,
        mutation TEXT,
        llr_score REAL,
        is_wildtype INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE structural_models (
        variant TEXT PRIMARY KEY,
        mean_plddt REAL,
        pdb_path TEXT,
        folding_status TEXT,
        FOREIGN KEY (variant) REFERENCES sequence_designs(variant)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE physics_simulations (
        variant TEXT PRIMARY KEY,
        mean_rmsf_angstroms REAL,
        platform_used TEXT,
        simulation_status TEXT,
        FOREIGN KEY (variant) REFERENCES sequence_designs(variant)
    );
    """)
    
    conn.commit()
    return conn

def populate_warehouse(conn: sqlite3.Connection, config: dict):
    """Parses phase result CSVs and populates the localized database staging layers."""
    target_name = config["phase1_design"]["target_name"]
    
    # --- 1. INGEST SEQUENCE DESIGNS ---
    p1_csv = "data/results/top_candidate_variants.csv"
    if os.path.exists(p1_csv):
        df_p1 = pd.read_csv(p1_csv)
        df_p1.to_sql("sequence_designs", conn, if_exists="append", index=False)
        logger.info(f"Ingested {len(df_p1)} variants into 'sequence_designs' table.")
    else:
        logger.warning(f"Phase 1 dataset missing at: {p1_csv}")

    # --- 2. INGEST STRUCTURAL MODELS ---
    p2_csv = "data/results/structural_generation_summary.csv"
    if os.path.exists(p2_csv):
        df_p2 = pd.read_csv(p2_csv)
        # Standardize column mappings
        df_p2 = df_p2.rename(columns={"status": "folding_status"})
        df_p2[["variant", "mean_plddt", "pdb_path", "folding_status"]].to_sql(
            "structural_models", conn, if_exists="append", index=False
        )
        logger.info(f"Ingested {len(df_p2)} models into 'structural_models' table.")
    else:
        logger.warning(f"Phase 2 dataset missing at: {p2_csv}")

    # --- 3. INGEST PHYSICS SIMULATIONS ---
    p3_csv = "data/results/md_simulation_summary.csv"
    if os.path.exists(p3_csv):
        df_p3 = pd.read_csv(p3_csv)
        
        # Map physics variant names (e.g., 'IsPETase_WT_C239F') back to raw identifiers ('C239F')
        sanitized_records = []
        for _, row in df_p3.iterrows():
            v_id = row["variant_id"]
            if v_id == target_name:
                clean_variant = "WT"  # Control baseline tag
            else:
                clean_variant = v_id.replace(f"{target_name}_", "")
                
            sanitized_records.append({
                "variant": clean_variant,
                "mean_rmsf_angstroms": row["mean_rmsf_angstroms"],
                "platform_used": row["platform_used"],
                "simulation_status": row["status"]
            })
            
        df_p3_clean = pd.DataFrame(sanitized_records)
        df_p3_clean.to_sql("physics_simulations", conn, if_exists="append", index=False)
        logger.info(f"Ingested {len(df_p3_clean)} simulation runs into 'physics_simulations' table.")
    else:
        logger.warning(f"Phase 3 dataset missing at: {p3_csv}")

def execute_data_mart_build():
    """Main execution block orchestrating relational aggregation."""
    config = load_config()
    db_path = config["database"]["db_name"]
    
    # Guarantee results folder existence
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    connection = initialize_database(db_path)
    try:
        populate_warehouse(connection, config)
        logger.info("Database ingestion pipeline executed successfully.")
    finally:
        connection.close()

if __name__ == "__main__":
    execute_data_mart_build()
