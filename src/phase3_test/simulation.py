import os
import sys

# =========================================================================
# CRITICAL DEVOPS / WSL FIX: Mask broken host OpenCL virtualization drivers
# This MUST execute before importing openmm to prevent background runtime crashes.
# =========================================================================
os.environ["OCL_ICD_VENDORS"] = "/dev/null"
os.environ["OPENCL_VENDOR_PATH"] = "/dev/null"
os.environ["OCL_ICD_FILENAMES"] = ""

import yaml
import glob
import pandas as pd
import numpy as np
import mdtraj as md
import openmm as mm
import openmm.app as app
from openmm import unit
from src.utils.logger import setup_logger

# Initialize structured logging
logger = setup_logger("Phase3_Test")

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Parses the central YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def select_best_platform() -> tuple:
    """
    Dynamically queries available OpenMM execution architectures.
    Forces 'Reference' to ensure absolute sandbox stability inside WSL.
    """
    num_platforms = mm.Platform.getNumPlatforms()
    available_platforms = [mm.Platform.getPlatform(i).getName() for i in range(num_platforms)]
    logger.info(f"Detected hardware platforms: {available_platforms}")
    
    priority = ["CPU","Reference"]
    for target in priority:
        if target in available_platforms:
            logger.info(f"Selected runtime execution hardware: [{target}]")
            return mm.Platform.getPlatformByName(target), target
            
    return mm.Platform.getPlatform(0), mm.Platform.getPlatform(0).getName()

def inject_oxt_and_ter_records(pdb_path: str) -> str:
    """
    Bioinformatics text-utility that fixes ML structural truncation.
    Locates the final residue, duplicates the carbonyl 'O' line, displaces it 
    by 1.25 Angstroms to avoid steric clashes, and appends a standard TER record.
    """
    fixed_path = pdb_path.replace(".pdb", "_repaired.pdb")
    
    with open(pdb_path, "r") as f:
        lines = f.readlines()
        
    last_atom_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("ATOM  "):
            last_atom_idx = idx
            
    if last_atom_idx == -1:
        return pdb_path

    last_line = lines[last_atom_idx]
    res_seq = last_line[22:26].strip()
    chain_id = last_line[21]
    
    o_line = None
    for line in lines:
        if line.startswith("ATOM  ") and line[12:16].strip() == "O" and line[22:26].strip() == res_seq and line[21] == chain_id:
            o_line = line
            break
            
    if not o_line:
        o_line = last_line
        
    try:
        atom_serial = int(o_line[6:11].strip()) + 1
        x_val = float(o_line[30:38].strip())
        new_x_str = f"{x_val + 1.25:>8.3f}"
        
        line_start = f"ATOM  {atom_serial:>5}  OXT " + o_line[17:30]
        line_coords = new_x_str + o_line[38:]
        oxt_line = line_start + line_coords
        
        ter_line = f"TER   {atom_serial+1:>5}      {o_line[17:20]} {chain_id}{int(res_seq):>4}\n"
        
    except ValueError:
        atom_serial = int(o_line[6:11].strip()) + 1
        oxt_line = o_line[:6] + f"{atom_serial:>5}" + "  OXT" + o_line[16:]
        ter_line = f"TER   {atom_serial+1:>5}      {o_line[17:20]} {chain_id}{int(res_seq):>4}\n"
    
    new_lines = []
    for idx, line in enumerate(lines):
        new_lines.append(line)
        if idx == last_atom_idx:
            new_lines.append(oxt_line)
            new_lines.append(ter_line)
            
    with open(fixed_path, "w") as f:
        f.writelines(new_lines)
        
    logger.info(f"Text pre-processor: Injected structural OXT and TER records successfully.")
    return fixed_path

def run_molecular_dynamics(pdb_path: str, variant_name: str, platform: mm.Platform) -> tuple:
    """Preprocesses raw AI coordinates, generates hydrogens, and runs implicit solvent MD."""
    repaired_pdb_path = inject_oxt_and_ter_records(pdb_path)
    pdb = app.PDBFile(repaired_pdb_path)
    forcefield = app.ForceField('amber99sbildn.xml', 'implicit/obc2.xml')
    
    logger.info(f"Assembling structural protonation layers via OpenMM Modeller...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    
    final_pdb_path = pdb_path.replace(".pdb", "_final.pdb")
    with open(final_pdb_path, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0*unit.nanometers,
        constraints=app.HBonds
    )
    
    integrator = mm.LangevinMiddleIntegrator(
        300*unit.kelvin,       
        1.0/unit.picosecond,   
        2.0*unit.femtoseconds  
    )
    
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    logger.info(f"Executing energy minimization for {variant_name}...")
    simulation.minimizeEnergy(maxIterations=200)
    
    output_dcd = f"data/processed/{variant_name}_trajectory.dcd"
    logger.info(f"Launching production trajectory loop (1000 steps) -> {output_dcd}")
    
    simulation.reporters.append(app.DCDReporter(output_dcd, 100))
    simulation.step(1000)
    
    if os.path.exists(repaired_pdb_path):
        os.remove(repaired_pdb_path)
        
    return output_dcd, final_pdb_path

def analyze_backbone_fluctuations(final_pdb_path: str, dcd_path: str) -> float:
    """Parses trajectories using MDTraj to extract global mean RMSF."""
    logger.info(f"Analyzing trajectory file via MDTraj: {dcd_path}")
    traj = md.load(dcd_path, top=final_pdb_path)
    ca_indices = traj.topology.select("name CA")
    
    rmsf_per_residue = md.rmsf(traj, traj[0], atom_indices=ca_indices)
    mean_global_rmsf = float(np.mean(rmsf_per_residue) * 10.0)
    return round(mean_global_rmsf, 4)

def orchestrate_testing_phase():
    """Main execution loop finding and analyzing all original un-repaired structural models."""
    raw_pdb_files = [
        f for f in glob.glob("data/processed/pdb/*.pdb") 
        if "_fixed" not in f and "_final" not in f and "_repaired" not in f
    ]
    
    if not raw_pdb_files:
        logger.error("No raw structural coordinate files found in data/processed/pdb/. Run Phase 2 first!")
        return
        
    logger.info(f"Discovered {len(raw_pdb_files)} structural targets for dynamic physical simulation.")
    selected_platform, platform_name = select_best_platform()
    simulation_results = []
    
    for pdb_file in raw_pdb_files:
        filename = os.path.basename(pdb_file)
        variant_id = filename.replace(".pdb", "")
        
        logger.info(f"\n======== SIMULATING TARGET: {variant_id} ========")
        
        try:
            trajectory_path, final_pdb_path = run_molecular_dynamics(pdb_file, variant_id, selected_platform)
            mean_rmsf = analyze_backbone_fluctuations(final_pdb_path, trajectory_path)
            logger.info(f"Physical validation complete for {variant_id} | Global Mean RMSF: {mean_rmsf} Å")
            
            simulation_results.append({
                "variant_id": variant_id,
                "mean_rmsf_angstroms": mean_rmsf,
                "platform_used": platform_name,
                "status": "VALIDATED"
            })
            
            if os.path.exists(trajectory_path):
                os.remove(trajectory_path)
                
        except Exception as e:
            logger.error(f"Physics simulation aborted for variant target {variant_id}: {e}")
            simulation_results.append({
                "variant_id": variant_id,
                "mean_rmsf_angstroms": 0.0,
                "platform_used": platform_name,
                "status": f"FAILED: {str(e)}"
            })
            
    df_md = pd.DataFrame(simulation_results)
    output_report = "data/results/md_simulation_summary.csv"
    df_md.to_csv(output_report, index=False)
    logger.info(f"\nPhase 3 processing completed. Results saved to: {output_report}")
    print("\n" + df_md.to_string(index=False))

if __name__ == "__main__":
    logger.info("Initializing Phase 3 Automated Physics Validation Layer...")
    orchestrate_testing_phase()
    logger.info("Phase 3 operational deployment successfully terminated.")
