from src import config
import os
import sys
import subprocess
import time
import logging
import json
import webbrowser
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("greenops.master")

def run_step(name, command):
    log.info(f"🚀 Starting Step: {name}")
    try:
        subprocess.run(command, check=True, shell=True)
        log.info(f"✅ Step Complete: {name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Step Failed: {name} | Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🍀 GREEN-OPS MASTER PIPELINE ORCHESTRATOR 🍀")
    print("="*60 + "\n")

    # 1. DATABASE INITIALIZATION
    if not Path(config.GREENOPS_DB).exists():
        if not run_step("Seeding Carbon Database", "python india_carbon_pipeline.py --csv ember_india_carbon.csv"):
            sys.exit(1)
    else:
        log.info("✓ Carbon Database already exists.")

    # 2. ML GATEKEEPER TRAINING (Simulated Smoke Test)
    log.info("🚀 Preparing ML Gatekeeper...")
    if not (config.OUTPUT_DIR / "gatekeeper_model.json").exists():
        if not run_step("Training ML Gatekeeper", "python src/ml/gatekeeper.py"):
             log.warning("Real training failed, ensuring greenops_output directory exists.")
             os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    else:
        log.info("✓ Gatekeeper model ready.")

    # 3. PR ANALYSIS SIMULATION
    log.info("🚀 Simulating CI Pull Request Analysis (GreenOps Scenario)...")
    if not run_step("PR Diff Analysis", "python github_ci_integration.py"):
        sys.exit(1)

    # 4. CARBON-AWARE SCHEDULING
    log.info("🚀 Generating Carbon-Aware Schedule...")
    
    # Use config.GITHUB_PR to stay synchronized
    pr_num = config.GITHUB_PR
    scheduling_logic = f"""
from src import config
from src.ml.gatekeeper import run_gatekeeper_pipeline
from carbon_aware_scheduler import CarbonAwareScheduler
import json, numpy as np, os

artifact_path = os.path.join(config.OUTPUT_DIR, f'preprocessing_artifacts_pr{{config.GITHUB_PR}}.json')
with open(artifact_path) as f:
    artifacts = json.load(f)

# Mock embeddings for simulation
dim = config.EMBEDDING_DIM
changed_emb = {{m['module_hash']: np.random.rand(dim).astype(np.float32)
               for m in artifacts['modules'] if m.get('module_hash')}}
test_emb = {{'CoreEngineTest': np.random.rand(dim), 'AuthTest': np.random.rand(dim)}}

decision = run_gatekeeper_pipeline(
    changed_modules=artifacts['modules'],
    module_registry=artifacts['modules'],
    changed_embeddings=changed_emb,
    test_embeddings=test_emb,
    change_size=120
)

scheduler = CarbonAwareScheduler(provider='aws')
schedule  = scheduler.schedule(decision)

# Save result for dashboard consumption
with open(os.path.join(config.OUTPUT_DIR, 'test_schedule.json'), 'w') as f:
    json.dump(schedule, f, indent=2, default=str)
print('✓ Schedule generated successfully.')
"""
    subprocess.run([sys.executable, "-c", scheduling_logic], check=True)

    print("\n" + "="*60)
    print("🎉 PIPELINE EXECUTION COMPLETE!")
    print("="*60)
    print(f"  Carbon DB     : {config.GREENOPS_DB}")
    print(f"  Decision JSON : {config.OUTPUT_DIR}/pruning_decision.json")
    print(f"  Schedule JSON : {config.OUTPUT_DIR}/test_schedule.json")
    print("="*60 + "\n")

    ans = input("📊 Would you like to launch the Dashboard now? (y/n): ")
    if ans.lower() == 'y':
        url = f"http://localhost:{config.DASHBOARD_PORT}"
        log.info(f"Launching Flask Dashboard on {url}...")
        webbrowser.open(url)
        # Step into the dashboard dir and run
        os.chdir("src/flask_real")
        subprocess.run(["python", "app.py"])

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
