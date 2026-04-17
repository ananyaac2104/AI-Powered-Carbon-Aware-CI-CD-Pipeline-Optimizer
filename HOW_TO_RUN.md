# 🚀 How to Run Green-Ops CI/CD Pipeline

The easiest way to run the entire Green-Ops project is using the **One-Click Orchestrator**.

---

## ⚡ Option 1: Quick Start (Recommended)

Run these two commands to set up your environment and execute the full pipeline:

```bash
# 1. Setup the environment
bash setup_env.sh

# 2. Run the master pipeline
source venv/bin/activate
python greenops_run_master.py
```

### What does the Orchestrator do?
1. **Initializes Database**: Seeds the `greenops.db` with Indian carbon data.
2. **Prepares ML**: Ensures the XGBoost failure prediction model is ready.
3. **Simulates CI**: Reconstructs a Git PR diff, parses it, and predicts test failure risk.
4. **Schedules**: Picks the cleanest DC and generates a carbon-aware schedule.
5. **Launches Dashboard**: Prompts you to view results in your browser.

---

## 🛠️ Option 2: Manual Developer Mode

If you need to run specific parts of the pipeline individually:

### Step A: Setup & Carbon DB
```bash
bash setup_env.sh
source venv/bin/activate
python india_carbon_pipeline.py --csv ember_india_carbon.csv
```

### Step B: Preprocessing
```bash
python preprocessing.py --presubmit presubmit_clean.csv --postsubmit postsubmit_clean.csv
```

### Step C: CI Integration (Diff Parsing)
```bash
export REPO_NAME='akshaya209/GreenOps-AI-Pipeline'
export PR_NUMBER=1
python github_ci_integration.py
```

### Step D: Dashboard
```bash
cd src/flask_real
python app.py
```
*Visit `http://localhost:5001` for the visualization tool.*

---

## 🧪 Verification
Run the integration test to verify your API keys and configuration:
```bash
python test_integration.py
```

> [!NOTE]
> **API Keys**: All keys are pre-configured in `src/config.py`. You do not need to set environment variables manually for a demo run.
