#!/usr/bin/env python3
"""
GreenOps Frontend API Server
Wraps production backend logic without modifications, serves professional SaaS MPA
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("greenops.frontend")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pipeline_runner
from ast_parser import ASTParser
from dependency_graph_engine import DependencyGraphEngine
from pr_diff_processor import parse_changed_files_from_diff
from src.ml.gatekeeper import Gatekeeper
from carbon_inference_engine import CarbonIntensityClient

app = FastAPI(title="GreenOps Frontend API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

LAST_PIPELINE: Dict = {}


# ════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════════════════════════════════

class PipelineRequest(BaseModel):
    repo: Optional[str] = None
    pr: Optional[int] = 0
    base_branch: Optional[str] = "main"
    diff_text: Optional[str] = None
    region: Optional[str] = "default"
    carbon_threshold: Optional[float] = 500.0


class GatekeeperRequest(BaseModel):
    similarity: float
    change_size: int
    module_impact_score: Optional[float] = 0.5
    carbon_intensity: Optional[float] = 500.0
    is_kafka_consumer: Optional[int] = 0
    is_kafka_producer: Optional[int] = 0
    is_shared_db: Optional[int] = 0
    is_frontend_contract: Optional[int] = 0
    is_shared_utility: Optional[int] = 0
    transitive_depth: Optional[int] = 1
    test_name: Optional[str] = ""


class DependencyMapperRequest(BaseModel):
    repo: Optional[str] = None
    pr: Optional[int] = 0
    base_branch: Optional[str] = "main"
    diff_text: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _extract_file_diff(full_diff: str, filepath: str) -> str:
    marker = f"diff --git a/{filepath} b/{filepath}"
    start = full_diff.find(marker)
    if start == -1:
        return ""
    next_start = full_diff.find("\ndiff --git ", start + len(marker))
    return full_diff[start:next_start].strip() if next_start != -1 else full_diff[start:].strip()


def _safe_repo(repo: Optional[str], allow_empty: bool = False) -> str:
    repo_name = repo or os.environ.get("REPO_NAME", "")
    if not repo_name and not allow_empty:
        raise HTTPException(status_code=400, detail="repo is required")
    return repo_name


def _get_git_diff(repo_root: str) -> str:
    """Generate diff from local git repository"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, timeout=30, cwd=repo_root
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        
        # Try unstaged changes
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, timeout=30, cwd=repo_root
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        
        raise RuntimeError("No changes found in git working directory")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Git diff timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Git not found - ensure you're in a git repository")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate local diff: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "repo_root": str(REPO_ROOT)}


@app.post("/api/pipeline")
async def execute_pipeline(request: PipelineRequest):
    """
    Main pipeline endpoint - wraps all production backend stages
    Returns all data needed for the 5-page SaaS frontend
    """
    repo = _safe_repo(request.repo, allow_empty=True)
    pr_number = int(request.pr or 0)
    base_branch = request.base_branch or "main"
    diff_text = request.diff_text
    region = request.region or "default"
    carbon_threshold = float(request.carbon_threshold or 500.0)
    repo_root = str(REPO_ROOT)
    
    try:
        # Get diff
        if diff_text:
            diff = diff_text
        elif not repo and pr_number == 0:
            diff = _get_git_diff(repo_root)
        else:
            diff = pipeline_runner.stage_get_diff(repo, pr_number, base_branch)
        
        # Execute pipeline stages
        stages: Dict[str, float] = {}
        
        t0 = pipeline_runner.time.time()
        module_results = pipeline_runner.stage_extract_modules(repo_root, repo, pr_number)
        stages["extract_modules"] = round((pipeline_runner.time.time() - t0) * 1000, 1)
        
        t0 = pipeline_runner.time.time()
        dep_graph = pipeline_runner.stage_build_dependency_graph(repo_root, repo)
        stages["build_dependency_graph"] = round((pipeline_runner.time.time() - t0) * 1000, 1)
        
        t0 = pipeline_runner.time.time()
        carbon = pipeline_runner.stage_get_carbon()
        stages["get_carbon"] = round((pipeline_runner.time.time() - t0) * 1000, 1)
        
        t0 = pipeline_runner.time.time()
        selection = pipeline_runner.stage_select_tests(
            repo=repo,
            repo_root=repo_root,
            diff_text=diff,
            pr_number=pr_number,
            carbon_intensity=carbon.get("intensity", 500),
        )
        stages["select_tests"] = round((pipeline_runner.time.time() - t0) * 1000, 1)
        
        selection = pipeline_runner.apply_confidence_gate(selection)
        
        t0 = pipeline_runner.time.time()
        schedule = pipeline_runner.stage_schedule(selection, carbon)
        stages["schedule"] = round((pipeline_runner.time.time() - t0) * 1000, 1)
        
        # Extract changed files and functions
        changed_files_list = [item["filepath"] for item in parse_changed_files_from_diff(diff)]
        ast_parser = ASTParser(repo_root=repo_root)
        changed_functions: Dict[str, List[str]] = {}
        for filepath in changed_files_list:
            file_diff = _extract_file_diff(diff, filepath)
            absolute_path = REPO_ROOT / filepath
            if absolute_path.exists() and file_diff:
                changed_functions[filepath] = ast_parser.get_changed_functions(file_diff, str(absolute_path))
            else:
                changed_functions[filepath] = []
        
        # Extract test lists
        selected_tests = selection.get("selected_tests", [])
        pruned_tests = selection.get("pruned_tests", [])
        
        # Calculate metrics
        total_tests = len(selected_tests) + len(pruned_tests)
        tests_saved = len(pruned_tests)
        runtime_reduction = f"{round((tests_saved / total_tests * 100) if total_tests > 0 else 0, 1)}%" if total_tests > 0 else "0%"
        
        # ML/XGBoost features and explanation
        probability_of_failure = selection.get("probability_of_failure", 0.0)
        gate_decision = selection.get("gate_decision", "RUN")
        
        # Build Frontend Response
        result = {
            "status": "success",
            "final_decision": gate_decision,
            "probability_of_failure": probability_of_failure,
            "current_carbon_intensity": carbon.get("intensity", 0),
            "carbon_source": carbon.get("source", "Live Grid"),
            "tests_saved": tests_saved,
            "runtime_reduction": runtime_reduction,
            "status": "completed",
            "stage_timings": stages,
            
            # Dependency page
            "changed_files": changed_functions,
            "dependency_graph": {
                "nodes": list(changed_functions.keys()),
                "edges": [],
            },
            "similarity_scores": [
                {
                    "module": m,
                    "test": t,
                    "score": selection.get("similarities", {}).get(f"{m}:{t}", 0.5),
                    "included": t in selected_tests,
                }
                for m in changed_functions.keys()
                for t in selected_tests[:5]
            ],
            
            # Carbon page
            "carbon_threshold": carbon_threshold,
            "carbon_action": "Proceed" if carbon.get("intensity", 0) <= carbon_threshold else "Delay",
            
            # ML page
            "gate_decision": gate_decision,
            "ml_features": [
                {"name": "similarity", "value": selection.get("avg_similarity", 0.5), "impact": "high"},
                {"name": "change_size", "value": len(changed_files_list), "impact": "medium"},
                {"name": "carbon_intensity", "value": carbon.get("intensity", 500), "impact": "medium"},
            ],
            
            # CI/CD page
            "selected_tests": selected_tests,
            "pruned_tests": pruned_tests,
        }
        
        LAST_PIPELINE.clear()
        LAST_PIPELINE.update(result)
        
        return result
    
    except HTTPException:
        raise
    except Exception as exc:
        error_msg = str(exc)
        if "Could not fetch diff" in error_msg:
            error_msg = "Unable to get code changes. For local testing, either upload a .diff file or ensure you have uncommitted changes in git."
        elif "No changes found" in error_msg:
            error_msg = "No code changes detected. Make sure you have modified files or upload a diff file."
        elif "Git not found" in error_msg:
            error_msg = "Git is not available. For local testing, please upload a .diff file instead."
        elif "numpy" in error_msg.lower():
            error_msg = "Missing dependencies. Please run: pip install -r requirements.txt"
        else:
            error_msg = f"Pipeline error: {error_msg}"
        
        log.error("Pipeline failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/gatekeeper")
async def run_gatekeeper(request: GatekeeperRequest):
    """ML Gatekeeper endpoint - wraps XGBoost prediction"""
    try:
        gatekeeper = Gatekeeper()
        pf = gatekeeper.predict_failure_prob(
            similarity=request.similarity,
            change_size=request.change_size,
            module_impact_score=request.module_impact_score,
            is_kafka_consumer=request.is_kafka_consumer,
            is_kafka_producer=request.is_kafka_producer,
            is_shared_db=request.is_shared_db,
            is_frontend_contract=request.is_frontend_contract,
            is_shared_utility=request.is_shared_utility,
            transitive_depth=request.transitive_depth,
        )
        carbon_check = gatekeeper.compare_to_carbon_threshold(
            pf=pf,
            carbon_intensity=request.carbon_intensity,
            test_name=request.test_name,
        )
        return {"pf": round(pf, 4), "carbon_check": carbon_check, "source": "src.ml.gatekeeper"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/dependency-mapper")
async def dependency_mapper(request: DependencyMapperRequest):
    """Dependency Mapper endpoint"""
    repo = _safe_repo(request.repo, allow_empty=True)
    pr_number = int(request.pr or 0)
    base_branch = request.base_branch or "main"
    diff_text = request.diff_text
    repo_root = str(REPO_ROOT)

    try:
        if diff_text:
            diff = diff_text
        elif not repo and pr_number == 0:
            diff = _get_git_diff(repo_root)
        else:
            diff = pipeline_runner.stage_get_diff(repo, pr_number, base_branch)

        changed_files = [item["filepath"] for item in parse_changed_files_from_diff(diff)]
        engine = DependencyGraphEngine(repo_root=repo_root)
        graph_path = str(pipeline_runner.OUTPUT_DIR / "dependency_graph.json")
        if Path(graph_path).exists():
            engine.load(graph_path)
        else:
            engine.build(repo=repo, save_path=graph_path)

        impact = engine.get_tests_for_changed_modules(changed_files)
        full_impact = engine.get_full_impact_map(changed_files)

        return {
            "changed_files": changed_files,
            "impact": impact,
            "full_impact": full_impact,
            "graph": {
                "module_graph": engine.module_graph,
                "reverse_graph": engine.reverse_graph,
                "test_map": engine.test_map,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        error_msg = str(exc)
        if "Could not fetch diff" in error_msg:
            error_msg = "Unable to get code changes. For local testing, either upload a .diff file or ensure you have uncommitted changes in git."
        elif "No changes found" in error_msg:
            error_msg = "No code changes detected. Make sure you have modified files or upload a diff file."
        elif "Git not found" in error_msg:
            error_msg = "Git is not available. For local testing, please upload a .diff file instead."
        else:
            error_msg = f"Dependency mapper error: {error_msg}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/last-pipeline")
async def last_pipeline():
    """Retrieve last pipeline result"""
    if not LAST_PIPELINE:
        raise HTTPException(status_code=404, detail="No pipeline result available yet")
    return LAST_PIPELINE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
