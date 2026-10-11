import sys
import json
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.evaluation.deepeval_runner import run_evaluation_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DeepEval evaluation suite on the golden set.")
    parser.add_argument("--limit", type=int, default=None, help="Number of golden queries (default: all)")
    parser.add_argument("--provider", default=None, help="LLM provider for the pipeline and judge (default from config)")
    parser.add_argument("--model", default=None, help="LLM model (default: provider's configured model)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume a previously interrupted eval run (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh, ignoring any checkpoint from a previous run")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated DeepEval metrics to run, or 'all' (overrides EVAL_METRICS in config)")
    args = parser.parse_args()

    # Override configured metrics if --metrics is passed
    if args.metrics:
        from backend.app.config import settings
        settings.EVAL_METRICS = args.metrics

    job = run_evaluation_sync(run_id="eval_cli", limit=args.limit, provider=args.provider,
                              model=args.model, resume=args.resume)
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Status:   {job.get('status')}")
    print(f"DeepEval: {job.get('deepeval_status')}  (judge: {job.get('judge_model')})")
    resumed = job.get("resumed_samples", 0)
    if resumed:
        print(f"Resumed:  {resumed} samples loaded from checkpoint")
    print(json.dumps(job.get("aggregate_metrics"), indent=2))
    sys.exit(0 if job.get("status") == "completed" else 1)
