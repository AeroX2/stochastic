"""
Small-scale test for Hugging Face checkpoint upload.

Creates a minimal fake checkpoint dir, uploads it to the given HF repo,
then removes the temp dir. Use this to verify HF_TOKEN and upload_folder work
before running full training with --hf-repo.

Usage (from repo root, with PYTHONPATH including repo and nanochat):
  export HF_TOKEN=your_token
  export HF_REPO=your_username/stochastic-upload-test
  python -m experiments.test_hf_upload

Or with explicit args:
  python -m experiments.test_hf_upload --repo=your_username/stochastic-upload-test
"""
import argparse
import json
import os
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Test HF checkpoint upload with a minimal fake checkpoint.")
    parser.add_argument("--repo", type=str, default=os.environ.get("HF_REPO"), help="HF repo id (USER/REPO). Default: HF_REPO env.")
    args = parser.parse_args()
    repo = args.repo
    if not repo or "/" not in repo:
        print("Error: set --repo=USERNAME/REPO or HF_REPO env (e.g. myuser/stochastic-upload-test)", file=__import__("sys").stderr)
        raise SystemExit(1)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN", file=__import__("sys").stderr)
        raise SystemExit(1)

    try:
        import torch
    except ImportError:
        torch = None

    with tempfile.TemporaryDirectory(prefix="stochastic_hf_test_") as tmpdir:
        # Minimal checkpoint-like files (same names nanochat uses)
        if torch is not None:
            torch.save({"step": 1, "dummy": torch.zeros(2, 2)}, os.path.join(tmpdir, "model_000001.pt"))
        else:
            open(os.path.join(tmpdir, "model_000001.pt"), "wb").write(b"dummy checkpoint")
        with open(os.path.join(tmpdir, "meta_000001.json"), "w") as f:
            json.dump({"step": 1, "test": True}, f)

        print(f"Uploading minimal checkpoint to {repo} ...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(folder_path=tmpdir, repo_id=repo, repo_type="model", token=token)
        print(f"Done. Checkpoint at https://huggingface.co/{repo}")
        print("(You can delete this test repo on the Hub or leave it; re-running will overwrite.)")


if __name__ == "__main__":
    main()
