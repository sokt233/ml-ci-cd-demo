import json, os, tempfile, subprocess, sys

def test_train_fast_produces_artifacts():
    with tempfile.TemporaryDirectory() as d:
        cmd = [sys.executable, "-m", "src.train", "--fast", "--out_dir", d]
        subprocess.check_call(cmd)
        assert os.path.exists(os.path.join(d, "model.pkl"))
        mpath = os.path.join(d, "metrics.json")
        assert os.path.exists(mpath)
        with open(mpath) as f:
            metrics = json.load(f)
        assert "accuracy" in metrics and 0 <= metrics["accuracy"] <= 1
