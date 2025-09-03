import argparse, json, os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data import small_dataset

def main(fast: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = small_dataset()
    n_estimators = 20 if fast else 200
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()
    main(args.fast, args.out_dir)
