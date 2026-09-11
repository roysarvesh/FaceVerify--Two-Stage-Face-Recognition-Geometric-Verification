"""
train_classifier.py
---------------------
Replaces the hand-picked Stage-2 thresholds in config.py with a small,
genuinely trained decision rule, and reports the honest evaluation
methodology that justifies it: cross-validated ROC curves and Equal Error
Rate (EER) - the standard metrics in face verification literature - rather
than a single hand-tuned operating point.

Three things are compared on the SAME held-out pairs (see src/pairs.py for
how pairs are mined - test images only, never used to build the training
database):

    A) Stage 1 alone       - embedding distance thresholded, full ROC curve
    B) Current config.py    - the hand-tuned nested-threshold rule, a single
       (rule)                (FPR, FNR) operating point, not a curve
    C) Learned classifier   - logistic regression over
                              [embedding_distance, landmark_distance],
                              5-fold cross-validated, full ROC curve

The final classifier is then refit on ALL available pairs (standard
practice once cross-validation has honestly estimated its performance) and
saved as a small JSON of coefficients - no sklearn needed at inference time,
just a sigmoid, so this doesn't add a dependency to the deployed app.

Usage:
    python train_classifier.py
    python train_classifier.py --test-dir data/test --target-fpr 0.01
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc

import config
from src.database import load_database
from src.pairs import build_pairs, flatten_pairs


def compute_eer(fpr, tpr, thresholds):
    """Equal Error Rate: the point on the ROC curve where FPR == FNR
    (FNR = 1 - TPR). Standard summary metric in face verification papers -
    a single number that's harder to cherry-pick than an arbitrarily chosen
    operating point."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def threshold_at_target_fpr(fpr, tpr, thresholds, target_fpr):
    """Finds the score threshold achieving the desired false-positive rate,
    for picking a production operating point (e.g. 1% FPR) rather than
    defaulting to the EER point, which isn't always what a real deployment
    wants (a security application usually cares far more about FPR than
    FNR)."""
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(thresholds) - 1))
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def evaluate_rule_based(X, y, emb_match_thr, emb_uncertain_thr, landmark_thr):
    """Scores the CURRENT config.py nested-threshold rule on the same pairs,
    as a single (FPR, FNR) operating point for direct comparison."""
    emb_dist, land_dist = X[:, 0], X[:, 1]
    predicted_match = (emb_dist <= emb_uncertain_thr) & (land_dist <= landmark_thr)
    tp = np.sum((predicted_match == 1) & (y == 1))
    fp = np.sum((predicted_match == 1) & (y == 0))
    tn = np.sum((predicted_match == 0) & (y == 0))
    fn = np.sum((predicted_match == 0) & (y == 1))
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / len(y)
    return {"fpr": fpr, "fnr": fnr, "accuracy": accuracy, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a learned Stage-2 classifier.")
    parser.add_argument("--test-dir", default=config.TEST_DIR)
    parser.add_argument("--db", default=config.EMBEDDINGS_DB_PATH)
    parser.add_argument("--target-fpr", type=float, default=0.01,
                         help="False-positive rate to pick the deployed operating point at (default 1%%).")
    parser.add_argument("--output-model", default=os.path.join(config.PROJECT_ROOT, "data", "stage2_classifier.json"))
    parser.add_argument("--output-plot", default=os.path.join(config.PROJECT_ROOT, "data", "roc_curve.png"))
    parser.add_argument("--output-report", default=os.path.join(config.PROJECT_ROOT, "data", "evaluation_report.md"))
    args = parser.parse_args()

    print(f"Loading training database from {args.db} ...")
    train_database = load_database(args.db)

    print(f"Mining genuine/impostor pairs from {args.test_dir} ...")
    pairs = build_pairs(train_database, args.test_dir,
                         embedding_model=config.EMBEDDING_MODEL,
                         detector_backend=config.DETECTOR_BACKEND)
    X, y, meta = flatten_pairs(pairs, require_landmark=True)
    print(f"{len(X)} labeled pairs ({int(y.sum())} genuine, {int((1 - y).sum())} impostor)")

    if len(X) < 30:
        raise RuntimeError(
            "Too few pairs to train/evaluate meaningfully. Check --test-dir "
            "points at a populated data/test/ directory."
        )

    # ---- A) Stage 1 alone (embedding distance only) ----------------------
    stage1_score = -X[:, 0]  # negate: higher score = more similar = more likely genuine
    fpr_a, tpr_a, thr_a = roc_curve(y, stage1_score)
    eer_a, _ = compute_eer(fpr_a, tpr_a, thr_a)
    auc_a = auc(fpr_a, tpr_a)

    # ---- B) Current hand-tuned nested-threshold rule (single point) ------
    rule_result = evaluate_rule_based(
        X, y,
        config.EMBEDDING_MATCH_THRESHOLD,
        config.EMBEDDING_UNCERTAIN_THRESHOLD,
        config.LANDMARK_MATCH_THRESHOLD,
    )

    # ---- C) Learned classifier, 5-fold cross-validated for an honest ROC -
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = LogisticRegression()
    oof_proba = cross_val_predict(clf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr_c, tpr_c, thr_c = roc_curve(y, oof_proba)
    eer_c, eer_threshold_c = compute_eer(fpr_c, tpr_c, thr_c)
    auc_c = auc(fpr_c, tpr_c)
    deploy_threshold, deploy_fpr, deploy_tpr = threshold_at_target_fpr(fpr_c, tpr_c, thr_c, args.target_fpr)

    # ---- Refit on ALL pairs for the model actually shipped ----------------
    final_clf = LogisticRegression()
    final_clf.fit(X, y)
    w_embedding, w_landmark = final_clf.coef_[0]
    bias = final_clf.intercept_[0]

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model_json = {
        "w_embedding_distance": float(w_embedding),
        "w_landmark_distance": float(w_landmark),
        "bias": float(bias),
        "decision_threshold_probability": deploy_threshold,
        "note": (
            "probability = sigmoid(w_embedding_distance*emb_dist + "
            "w_landmark_distance*land_dist + bias); match if probability >= "
            "decision_threshold_probability"
        ),
        "trained_on_n_pairs": len(X),
        "cv_auc": float(auc_c),
        "cv_eer": float(eer_c),
        "target_fpr": args.target_fpr,
        "achieved_fpr_at_threshold": deploy_fpr,
        "achieved_tpr_at_threshold": deploy_tpr,
    }
    with open(args.output_model, "w") as f:
        json.dump(model_json, f, indent=2)
    print(f"\nSaved trained Stage-2 classifier -> {args.output_model}")

    # ---- Plot --------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(fpr_a, tpr_a, label=f"A) Stage 1 only (AUC={auc_a:.3f}, EER={eer_a:.1%})", linewidth=2)
        ax.plot(fpr_c, tpr_c, label=f"C) Learned classifier, 5-fold CV (AUC={auc_c:.3f}, EER={eer_c:.1%})", linewidth=2)
        ax.scatter([rule_result["fpr"]], [1 - rule_result["fnr"]], color="red", zorder=5, s=70,
                   label=f"B) Hand-tuned rule (config.py) - FPR={rule_result['fpr']:.1%}, FNR={rule_result['fnr']:.1%}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Stage 1 vs. hand-tuned rule vs. learned Stage 2 classifier")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=150)
        print(f"Saved ROC comparison plot -> {args.output_plot}")
    except ImportError:
        print("matplotlib not available - skipped saving the ROC plot.")

    # ---- Report --------------------------------------------------------------
    report = f"""# Stage-2 Evaluation Report

Generated from {len(X)} held-out pairs ({int(y.sum())} genuine, {int((1 - y).sum())} impostor),
mined from `{args.test_dir}` against the training database at `{args.db}`.
See `src/pairs.py` for the pair-mining methodology (impostors are the
*hardest* wrong candidate per image, not random - this is a realistic,
not inflated, evaluation).

## A) Stage 1 only (appearance embedding, no geometry)
- AUC: {auc_a:.4f}
- EER: {eer_a:.2%}

## B) Current hand-tuned rule (config.py thresholds)
- False Positive Rate: {rule_result['fpr']:.2%}
- False Negative Rate: {rule_result['fnr']:.2%}
- Accuracy: {rule_result['accuracy']:.2%}
- Confusion: TP={rule_result['tp']} FP={rule_result['fp']} TN={rule_result['tn']} FN={rule_result['fn']}

## C) Learned classifier (logistic regression, embedding + landmark distance)
- 5-fold cross-validated AUC: {auc_c:.4f}
- 5-fold cross-validated EER: {eer_c:.2%}
- Operating point at target FPR={args.target_fpr:.1%}: achieved FPR={deploy_fpr:.2%}, TPR={deploy_tpr:.2%}
- Final model (refit on all {len(X)} pairs), saved to `{os.path.basename(args.output_model)}`:
  - w_embedding_distance = {w_embedding:.4f}
  - w_landmark_distance  = {w_landmark:.4f}
  - bias                 = {bias:.4f}

## Interpretation
Comparing A vs C's EER shows how much the landmark-geometry signal adds
beyond appearance alone. Comparing B vs C shows how much a properly fit
decision boundary improves on hand-picked thresholds over the same two
features. Set `config.USE_LEARNED_STAGE2 = True` to have `FaceVerifier`
use this trained classifier instead of the nested-threshold rule.
"""
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Saved evaluation report -> {args.output_report}")
    print("\n" + report)


if __name__ == "__main__":
    main()
