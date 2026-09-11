# Stage-2 Evaluation Report

Generated from 1009 held-out pairs (507 genuine, 502 impostor),
mined from `/home/claude/updated_project/face_recognition_project/data/test` against the training database at `/home/claude/updated_project/face_recognition_project/data/embeddings_db.pkl`.
See `src/pairs.py` for the pair-mining methodology (impostors are the
*hardest* wrong candidate per image, not random - this is a realistic,
not inflated, evaluation).

## A) Stage 1 only (appearance embedding, no geometry)
- AUC: 0.9929
- EER: 3.37%

## B) Current hand-tuned rule (config.py thresholds)
- False Positive Rate: 27.09%
- False Negative Rate: 23.27%
- Accuracy: 74.83%
- Confusion: TP=389 FP=136 TN=366 FN=118

## C) Learned classifier (logistic regression, embedding + landmark distance)
- 5-fold cross-validated AUC: 0.9924
- 5-fold cross-validated EER: 3.77%
- Operating point at target FPR=1.0%: achieved FPR=1.00%, TPR=92.70%
- Final model (refit on all 1009 pairs), saved to `stage2_classifier.json`:
  - w_embedding_distance = -13.4358
  - w_landmark_distance  = -0.9366
  - bias                 = 4.2282

## Interpretation
Comparing A vs C's EER shows how much the landmark-geometry signal adds
beyond appearance alone. Comparing B vs C shows how much a properly fit
decision boundary improves on hand-picked thresholds over the same two
features. Set `config.USE_LEARNED_STAGE2 = True` to have `FaceVerifier`
use this trained classifier instead of the nested-threshold rule.
