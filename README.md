# The Accuracy–Transparency Trade-Off in Responsible AI

Individual applied research project for the module **Governing Intelligent
Systems for Sustainability**.

**Student:** Umut Türklay (49848742)

## Research Questions

1. **RQ1** — How large is the predictive performance gap between inherently
   interpretable models (logistic regression, shallow decision tree) and
   black-box ensembles (random forest, gradient boosting) on a high-stakes
   income prediction task?
2. **RQ2** — How does predictive performance scale with model complexity, and
   at what point do additional complexity and reduced transparency stop
   yielding meaningful accuracy gains?
3. **RQ3** — Do transparent and black-box models attribute their predictions
   to the same features, and how strongly do their feature-importance rankings
   agree?
4. **RQ4** — Does the accuracy advantage of black-box models hold uniformly
   across demographic subgroups (sex, race), or does the accuracy–transparency
   trade-off carry a fairness dimension?
5. **RQ5** — How faithfully can a black-box model be approximated by a compact
   interpretable surrogate (distilled decision tree), and how much accuracy is
   sacrificed in the distillation?

## Dataset

[UCI Adult Census Income](https://www.openml.org/d/1590) — 48,842 rows,
12 features after cleaning, binary target (income > 50K, positive rate 23.9%).
Loaded automatically from OpenML (or from the Kaggle copy when run on Kaggle).

## Repository structure

```
adult_accuracy_transparency.ipynb  main notebook, executed with all outputs
analysis.py                        same pipeline as a plain script
outputs/figures/                   one vector-PDF figure per RQ
outputs/tables/                    one CSV result table per RQ
outputs/key_findings.txt           headline numbers of all five RQs
```

## How to run

```bash
pip install scikit-learn pandas matplotlib scipy
python3 analysis.py        # or run the notebook top to bottom
```

Every random component is seeded (`SEED = 42`); rerunning reproduces all
figures and tables exactly.

## Key results

| Model | Test ROC-AUC | Test accuracy | Decision units |
|---|---|---|---|
| Decision Tree (depth 4) | 0.874 | 0.846 | 16 |
| Logistic Regression | 0.905 | 0.855 | 91 |
| Random Forest | 0.897 | 0.852 | 2,013,240 |
| Gradient Boosting | **0.930** | **0.875** | 2,635 |

- The black-box advantage is real but modest: **2.48 AUC points** (RQ1), and
  complexity alone does not buy accuracy — Random Forest uses ~2M leaves yet
  trails logistic regression.
- A depth-8 tree (132 leaves) already matches logistic regression; deeper
  trees overfit (RQ2).
- Model explanations agree strongly (Spearman ρ = 0.84) but the black-box
  exploits the non-linear age effect that a linear model cannot (RQ3).
- The black-box advantage is **not uniform**: 0.031 AUC for men vs 0.018 for
  women (RQ4).
- A depth-6 surrogate (47 leaves) reproduces **95.4%** of the black-box
  decisions at 0.861 accuracy vs the teacher's 0.875 (RQ5).

## AI usage

AI tools (Claude) were used in line with the module instructions: refining
research questions, generating and iterating on the analysis code, and
drafting the workflow layout. All code was executed and verified ; all results come from real data.
