"""
The Accuracy-Transparency Trade-Off in Responsible AI
======================================================
Dataset: UCI Adult Census Income (48,842 rows) - high-stakes income prediction.

Answers 5 research questions, each with one publication-style figure (PDF)
and one result table (CSV):
  RQ1: Performance gap between interpretable and black-box models
  RQ2: Diminishing returns of model complexity
  RQ3: Feature-importance agreement between transparent and black-box models
  RQ4: Subgroup (sex, race) uniformity of the black-box accuracy advantage
  RQ5: Surrogate distillation - fidelity vs. transparency of a compact tree

All randomness is seeded (SEED=42) so outputs are fully reproducible.
Runs locally and on Kaggle (pure scikit-learn).
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

SEED = 42
FIG_DIR = os.path.join("outputs", "figures")
TAB_DIR = os.path.join("outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Poster-quality plot defaults: vector PDF output, readable fonts, no chartjunk
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,          # keep text editable/selectable in PDF
    "figure.constrained_layout.use": True,
})

# Colorblind-safe palette: cool = interpretable, warm = black-box
C_LR, C_DT, C_RF, C_GB = "#0173B2", "#029E73", "#CC78BC", "#D55E00"


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  saved figure: {name}.pdf / .png")


def save_table(df, name):
    df.to_csv(os.path.join(TAB_DIR, name + ".csv"), index=False)
    print(f"  saved table:  {name}.csv")


# ----------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ----------------------------------------------------------------------------
print("Loading Adult Census Income dataset...")

KAGGLE_CSV = "/kaggle/input/adult-census-income/adult.csv"
if os.path.exists(KAGGLE_CSV):
    raw = pd.read_csv(KAGGLE_CSV).replace("?", np.nan)
    raw = raw.rename(columns={"education.num": "education-num",
                              "marital.status": "marital-status",
                              "capital.gain": "capital-gain",
                              "capital.loss": "capital-loss",
                              "hours.per.week": "hours-per-week",
                              "native.country": "native-country",
                              "income": "class"})
else:
    raw = fetch_openml("adult", version=2, as_frame=True).frame

# fnlwgt is a census sampling weight (not a person-level attribute) and
# education duplicates education-num as text -> both dropped
df = raw.drop(columns=["fnlwgt", "education"])
y = (df.pop("class").astype(str).str.strip() == ">50K").astype(int)

CAT_COLS = ["workclass", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss",
            "hours-per-week"]
X = df[CAT_COLS + NUM_COLS].copy()
X[CAT_COLS] = X[CAT_COLS].astype(object)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"  {len(X_train)} train / {len(X_test)} test rows, "
      f"positive rate = {y.mean():.3f}")

preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CAT_COLS),
    ("num", StandardScaler(), NUM_COLS),
])


def make_pipe(model):
    return Pipeline([("prep", preprocess), ("model", model)])


# Interpretable models (cool colors) vs black-box ensembles (warm colors)
MODELS = {
    "Logistic Regression": (make_pipe(LogisticRegression(max_iter=1000,
                                                         random_state=SEED)),
                            "interpretable", C_LR),
    "Decision Tree (depth 4)": (make_pipe(DecisionTreeClassifier(max_depth=4,
                                                                 random_state=SEED)),
                                "interpretable", C_DT),
    "Random Forest": (make_pipe(RandomForestClassifier(n_estimators=300,
                                                       random_state=SEED)),
                      "black-box", C_RF),
    "Gradient Boosting": (make_pipe(HistGradientBoostingClassifier(random_state=SEED)),
                          "black-box", C_GB),
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def n_decision_units(name, pipe):
    """Complexity proxy: how many learned units (weights or tree leaves) a
    human must inspect to fully understand the model."""
    m = pipe.named_steps["model"]
    if name == "Logistic Regression":
        return m.coef_.size
    if name.startswith("Decision Tree"):
        return int(m.tree_.n_leaves)
    if name == "Random Forest":
        return int(sum(e.tree_.n_leaves for e in m.estimators_))
    try:  # HistGradientBoosting: count leaves over all boosting stages
        return int(sum(p.get_n_leaf_nodes()
                       for stage in m._predictors for p in stage))
    except AttributeError:
        return int(m.n_iter_ * m.max_leaf_nodes)


# ----------------------------------------------------------------------------
# RQ1: How large is the performance gap between interpretable models and
#      black-box ensembles?
# ----------------------------------------------------------------------------
print("\nRQ1: model comparison (5-fold CV + held-out test)...")
rows = []
fitted = {}
for name, (pipe, category, color) in MODELS.items():
    cv_res = cross_validate(pipe, X_train, y_train, cv=CV,
                            scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    fitted[name] = pipe
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "model": name,
        "category": category,
        "cv_accuracy_mean": cv_res["test_accuracy"].mean(),
        "cv_accuracy_std": cv_res["test_accuracy"].std(),
        "cv_f1_mean": cv_res["test_f1"].mean(),
        "cv_f1_std": cv_res["test_f1"].std(),
        "cv_auc_mean": cv_res["test_roc_auc"].mean(),
        "cv_auc_std": cv_res["test_roc_auc"].std(),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_auc": roc_auc_score(y_test, proba),
        "fit_time_s": fit_time,
        "n_decision_units": n_decision_units(name, pipe),
    })
    print(f"  {name:26s} test AUC={rows[-1]['test_auc']:.4f} "
          f"acc={rows[-1]['test_accuracy']:.4f} units={rows[-1]['n_decision_units']}")

rq1 = pd.DataFrame(rows).round(4)
save_table(rq1, "rq1_model_performance")

# Per-model label offsets so annotations stay clear of axes and edges
LABEL_POS = {"Logistic Regression": ((0, 12), "center"),
             "Decision Tree (depth 4)": ((14, 4), "left"),
             "Random Forest": ((-14, 4), "right"),
             "Gradient Boosting": ((0, 12), "center")}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for _, r in rq1.iterrows():
    color = MODELS[r["model"]][2]
    marker = "o" if r["category"] == "interpretable" else "s"
    ax.errorbar(r["n_decision_units"], r["test_auc"], yerr=r["cv_auc_std"],
                fmt=marker, ms=11, color=color, capsize=4, lw=1.5)
    offset, ha = LABEL_POS[r["model"]]
    ax.annotate(r["model"], (r["n_decision_units"], r["test_auc"]),
                textcoords="offset points", xytext=offset, ha=ha,
                fontsize=9.5)
ax.set_xscale("log")
ax.set_xlabel("Model complexity: decision units a human must inspect (log scale)")
ax.set_ylabel("Test ROC-AUC (error bar: 5-fold CV std)")
ax.set_title("RQ1: The Accuracy-Transparency Pareto Landscape")
ax.set_ylim(rq1["test_auc"].min() - 0.02, rq1["test_auc"].max() + 0.015)
save_fig(fig, "rq1_accuracy_transparency_pareto")

# ----------------------------------------------------------------------------
# RQ2: Where do additional complexity and lost transparency stop paying off?
# ----------------------------------------------------------------------------
print("\nRQ2: decision-tree depth sweep...")
depths = list(range(1, 21))
sweep = []
for d in depths:
    pipe = make_pipe(DecisionTreeClassifier(max_depth=d, random_state=SEED))
    cv_res = cross_validate(pipe, X_train, y_train, cv=CV,
                            scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)
    sweep.append({
        "max_depth": d,
        "cv_auc_mean": cv_res["test_score"].mean(),
        "cv_auc_std": cv_res["test_score"].std(),
        "test_auc": roc_auc_score(y_test,
                                  pipe.predict_proba(X_test)[:, 1]),
        "n_leaves": int(pipe.named_steps["model"].tree_.n_leaves),
    })
rq2 = pd.DataFrame(sweep)

# Sweet spot: smallest depth within 0.002 AUC of the best CV mean
best = rq2["cv_auc_mean"].max()
sweet = rq2[rq2["cv_auc_mean"] >= best - 0.002].iloc[0]
rq2["is_sweet_spot"] = rq2["max_depth"] == sweet["max_depth"]
save_table(rq2.round(4), "rq2_complexity_sweep")
print(f"  sweet spot: depth={int(sweet['max_depth'])} "
      f"(CV AUC {sweet['cv_auc_mean']:.4f}, best {best:.4f})")

lr_auc = rq1.loc[rq1["model"] == "Logistic Regression", "test_auc"].iloc[0]
gb_auc = rq1.loc[rq1["model"] == "Gradient Boosting", "test_auc"].iloc[0]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(rq2["max_depth"], rq2["cv_auc_mean"], "-o", color=C_DT, ms=5,
        label="Decision tree (5-fold CV AUC)")
ax.fill_between(rq2["max_depth"],
                rq2["cv_auc_mean"] - rq2["cv_auc_std"],
                rq2["cv_auc_mean"] + rq2["cv_auc_std"],
                color=C_DT, alpha=0.15)
ax.axhline(gb_auc, color=C_GB, ls="--", lw=1.5,
           label=f"Gradient Boosting ceiling ({gb_auc:.3f})")
ax.axhline(lr_auc, color=C_LR, ls=":", lw=1.5,
           label=f"Logistic Regression ({lr_auc:.3f})")
ax.axvline(sweet["max_depth"], color="gray", ls="--", lw=1)
ax.annotate(f"diminishing returns\nbeyond depth {int(sweet['max_depth'])}\n"
            f"({int(sweet['n_leaves'])} leaves)",
            (sweet["max_depth"], sweet["cv_auc_mean"]),
            textcoords="offset points", xytext=(14, -52), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Decision tree max depth (lower = more transparent)")
ax.set_ylabel("ROC-AUC")
ax.set_title("RQ2: Diminishing Returns of Model Complexity")
ax.set_xticks(depths[::2])
ax.legend(loc="lower right", fontsize=9)
save_fig(fig, "rq2_complexity_sweep")

# ----------------------------------------------------------------------------
# RQ3: Do transparent and black-box models rely on the same features?
#      Same attribution method (permutation importance) for both models so
#      the comparison is not confounded by the explanation technique.
# ----------------------------------------------------------------------------
print("\nRQ3: permutation feature importance (10 repeats each)...")
imp = {}
for name in ["Logistic Regression", "Gradient Boosting"]:
    res = permutation_importance(fitted[name], X_test, y_test,
                                 scoring="roc_auc", n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    imp[name] = pd.Series(res.importances_mean, index=X.columns)

rho, pval = spearmanr(imp["Logistic Regression"], imp["Gradient Boosting"])
rq3 = pd.DataFrame({
    "feature": X.columns,
    "importance_logreg": imp["Logistic Regression"].values,
    "rank_logreg": imp["Logistic Regression"].rank(ascending=False).astype(int).values,
    "importance_gb": imp["Gradient Boosting"].values,
    "rank_gb": imp["Gradient Boosting"].rank(ascending=False).astype(int).values,
}).sort_values("importance_gb", ascending=False)
rq3["spearman_rho_all_features"] = round(rho, 4)
save_table(rq3.round(4), "rq3_feature_importance_agreement")
print(f"  Spearman rank correlation = {rho:.3f} (p={pval:.4f})")

top = rq3.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ypos = np.arange(len(top))
ax.barh(ypos + 0.2, top["importance_logreg"], height=0.38, color=C_LR,
        label="Logistic Regression (transparent)")
ax.barh(ypos - 0.2, top["importance_gb"], height=0.38, color=C_GB,
        label="Gradient Boosting (black-box)")
ax.set_yticks(ypos)
ax.set_yticklabels(top["feature"])
ax.set_xlabel("Permutation importance (mean ROC-AUC drop, 10 repeats)")
ax.set_title(f"RQ3: Feature Reliance Agreement  (Spearman ρ = {rho:.2f})")
ax.legend(fontsize=9, loc="lower right")
save_fig(fig, "rq3_feature_importance_agreement")

# ----------------------------------------------------------------------------
# RQ4: Is the black-box accuracy advantage uniform across demographic
#      subgroups (sex, race)?
# ----------------------------------------------------------------------------
print("\nRQ4: subgroup analysis...")
proba_lr = fitted["Logistic Regression"].predict_proba(X_test)[:, 1]
proba_gb = fitted["Gradient Boosting"].predict_proba(X_test)[:, 1]

groups = [("overall", "All", np.ones(len(X_test), dtype=bool))]
for attr in ["sex", "race"]:
    for val in X_test[attr].dropna().unique():
        groups.append((attr, str(val), (X_test[attr] == val).values))

rows = []
for attr, val, mask in groups:
    if mask.sum() < 30 or y_test[mask].nunique() < 2:
        continue
    auc_lr = roc_auc_score(y_test[mask], proba_lr[mask])
    auc_gb = roc_auc_score(y_test[mask], proba_gb[mask])
    rows.append({"attribute": attr, "group": val, "n_test": int(mask.sum()),
                 "positive_rate": y_test[mask].mean(),
                 "auc_logreg": auc_lr, "auc_gb": auc_gb,
                 "blackbox_advantage": auc_gb - auc_lr})
rq4 = pd.DataFrame(rows)
save_table(rq4.round(4), "rq4_subgroup_performance")
print(rq4[["attribute", "group", "n_test", "auc_logreg", "auc_gb",
           "blackbox_advantage"]].round(4).to_string(index=False))

plot4 = rq4[(rq4["attribute"] == "overall") | (rq4["n_test"] >= 150)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)
ypos = np.arange(len(plot4))[::-1]
labels = [f"{g}\n(n={n:,})" for g, n in zip(plot4["group"], plot4["n_test"])]
for yp, (_, r) in zip(ypos, plot4.iterrows()):
    ax1.plot([r["auc_logreg"], r["auc_gb"]], [yp, yp], color="gray",
             lw=1.5, zorder=1)
ax1.scatter(plot4["auc_logreg"], ypos, s=70, color=C_LR, zorder=2,
            label="Logistic Regression")
ax1.scatter(plot4["auc_gb"], ypos, s=70, color=C_GB, zorder=2,
            label="Gradient Boosting")
ax1.set_yticks(ypos)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel("ROC-AUC")
ax1.set_title("Subgroup performance")
ax1.legend(fontsize=8.5, loc="lower left")

colors = ["#555555" if a == "overall" else C_GB
          for a in plot4["attribute"]]
ax2.barh(ypos, plot4["blackbox_advantage"], color=colors, height=0.55)
overall_adv = rq4.loc[rq4["attribute"] == "overall",
                      "blackbox_advantage"].iloc[0]
ax2.axvline(overall_adv, color="black", ls="--", lw=1,
            label=f"overall advantage ({overall_adv:.3f})")
ax2.set_xlabel("Black-box AUC advantage (GB − LR)")
ax2.set_title("Is the advantage uniform?")
ax2.legend(fontsize=8.5)
fig.suptitle("RQ4: The Fairness Dimension of the Trade-Off",
             fontweight="bold")
save_fig(fig, "rq4_subgroup_fairness")

# ----------------------------------------------------------------------------
# RQ5: How faithfully can a compact tree mimic the black-box (distillation)?
# ----------------------------------------------------------------------------
print("\nRQ5: surrogate distillation...")
teacher = fitted["Gradient Boosting"]
teacher_train = teacher.predict(X_train)
teacher_test = teacher.predict(X_test)
teacher_acc = accuracy_score(y_test, teacher_test)

rows = []
for d in range(1, 13):
    surr = make_pipe(DecisionTreeClassifier(max_depth=d, random_state=SEED))
    surr.fit(X_train, teacher_train)          # trained to mimic the teacher
    pred = surr.predict(X_test)
    rows.append({
        "max_depth": d,
        "n_leaves": int(surr.named_steps["model"].tree_.n_leaves),
        "fidelity_to_blackbox": accuracy_score(teacher_test, pred),
        "surrogate_test_accuracy": accuracy_score(y_test, pred),
        "teacher_test_accuracy": teacher_acc,
    })
rq5 = pd.DataFrame(rows)
save_table(rq5.round(4), "rq5_surrogate_distillation")
hi_fid = rq5[rq5["fidelity_to_blackbox"] >= 0.95].iloc[0] \
    if (rq5["fidelity_to_blackbox"] >= 0.95).any() else rq5.iloc[-1]
print(f"  depth {int(hi_fid['max_depth'])} surrogate reproduces "
      f"{hi_fid['fidelity_to_blackbox']:.1%} of black-box decisions")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(rq5["max_depth"], rq5["fidelity_to_blackbox"], "-o", color=C_GB,
        ms=5, label="Fidelity to black-box (agreement)")
ax.plot(rq5["max_depth"], rq5["surrogate_test_accuracy"], "-s", color=C_DT,
        ms=5, label="Surrogate accuracy (true labels)")
ax.axhline(teacher_acc, color="gray", ls="--", lw=1.5,
           label=f"Black-box accuracy ({teacher_acc:.3f})")
ax.annotate(f"depth {int(hi_fid['max_depth'])} tree "
            f"({int(hi_fid['n_leaves'])} leaves)\nreproduces "
            f"{hi_fid['fidelity_to_blackbox']:.0%} of decisions",
            (hi_fid["max_depth"], hi_fid["fidelity_to_blackbox"]),
            textcoords="offset points", xytext=(10, -45), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Surrogate tree max depth")
ax.set_ylabel("Score on test set")
ax.set_title("RQ5: Distilling the Black-Box into a Transparent Surrogate")
ax.legend(fontsize=9, loc="lower right")
save_fig(fig, "rq5_surrogate_distillation")

# ----------------------------------------------------------------------------
# Key findings summary (for poster text written by the student)
# ----------------------------------------------------------------------------
dt4_auc = rq1.loc[rq1["model"] == "Decision Tree (depth 4)", "test_auc"].iloc[0]
gap_pp = (gb_auc - lr_auc) * 100
summary = f"""KEY NUMBERS (test set, seed={SEED})
RQ1  Gradient Boosting AUC {gb_auc:.4f} vs Logistic Regression {lr_auc:.4f}
     -> black-box advantage = {gap_pp:.2f} AUC points; depth-4 tree {dt4_auc:.4f}
RQ2  sweet spot at depth {int(sweet['max_depth'])} ({int(sweet['n_leaves'])} leaves),
     CV AUC {sweet['cv_auc_mean']:.4f} vs best tree {best:.4f}
RQ3  Spearman rank agreement between model explanations: rho = {rho:.3f}
RQ4  black-box advantage overall {overall_adv:.4f} AUC; per-group range
     {rq4.loc[rq4['attribute'] != 'overall', 'blackbox_advantage'].min():.4f} .. {rq4.loc[rq4['attribute'] != 'overall', 'blackbox_advantage'].max():.4f}
RQ5  depth-{int(hi_fid['max_depth'])} surrogate: fidelity {hi_fid['fidelity_to_blackbox']:.4f},
     accuracy {hi_fid['surrogate_test_accuracy']:.4f} (teacher {teacher_acc:.4f})
"""
with open(os.path.join("outputs", "key_findings.txt"), "w") as f:
    f.write(summary)
print("\n" + summary)
print("Done. All figures (PDF+PNG) and tables (CSV) are in outputs/.")
