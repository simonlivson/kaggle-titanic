# Titanic: Beginner Gradient Boosting Pipeline

A beginner-friendly notebook for the [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) competition. It predicts whether a passenger survived, using a leak-free scikit-learn pipeline and comparing models with 5-fold stratified cross-validation.

This work was built with AI assistance. It was not written entirely by AI.

## Results

These figures come from the executed notebook and from Kaggle's official submission response. They are not hand-typed estimates.

- Winning model: Gradient Boosting, selected in code by mean cross-validation accuracy
- Mean 5-fold stratified CV accuracy: **0.8473** (std **0.0177**), printed by `titanic_survival.ipynb`
- Mean 5-fold CV ROC-AUC: **0.8822** (std **0.0165**), printed by the same notebook
- Public Kaggle leaderboard score: **0.77033**, returned by Kaggle for submission `58740009` on 22 November 2026 at 22:07 Israel time (UTC+2)

The CV score can be optimistic because the same folds were used to choose the model. Kaggle's public score is computed on a different set of passengers, so it can differ from local CV.

## Approach

1. Load `train.csv` and `test.csv` from the first folder that contains both files (`/kaggle/input/titanic` on Kaggle, or a local `data/` folder).
2. Explore survival by sex, class, and age, and inspect missing values.
3. Clean the data without leakage. Fill values are learned from training data only: age median by Pclass and Sex, embarked mode, fare median by Pclass, and a cabin deck letter with an unknown category.
4. Build features: title from the name (rare titles grouped on train), family size, an is-alone flag, and one-hot encoding fitted on train.
5. Compare a most-frequent baseline, logistic regression on Sex and Pclass only, full logistic regression, random forest, gradient boosting, and XGBoost with `StratifiedKFold` (5 splits, `random_state=42`).
6. Refit the winning model on all training rows and write `submission.csv`.

All random seeds are fixed at 42.

## How to run

### Locally

Download `train.csv` and `test.csv` from the [Titanic competition](https://www.kaggle.com/competitions/titanic/data) after accepting the rules, and place them in a `data/` folder next to the notebook.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m jupyter nbconvert --to notebook --execute --inplace titanic_survival.ipynb
```

You can also open `titanic_survival.ipynb` in Jupyter and use **Cell → Run All**.

### Kaggle

The public notebook is here:

https://www.kaggle.com/code/simonlivson/titanic-beginner-gradient-boosting-pipeline

Add the Titanic competition data as input. The notebook looks for files in `/kaggle/input/titanic` first. Use **Run All**. A CPU session is enough; GPU and internet are not required.

### Colab

Upload `train.csv` and `test.csv` next to the notebook or into a `data/` folder, then use **Runtime → Run all**.

## Files

- `titanic_survival.ipynb` — executed notebook with saved outputs
- `submission.csv` — test-set predictions submitted to Kaggle
- `requirements.txt` — Python packages used for this run
- `kernel-metadata.json` — metadata for pushing the notebook as a Kaggle kernel

The competition CSVs are not stored in this repository.

## Links

- Competition: https://www.kaggle.com/competitions/titanic
- Kaggle notebook: https://www.kaggle.com/code/simonlivson/titanic-beginner-gradient-boosting-pipeline
