# Recys_HW1

Introduction to Recommender Systems (Fall 2026) — Homework #1

Experiments with [RecBole](https://github.com/RUCAIBox/RecBole) on the `ml-100k` dataset.

1. Neighborhood-based CF: ItemKNN / UserKNN, k ∈ {1, 5, 10, 100, 500}
2. Latent factor model: BPR, embedding size ∈ {32, 64, 128, 256}, regularization ∈ {0.01, 0.1, 0.5}
3. Item-to-item CF: SLIM (SLIMElastic), EASE vs. ItemKNN

## Results

Top-10 metrics on ml-100k, best setting for each model.

| Model | Setting | NDCG@10 | Recall@10 |
|---|---|---|---|
| ItemKNN (item) | k = 100 | 0.281 | 0.240 |
| ItemKNN (user) | k = 100 | 0.282 | 0.240 |
| BPR + L2 | dim 64, λ = 0.01 | 0.304 | 0.253 |
| SLIMElastic | default | 0.326 | 0.273 |
| EASE | default | 0.324 | 0.278 |

KNN did best around k = 100 for both the item-based and user-based versions, and got worse again at k = 500. The user-based version was much weaker with very few neighbors (NDCG@10 of 0.111 at k = 1, compared to 0.204 for item-based).

For BPR, the embedding size barely mattered (32 to 256 moved NDCG@10 by less than 0.006), but regularization made a big difference: NDCG@10 dropped from 0.304 at λ = 0.01 to 0.116 at λ = 0.5. In hindsight the embedding-size sweep should have used a smaller λ, since 0.1 turned out not to be a good value.

SLIM and EASE came out on top, roughly 15% higher NDCG@10 than the best KNN setting.

| k for KNN | λ for BPR |
|---|---|
| ![](results/fig1_knn_k.png) | ![](results/fig3_bpr_reg.png) |

## Setup

```bash
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -r requirements.txt
```

## Run

```bash
python run_experiments.py          # all experiments -> results/results.csv
python run_experiments.py --exp knn  # or: bpr, i2i
python plot_results.py             # figures -> results/*.png, prints the (3) table
```

`ml-100k` ships with RecBole (`recbole/dataset_example/ml-100k`), so no download is needed.
Built-in models (ItemKNN, SLIMElastic, EASE) are run through `recbole.quick_start.run_recbole()`
exactly as in the RecBole README; the custom BPR model goes through the same steps with the
model class passed to `Config` directly, since `run_recbole()` only accepts a model name.
Evaluation uses RecBole defaults: random 8:1:1 split, full ranking, top-10 metrics, seed 2026.

## Notes

- UserKNN is `ItemKNN` with `knn_method: 'user'` (the `--method` flag in the
  handout is not a RecBole config key and is silently ignored).
- RecBole's `BPR` has no `reg_lambda`; `models/bpr_reg.py` adds an L2 term
  `reg_lambda * (||u||² + ||i||²)` to the BPR loss so the regularization strength
  can be swept.
- SLIM is RecBole's `SLIMElastic`; SLIM and EASE use the default hyperparameters.

## Files

```
models/bpr_reg.py     BPR with L2 regularization
run_experiments.py    runs every configuration, one CSV row per run
plot_results.py       figures for the report
results/              results.csv and figures
```
