# DSA4060XA-Week6-670178

**Name:** Nicholas Kinyanjui
**Student ID:** 670178
**Course:** DSA 4060XA (Week 6 Guided Evaluation Lab)

## Objective

Compare three movie recommenders on rating prediction, ranking relevance, and discovery. Recommend a model for a streaming service that wants relevant movies without the same popular titles dominating every list.

## Files

- `DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb` — implementation, outputs, discussion answers, and extension A.
- `DSA4060_Week6_Movies.csv` — 20 movie IDs, titles, and pipe-separated genres.
- `DSA4060_Week6_Ratings.csv` — 120 ratings from 12 users, with 10 ratings per user.
- `requirements.txt` — Python dependencies.

## Method and assumptions

The supplied lab extract begins midway through Step 11. The earlier setup and model definitions are reconstructed and explicitly identified in the notebook. Results may differ from the original handout if its missing sections prescribe different definitions.

Use a per-user random 80/20 split with NumPy seed 42: eight training and two test ratings per user, giving 96 training and 24 test ratings. Train-only statistics prevent test-rating leakage. Catalogue genres are available to the content model. No hyperparameters are selected on the test data.

1. **Popularity baseline:** training mean per movie; ties use rating count then movie ID. Unknown movies fall back to the global training mean.
2. **Content-based:** binary genre features with cosine similarity; predict a similarity-weighted average of the user's training ratings. Fall back to the baseline when there is no genre overlap.
3. **Item-based CF:** cosine between movie vectors from the training user–item matrix, with missing entries represented by zero for similarity only. Predict from observed user ratings weighted by positive item similarities. Fall back to the baseline for zero total weight.

For ranking, candidates are all movies unseen in training. A held-out rating of 4 or 5 is relevant. Following the lab, ranking averages exclude users with no relevant held-out movie: **7 of 12 users, with 9 relevant items**, enter ranking evaluation. Coverage, novelty, and diversity include all 12 users. Unrated candidates have unknown relevance and count as non-hits. Ties are deterministic.

## Results

Lower MAE/RMSE is better. Higher ranking scores, coverage, novelty, and diversity are preferable within their respective objectives.

| Model               |    MAE |   RMSE | Precision@3 | Recall@3 | NDCG@3 | Coverage | Novelty (bits) | Diversity |
| ------------------- | -----: | -----: | ----------: | -------: | -----: | -------: | -------------: | --------: |
| Popularity Baseline | 0.7523 | 0.9204 |      0.0952 |   0.1429 | 0.0991 |      45% |         1.5304 |    0.9444 |
| Content-Based       | 0.7405 | 0.8496 |      0.2381 |   0.5714 | 0.3644 |      80% |         1.5332 |    0.5071 |
| Item-Based CF       | 0.8584 | 1.0195 |      0.0476 |   0.1429 | 0.0714 |      75% |         1.6280 |    0.7917 |

**Best for rating prediction:** Content-based has the lowest MAE and RMSE. Its MAE improvement over popularity is only 0.0118; this single small split does not establish a statistically clear advantage. CF performs worse than popularity on both metrics.

**Best for ranking:** Content-based leads all three ranking metrics on this split. Rating accuracy and ranking quality answer different questions; a winner should not be declared from one metric alone.

**Coverage:** Content exposes 16 distinct movies, CF 15, and popularity 9, across all users' Top-3 lists. Coverage measures distinct titles, not the evenness of exposure.

**Novelty:** CF has the highest average novelty. Content's 1.5332 bits is almost identical to popularity's 1.5304, so there is no meaningful observed rarity improvement over popularity. Novelty uses training interaction frequency, with a fallback count of one for unseen movies, as specified by the lab.

**Diversity:** Popularity produces the most genre-diverse lists, CF is intermediate, and content the least diverse. Diversity is one minus mean pairwise genre cosine similarity within each list, averaged over users.

## Final recommended model

Choose **content-based recommendations as the initial model for this dataset**, guided primarily by **NDCG@3, coverage, and novelty**. Content has the strongest observed relevance and the widest catalogue exposure. Novelty tempers the discovery claim: it is effectively unchanged from popularity and lower than CF.

The accepted trade-off is lower within-list genre diversity and somewhat lower novelty than CF in exchange for stronger relevance and broader exposure. A future diversity reranker and explicit exposure-concentration metric could address repeated genre/popularity concentration, but neither has been evaluated here. Treat this as a provisional offline decision.

## Extension A  change K

| Model               | K | Precision@K | Recall@K | NDCG@K |
| ------------------- | -: | ----------: | -------: | -----: |
| Popularity Baseline | 1 |      0.0000 |   0.0000 | 0.0000 |
| Popularity Baseline | 3 |      0.0952 |   0.1429 | 0.0991 |
| Popularity Baseline | 5 |      0.0571 |   0.1429 | 0.0991 |
| Content-Based       | 1 |      0.1429 |   0.0714 | 0.1429 |
| Content-Based       | 3 |      0.2381 |   0.5714 | 0.3644 |
| Content-Based       | 5 |      0.2000 |   0.8571 | 0.4812 |
| Item-Based CF       | 1 |      0.0000 |   0.0000 | 0.0000 |
| Item-Based CF       | 3 |      0.0476 |   0.1429 | 0.0714 |
| Item-Based CF       | 5 |      0.0571 |   0.2143 | 0.1092 |

Content recall rises as more positions recover relevant movies, while precision peaks at K=3 and falls at K=5. Popularity finds no additional relevant movie between K=3 and K=5, so recall stays constant and precision falls. CF recall increases with K. Precision need not decrease monotonically: later positions may improve on a poor first recommendation. Nested lists ensure recall cannot decrease.

## Limitations

1. Only 12 users, 20 movies, and 24 test ratings; one split is unstable. No repeated-split uncertainty estimates are provided.
2. Unrated candidates have unknown relevance; non-hit treatment can undercount useful discoveries. Offline metrics do not measure satisfaction or engagement.
3. Excluding five users without relevant held-out items changes the population represented by ranking averages.
4. Genre features are coarse; raw-rating cosine CF is sensitive to sparse overlap, popularity, and different rating habits. Cold-user behavior is not evaluated.
5. The random split is not a future-performance test. Timestamp values 1–120 are not verified real dates.
6. Coverage does not measure exposure concentration. Novelty is a frequency proxy, not user-specific unfamiliarity.
7. Missing Steps 1–11 require documented assumptions; the complete original instructions may change the results.

## Run and reproduce

Use Python 3.12. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Open the notebook, select the Python environment, then **Restart Kernel and Run All Cells**. Run from this folder so the two CSV paths resolve. The notebook prints Python, NumPy, and pandas versions and includes checks for split integrity, unseen-only recommendations, metric definitions, valid prediction ranges, and nondecreasing recall as K grows.

Calculation results were checked with Python 3.12.14, NumPy 2.3.5, and pandas 3.0.1. The notebook contains all outputs after a fresh-kernel execution.

## GitHub submission

The required repository name is **`DSA4060XA-Week6-670178`**. Upload the notebook, both CSV files, README, and requirements file. Exclude `.venv`, `.lab_tools`, and notebook checkpoints. This folder is prepared locally; no GitHub repository has been published by this workflow.
