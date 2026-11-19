# Credit-Card Fraud Detection

A local command-line tool that trains and saves a detector on chronological transaction data and exports a ranked queue for human review. The original Isolation Forest release is the benchmark for the supervised Random Forest and tuned Isolation Forest scorers.

`review_flag = 1` means a transaction's anomaly score met or exceeded a frozen review-capacity cutoff derived on validation data (`anomaly_score >= cutoff`). Anomaly scores express unusualness within the 29-feature space (`anomaly_score = -score_samples(X)`, larger means more unusual) and carry **no calibrated fraud probability**. Both high-scoring legitimate purchases and low-scoring frauds occur.

## Dataset and Tutorial Sources

- **Dataset:** Credit Card Fraud Detection, published by the Machine Learning Group (Université Libre de Bruxelles) and Worldline ([Kaggle dataset listing](https://www.kaggle.com/mlg-ulb/creditcardfraud/)).
- **Download mirror used:** `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` (Kaggle requires login; the file was retrieved from the public mirror linked by TensorFlow's imbalanced-data tutorial, preserved untouched at `data/creditcard.csv`, and verified against the publisher's reported totals).
- **License:** Database Contents License (DbCL) v1.0, as reported on the Kaggle listing.
- **Retrieved (UTC):** `2026-11-19T05:07:38+00:00`
- **File size:** `150,828,752` bytes
- **SHA-256:** `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- **Reference tutorial:** [Fraud Detection and Anomaly Detection by ChenDataBytes](https://medium.com/@chenycy/fraud-detection-and-anomaly-detection-e65dd11a3146). Whereas the tutorial scales columns before splitting and fixes an Isolation Forest contamination value, this project splits chronologically before fitting, uses continuous scores (`-score_samples(X)`), excludes `Time` and `Class` from the model matrix, and derives review cutoffs on validation data.

### Data Intake Audit (`data/creditcard.csv`)

| Audit check | Observed value | Notes |
| --- | --- | --- |
| Total rows | `284,807` | Matches publisher figure (`284,807`) |
| Confirmed frauds (`Class = 1`) | `492` (`0.1727%`) | Matches publisher figure (`492`) |
| Legitimate (`Class = 0`) | `284,315` (`99.8273%`) | Zero-alert rule is 99.83% accurate and catches 0 frauds |
| Missing / nonnumeric / infinite values | `0` across all 31 columns | Checked before splitting or publishing artifacts |
| `Time` range (seconds) | `0` to `172,792` (~48.0 hours) | Nonnegative; sorted deterministically by `(Time, row_number)` |
| `Amount` range | `0.00` to `25,691.16` | Nonnegative; preserved in raw scale |
| Exact duplicate full records (beyond first) | `1,081` (`1,854` rows in duplicate groups) | Kept as published; equal timestamps stay in the same partition |
| 30-field duplicates with conflicting `Class` | `0` groups (`0` rows) | No label conflicts observed in the downloaded file |

Repeated records are kept as published for the main evaluation so totals match the benchmark, and all rows sharing a timestamp remain in the same chronological partition. Repeated rows can still give identical transaction patterns extra weight during tree fitting and evaluation.

## Setup and Usage

Verified in a fresh virtual environment with Python `3.10.7` and the pinned packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Place `creditcard.csv` at `data/creditcard.csv`, then run training, held-out evaluation, and batch scoring:

```bash
python fraud_detection.py train --input data/creditcard.csv --run-dir outputs/baseline
python fraud_detection.py evaluate --input data/creditcard.csv --run-dir outputs/baseline
python fraud_detection.py score --input data/transactions_to_score.csv \
  --model outputs/baseline/model.joblib --output outputs/review_queue.csv
```

To create the unlabeled demonstration batch `data/transactions_to_score.csv` (last 1,000 rows of `data/creditcard.csv` with `Class` removed):

```bash
python -c "import pandas as pd; pd.read_csv('data/creditcard.csv', dtype=str).tail(1000).drop(columns=['Class']).to_csv('data/transactions_to_score.csv', index=False)"
```

> **Safety notes:** `train` refuses to overwrite an existing `--run-dir`. `evaluate` verifies the dataset SHA-256 and partition hashes and refuses to overwrite an existing `metrics.json`. `score` rejects output paths that resolve to `--input` or `--model` and only loads trusted local `model.joblib` files.

## Train and score a saved model

| Model choice | Historical CSV | Scoring CSV | Output score |
| --- | --- | --- | --- |
| `random_forest` | `Time`, `V1`–`V28`, `Amount`, binary `Class` | `V1`–`V28`, `Amount` | `fraud_score` |
| `if_combined` | `Time` and `V1`–`V28`. Optional binary `Class` supplies evaluation labels. | `V1`–`V28` | `anomaly_score` |

```bash
python fraud_detection.py train --model-kind random_forest \
  --input data/creditcard.csv --run-dir outputs/my-random-forest
python fraud_detection.py train --model-kind if_combined \
  --input data/creditcard.csv --run-dir outputs/my-if-combined
python fraud_detection.py score --input data/transactions_to_score.csv \
  --model outputs/my-random-forest/model.joblib --output outputs/my-random-forest/review_queue.csv
```

Pass either saved bundle to `score`. Its stored feature list and fixed 0.5% validation-budget cutoff control scoring. The queue contains the original CSV row number, model score, and review flag. Random Forest uses the class-1 score for ranking. Tuned Isolation Forest fits on feature values alone and uses its anomaly score for ranking.

With a labeled historical file, run `evaluate --input <same-csv> --run-dir <run-directory>` once to report metrics on the final chronological partition. For a run trained with an unlabeled file, use the validation alert counts and scored queue until outcomes are available for a separate evaluation. The original baseline command still trains the original Isolation Forest when `--model-kind` is omitted.

### Saved benchmark models

The following bundles are ready to load with `score --model`. The cutoff was frozen on validation for a 0.5% review target. The final benchmark partition contained 75 frauds.

| Bundle | Alerts on test | Frauds caught | Recall | Frauds in fixed top 284 |
| --- | ---: | ---: | ---: | ---: |
| [Random Forest](outputs/cli-random-forest/model.joblib) | 225 | 61 | 81.3% | 62 |
| [Tuned Isolation Forest](outputs/cli-if-combined/model.joblib) | 179 | 23 | 30.7% | 33 |

The original baseline caught 4 frauds at the same fixed top 284 review capacity. The hybrid caught 64 and tied plain Random Forest at the top 56. Rolling improved tuned-model recall in the final benchmark window while issuing 100 more alerts, and recall decreased in both development windows. Both saved bundles use fixed cutoffs. These are exploratory results on the previously inspected two-day benchmark. See the [change report](change-report.md) for the replay and validation details.

## Artifacts and Output Definitions

| Artifact | Contents |
| --- | --- |
| `outputs/baseline/model.joblib` | Fitted `IsolationForest`, ordered 29-feature list (`V1`–`V28`, `Amount`), full-precision validation cutoffs, hyperparameters, library versions, dataset SHA-256, and per-partition `row_number` hashes. |
| `outputs/baseline/run.json` | Dataset source and fingerprint, intake audit, chronological split manifest, train/validation inspection (amount quantiles and hourly fraud counts), model settings, environment versions, cutoff rule, and validation metrics. |
| `outputs/baseline/development_overview.png` | Train + validation amount distribution by class and labeled frauds per hour with the validation start line. |
| `outputs/baseline/validation_precision_recall.png` | Validation precision–recall curves for Isolation Forest and Amount-only with the validation prevalence reference line. |
| `outputs/baseline/metrics.json` | Held-out test interval and class counts, AP and no-skill reference, frozen-cutoff confusion counts and workload metrics, fixed-size `precision@K`/`recall@K`, train/val/test score quantiles, alert-rate drift, and error analysis. |
| `outputs/baseline/precision_recall.png` | Held-out test precision–recall curves with the test prevalence (`0.0013`) horizontal no-skill reference line. |
| `outputs/review_queue.csv` | Ranked scoring output with columns `row_number` (1-based input record index after header), `anomaly_score` (full round-trip precision), and `review_flag` (`1` if `anomaly_score >= cutoff`, else `0`), sorted by `anomaly_score` descending then `row_number` ascending. |

## Experiment Design

### Chronological Split

Rows are assigned a 1-based `row_number`, sorted by `(Time, row_number)`, and split at `floor(0.6n)` and `floor(0.8n)` moved forward past any cut timestamp group so equal timestamps never span partitions (`max(train.Time) < min(validation.Time)` and `max(validation.Time) < min(test.Time)`):

| Partition | Rows | Share | Time range (s) | Legitimate (`0`) | Fraud (`1`) | Fraud prevalence |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `train` | `170,888` | `60.00%` | `0` – `120,396` | `170,528` | `360` | `0.2107%` |
| `validation` | `56,957` | `19.998%` | `120,397` – `145,247` | `56,900` | `57` | `0.1001%` |
| `test` | `56,962` | `20.00%` | `145,248` – `172,792` | `56,887` | `75` | `0.1317%` |

### Model, Baseline, and Cutoff Rule

- **Isolation Forest:** `IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=42)` fitted on all `170,888` training rows using the 29 named features (`V1`–`V28`, `Amount`), with `Time`, `Class`, and `row_number` excluded.
- **Amount-only baseline:** Ranks transactions by raw `Amount` descending, using the same validation budgets and tie rule.
- **No-skill reference:** Horizontal prevalence line (`0.001001` on validation, `0.001317` on test); an empty review queue catches `0` frauds.
- **Cutoff rule:** For each method and budget (`0.1%`, `0.5%` main, `1.0%`), compute `K = floor(rate × N_val)` (`56`, `284`, `569`), group validation scores by exact ties from highest to lowest, take the largest cumulative count `<= K`, and save the last included score at full float64 precision. Apply `score >= cutoff` unchanged to test and new batches.

## Measured Results (`baseline-20261119T053308Z`)

### Validation Summary (Cutoffs Derived Here, `N = 56,957`, `57` Frauds, Prevalence `0.001001`)

| Method | Budget | Frozen cutoff | Alerts | Alert rate | Alerts / 100k | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest (`AP = 0.0216`) | `0.1%` | `0.69086972632739` | `56` | `0.098%` | `98.3` | `0` | `56` | `57` | `56,844` | `0.0%` | `0.0%` |
| Isolation Forest | **`0.5%`** | **`0.6253202111783525`** | **`284`** | **`0.499%`** | **`498.6`** | **`4`** | **`280`** | **`53`** | **`56,620`** | **`1.4%`** | **`7.0%`** |
| Isolation Forest | `1.0%` | `0.5989357625479854` | `569` | `0.999%` | `999.0` | `12` | `557` | `45` | `56,343` | `2.1%` | `21.1%` |
| Amount only (`AP = 0.0020`) | `0.1%` | `3492.39` | `56` | `0.098%` | `98.3` | `0` | `56` | `57` | `56,844` | `0.0%` | `0.0%` |
| Amount only | **`0.5%`** | **`1639.87`** | **`284`** | **`0.499%`** | **`498.6`** | **`1`** | **`283`** | **`56`** | **`56,617`** | **`0.4%`** | **`1.8%`** |
| Amount only | `1.0%` | `1128.35` | `568` | `0.997%` | `997.2` | `2` | `566` | `55` | `56,334` | `0.4%` | `3.5%` |

### Held-Out Test Summary (Frozen Validation Cutoffs, `N = 56,962`, `75` Frauds, Prevalence `0.001317`)

| Method | Budget | Frozen cutoff | Alerts | Alert rate | Alerts / 100k | TP | FP | FN | TN | Precision | Recall | Fixed `K` | `TP@K` | `P@K` | `R@K` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest (`AP = 0.0332`) | `0.1%` | `0.6909` | `32` | `0.056%` | `56.2` | `0` | `32` | `75` | `56,855` | `0.0%` | `0.0%` | `56` | `0` | `0.0%` | `0.0%` |
| Isolation Forest | **`0.5%`** | **`0.6253`** | **`228`** | **`0.400%`** | **`400.3`** | **`2`** | **`226`** | **`73`** | **`56,661`** | **`0.9%`** | **`2.7%`** | **`284`** | **`4`** | **`1.4%`** | **`5.3%`** |
| Isolation Forest | `1.0%` | `0.5989` | `418` | `0.734%` | `733.8` | `16` | `402` | `59` | `56,485` | `3.8%` | `21.3%` | `569` | `29` | `5.1%` | `38.7%` |
| Amount only (`AP = 0.0014`) | `0.1%` | `3492.39` | `25` | `0.044%` | `43.9` | `0` | `25` | `75` | `56,862` | `0.0%` | `0.0%` | `56` | `0` | `0.0%` | `0.0%` |
| Amount only | **`0.5%`** | **`1639.87`** | **`182`** | **`0.320%`** | **`319.5`** | **`0`** | **`182`** | **`75`** | **`56,705`** | **`0.0%`** | **`0.0%`** | **`284`** | **`1`** | **`0.4%`** | **`1.3%`** |
| Amount only | `1.0%` | `1128.35` | `416` | `0.730%` | `730.3` | `1` | `415` | `74` | `56,472` | `0.2%` | `1.3%` | `569` | `3` | `0.5%` | `4.0%` |

### Answers to the Project Questions

1. **Does Isolation Forest rank known frauds higher than the amount-only baseline on the held-out test set?**
   Yes. On the latest 20% holdout (`56,962` rows, `75` frauds), Isolation Forest achieved **AP = `0.0332`** versus **`0.0014`** for Amount-only (compared with the no-skill test prevalence of `0.0013`). Amount-only performed essentially at chance, while Isolation Forest outperformed it at both `0.5%` and `1.0%` budgets and in fixed-size top-K rankings, though absolute precision remained low.
2. **What does the frozen 0.5% validation cutoff (500 alerts per 100k on validation) produce on later test data?**
   The frozen Isolation Forest cutoff (`0.6253202111783525`) generated **`228` alerts** (`0.400%`, or **`400.3` per 100,000** — below the 500/100k validation target due to lower upper-tail test scores), catching **`2` of `75` frauds** (**`2.7%` recall**, **`0.9%` precision**, `226` false positives, `73` missed frauds). At the same validation budget, Amount-only (`cutoff = 1,639.87`) generated `182` alerts (`319.5` per 100k) and caught **`0` frauds**.
3. **How do review workload and fraud coverage change at smaller (`0.1%`) and larger (`1.0%`) budgets?**
   - At `0.1%`, neither method caught any fraud (`32` alerts / `0` TP for Isolation Forest; `25` alerts / `0` TP for Amount-only); the most extreme anomaly scores are dominated by legitimate outlier purchases.
   - Expanding from `0.5%` to `1.0%` raised Isolation Forest test workload from `228` to **`418` alerts** (`733.8` per 100k) and jumped recall from `2.7%` (`2` TP) to **`21.3%` (`16` TP)** while also improving precision from `0.9%` to **`3.8%`**. Using a fixed top-1% queue (`K = 569`, offsetting threshold drift) increased Isolation Forest coverage further to **`29` frauds (`38.7%` recall, `5.1%` precision)** versus `3` frauds (`4.0%` recall) for Amount-only.
4. **What can we learn from the false positives and missed frauds?**
   See the post-test error analysis below.

## Error Analysis (Frozen `0.5%` Cutoff on Test)

- **Highest-scoring false positives (`226` rows):** `Amount` spans `0.00` to `25,691.16` (median `35.00`) across `Time 145,283–172,786 s`. However, all 10 highest-scoring false positives are very large legitimate transactions (`Amount` between `1,201.93` and `25,691.16`), led by `row_number 274772` (`score = 0.7889`, `Amount = 25,691.16`, `Time = 166,198 s`), `row_number 231455` (`score = 0.7803`, `Amount = 3,552.96`), `row_number 238413` (`score = 0.7360`, `Amount = 6,900.47`), and `row_number 245475` (`score = 0.7314`, `Amount = 8,787.00`).
- **Lowest-scoring false negatives (`73` missed frauds):** `Amount` spans `0.00` to `1,504.93` with a **median of `4.69`** across `Time 146,022–170,348 s`. The lowest-scoring missed frauds (`row_number 276072`, `score = 0.4025`, `Amount = 19.95`; `row_number 254345`, `score = 0.4046`, `Amount = 187.11`; `row_number 281675`, `score = 0.4113`, `Amount = 42.53`) sit right near the overall test median score (`0.3976`), meaning they look globally typical in the 29-feature space.
- **Method disagreement:**
  - At `0.5%`, Isolation Forest caught `2` frauds (`row_number 243394`, `Amount = 0.01`, `score = 0.6393`; and `row_number 249608`, `Amount = 35.00`, `score = 0.6290`), whereas Amount-only caught `0`. One of those two falls in the moderate-amount interquartile range (`[4.49, 64.80]`, which contains `20` of the `75` test frauds). Both methods overlapped on `53` legitimate false positives.
  - At `1.0%`, Isolation Forest caught `16` frauds versus `1` for Amount-only (`15` caught exclusively by Isolation Forest, `1` by both).
- **Score distribution and alert-rate shift across partitions:**

| Partition | Min | P25 | Median | P75 | P90 | P99 | P99.5 | P99.9 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `0.3388` | `0.3683` | `0.3886` | `0.4158` | `0.4526` | `0.5872` | `0.6215` | `0.6910` | `0.7785` |
| `validation` | `0.3487` | `0.3823` | `0.4026` | `0.4277` | `0.4600` | `0.5985` | `0.6253` | `0.6864` | `0.7898` |
| `test` | `0.3469` | `0.3782` | `0.3976` | `0.4234` | `0.4547` | `0.5800` | `0.6161` | `0.6715` | `0.7889` |

Validation scores had a heavier upper tail than both train and test (`P99.5 = 0.6253` on validation vs `0.6161` on test; `P99 = 0.5985` vs `0.5800`). Applying the frozen validation cutoffs to test therefore reduced observed alert rates across all budgets for both Isolation Forest (`0.098% → 0.056%`, `0.499% → 0.400%`, `0.999% → 0.734%`) and Amount-only (`0.098% → 0.044%`, `0.499% → 0.320%`, `0.997% → 0.730%`). The anonymized dataset does not contain metadata to establish the underlying cause of this shift.

### Conclusion

On this chronological split of `creditcard.csv`, Isolation Forest ranked recorded fraud substantially better than the amount-only baseline (`AP = 0.0332` vs `0.0014`), catching `2` frauds (`226` legitimate rows queued) at the frozen `0.5%` cutoff and `16` frauds (`402` legitimate rows queued) at `1.0%`, compared with `0` (`182` FP) and `1` (`415` FP) for Amount-only. However, the first release exhibits two clear weaknesses worth targeting next: (1) extreme-amount legitimate transactions crowd out frauds in the top `0.1%–0.5%` score tail, and (2) fixed score cutoffs drift when partition score tails shift, leaving `13` additional frauds just below the frozen `1.0%` threshold (within test top-1%). Future experiments (`plan.md` §8) should compare density-based Local Outlier Factor (`novelty=True` with training-fit scaling) and a legitimate-trained autoencoder on the same fixed partitions before evaluating on a fresh holdout.

## Verified Run Record and Reproducibility

Executed in a clean `.venv` created via `python3 -m venv .venv && pip install -r requirements.txt`:

- **Environment:** Python `3.10.7`, `macOS-15.7.9-x86_64-i386-64bit`, `numpy==2.2.6`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.9`, `joblib==1.6.0`.
- **Unit & integration tests:** `python -m unittest discover -s tests` → **23 tests OK** (`77.3 s` in the fresh environment; `16.0 s` warm).
- **Training:** `python fraud_detection.py train --input data/creditcard.csv --run-dir outputs/baseline` → `22.6 s` (`fit_seconds = 0.38 s`), artifacts in `outputs/baseline/`.
- **Evaluation:** `python fraud_detection.py evaluate --input data/creditcard.csv --run-dir outputs/baseline` → `16.4 s`, artifacts in `outputs/baseline/metrics.json` and `outputs/baseline/precision_recall.png`.
- **Demo batch scoring:** `python fraud_detection.py score --input data/transactions_to_score.csv --model outputs/baseline/model.joblib --output outputs/review_queue.csv` → `2.5 s`, scored `1,000` rows, flagged `8` rows at cutoff `0.6253202111783525`; output joins 1:1 back to input by `row_number` and reproduces exact flags on reload.
- **Reproducibility rerun:** Re-executing `train`, `evaluate`, and `score` into a separate run directory with the same input file, settings, and package versions produced bit-for-bit identical `audit`, `split` (boundary indices and partition `row_number` SHA-256 hashes), `validation` cutoffs and metrics, `test` metrics, score quantiles, alert rates, `error_analysis`, and `review_queue.csv`.

## Limitations and Scope

- **Two-day historical snapshot:** All `284,807` rows span ~48 hours from one European cardholder sample in September 2013; performance on this split provides limited evidence about future fraud patterns or other portfolios.
- **Anonymized features without entity IDs:** `V1`–`V28` are PCA components and the dataset omits cardholder, card, and merchant identifiers. The detector can only measure aggregate cross-transaction unusualness in this feature space—it cannot model per-customer history, velocity, or real-world feature explanations.
- **Feature representation coupling:** Batch scoring requires new transactions to supply the exact same PCA-transformed features (`V1`–`V28`) and `Amount`.
- **Prototype scope:** The review queue is a learning prototype for human triage, not a production decision engine.

## Post-Baseline Improvement Experiments (`plan.md` §8)

All five post-baseline improvement tracks in `plan.md` §8 were implemented in `experiments.py` (with unit/integration tests in `tests/test_experiments.py`) and run sequentially on the unchanged `data/creditcard.csv` and `outputs/baseline/` (`baseline-20261119T053308Z`, `fraud_detection.py` unmodified).

### Shared Protocol & Reproducibility

- **Same partitions and budgets:** Uses the exact first-release 60/20/20 chronological split (`170,888` / `56,957` / `56,962`) and validation budgets (`0.1%`, `0.5%`, `1.0%`) with the same whole-tie-group cutoff rule.
- **Expanding-window chronological CV inside `train + validation`:** `train` is cut into three timestamp-group-safe blocks (`B1`: `56,962` rows / `157` frauds / `0–47,694 s`; `B2`: `56,964` rows / `84` frauds / `47,695–73,262 s`; `B3`: `56,962` rows / `119` frauds / `73,263–120,396 s`) and `validation` is `B4` (`56,957` rows / `57` frauds / `120,397–145,247 s`). Folds evaluate `B1 → B2`, `B1–B2 → B3`, and `B1–B3 (= train) → validation`.
- **Leakage-free preprocessing & selection:** Every scaler, `log1p(Amount)`, ECDF, and estimator fits on fit rows only. Candidates are selected by **mean CV AP** before computing **exploratory test metrics** (exploratory because the test partition was already inspected in the first release).

```bash
python -m unittest discover -s tests   # 35 tests OK (23 first-release + 12 post-baseline)
python experiments.py track1 --input data/creditcard.csv --run-dir outputs/track1-supervised
python experiments.py track2 --input data/creditcard.csv --run-dir outputs/track2-iforest-tuning
python experiments.py track3 --input data/creditcard.csv --run-dir outputs/track3-lof --budget-seconds 180
python experiments.py track4 --input data/creditcard.csv --run-dir outputs/track4-autoencoder
python experiments.py track5 --input data/creditcard.csv --run-dir outputs/track5-queue-stability \
  --models if_baseline,logistic_regression,random_forest
```

### Cross-Track Headline Comparison (vs. `baseline-20261119T053308Z`)

| Track | Selected / key model | Label use | CV mean AP | Val AP (`fold3`) | Exploratory test AP | Test `0.5%` cut: alerts / TP / P / R | Test `1.0%` cut: alerts / TP / P / R | Mean fit time | Improved vs. baseline? | Run directory |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| Baseline | `if_baseline` | None (unsupervised) | `0.1528 ± 0.1375` | `0.0216` | `0.0332` | `228` / `2` / `0.9%` / `2.7%` | `418` / `16` / `3.8%` / `21.3%` | `0.3–1.0 s` | Reference | `outputs/baseline/` |
| Track 1 | `random_forest+if_feature` | Supervised + IF feature | **`0.7926 ± 0.0204`** | `0.7713` | **`0.7960`** (**24.0×**) | `251` / **`63`** / **`25.1%`** / **`84.0%`** | `458` / **`65`** / **`14.2%`** / **`86.7%`** | `126.3 s` | **Yes — massive gain** | `outputs/track1-supervised/` |
| Track 1 (fast) | `logistic_regression` | Supervised | `0.6091 ± 0.1029` | `0.6507` | `0.6965` (21.0×) | `290` / `62` / `21.4%` / `82.7%` | `554` / `65` / `11.7%` / `86.7%` | **`1.2 s`** | **Yes — massive gain** | `outputs/track1-supervised/` |
| Track 2 | `if_combined` | None (unsupervised) | `0.2149 ± 0.1991` | `0.0329` | `0.0773` (2.33×) | `179` / `23` / `12.8%` / `30.7%` | `380` / `39` / `10.3%` / `52.0%` | `2.2 s` | **Yes — strong unsupervised gain** | `outputs/track2-iforest-tuning/` |
| Track 3 | `lof_log1p_standard_k20` | None (unsupervised) | `0.0037 ± 0.0028` | `0.0016` | `0.0017` (0.05×) | `337` / `0` / `0.0%` / `0.0%` | `725` / `0` / `0.0%` / `0.0%` | `56.9 s` | **No — collapsed to chance** | `outputs/track3-lof/` |
| Track 4 | `autoencoder_standard_seed0` | Semi-supervised (`Class = 0` fit only) | `0.2797 ± 0.2554` | `0.0599` | `0.1438` (4.33×) | `154` / `33` / `21.4%` / `44.0%` | `335` / `52` / `15.5%` / `69.3%` | `154.9 s` | **Yes — best non-fraud-label detector** | `outputs/track4-autoencoder/` |
| Track 5 | `batch_top_k_proportional` / `rolling_quantile` on `if_baseline` | Unlabeled score calibration | — | — | Same scores (`0.0332`) | Top-K: `282` / `8` / `2.8%` / **`10.7%`**; Rolling: `290` / `6` / `2.1%` / **`8.0%`** | Top-K: `565` / `31` / `5.5%` / **`41.3%`**; Rolling: `546` / `27` / `4.9%` / **`36.0%`** | — | **Yes — stabilizes volume & lifts recall** | `outputs/track5-queue-stability/` |

### Track 1 · Supervised Classifiers and Hybrid (`outputs/track1-supervised`, `1374.9 s`)

- **Run ID:** `track1-supervised-20261119T074827Z` (peak RSS `1,042.7 MB`).
- **Selection:** Stage 1 picked `random_forest` (`CV AP = 0.7876`); Stage 2 built two hybrids (`+if_feature` and `+rank_blend`), and overall CV AP selected **`random_forest+if_feature` (`CV AP = 0.7926`)**.

| Candidate | CV AP | Val AP | Test AP | Test `0.1%` (alerts / TP / P / R) | Test `0.5%` (alerts / TP / P / R) | Test `1.0%` (alerts / TP / P / R) | Test Brier | Mean fit / score |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| `if_baseline` | `0.1528 ± 0.1375` | `0.0216` | `0.0332` | `32` / `0` / `0.0%` / `0.0%` | `228` / `2` / `0.9%` / `2.7%` | `418` / `16` / `3.8%` / `21.3%` | — | `0.8 s` / `23.6 µs/row` |
| `amount_only` | `0.0024 ± 0.0003` | `0.0020` | `0.0014` | `25` / `0` / `0.0%` / `0.0%` | `182` / `0` / `0.0%` / `0.0%` | `416` / `1` / `0.2%` / `1.3%` | — | `0.0 s` / `0.0 µs/row` |
| `logistic_regression` | `0.6091 ± 0.1029` | `0.6507` | `0.6965` | `66` / `48` / `72.7%` / `64.0%` | `290` / `62` / `21.4%` / `82.7%` | `554` / `65` / `11.7%` / `86.7%` | `0.000656` | `1.2 s` / `1.5 µs/row` |
| `hist_gradient_boosting` | `0.0510 ± 0.0222` | `0.0218` | `0.0483` | `0` / `0` / `N/A` / `0.0%` | `233` / `26` / `11.2%` / `34.7%` | `233` / `26` / `11.2%` / `34.7%` | `0.003775` | `9.3 s` / `23.1 µs/row` |
| `random_forest` | `0.7876 ± 0.0138` | `0.7725` | `0.7855` | `74` / `57` / `77.0%` / `76.0%` | `225` / `61` / `27.1%` / `81.3%` | `473` / `62` / `13.1%` / `82.7%` | `0.000453` | `190.1 s` / `11.3 µs/row` |
| **`random_forest+if_feature`** | **`0.7926 ± 0.0204`** | `0.7713` | **`0.7960`** | **`72` / `57` / `79.2%` / `76.0%`** | **`251` / `63` / `25.1%` / `84.0%`** | **`458` / `65` / `14.2%` / `86.7%`** | **`0.000444`** | `126.3 s` / `11.7 µs/row` |
| `random_forest+rank_blend` | `0.2723 ± 0.2125` | `0.0499` | `0.0787` | `30` / `0` / `0.0%` / `0.0%` | `208` / `23` / `11.1%` / `30.7%` | `397` / `40` / `10.1%` / `53.3%` | — | `111.2 s` / `11.5 µs/row` |

- **Findings:** Supervised signal resolved the top-tail crowding problem completely: at `0.1%`, `random_forest+if_feature` caught `57 / 75` frauds (`79.2%` precision, only `15` false alerts) while every unsupervised/semi-supervised model caught `0 / 75` at `0.1%`. Calibration was close to observed rates (`top 0.5%` mean probability `0.208` vs `0.225` observed precision for `random_forest+if_feature`; `0.203` vs `0.218` for `logistic_regression`). Feeding Isolation Forest's anomaly score as a 30th tree feature modestly improved CV AP (`+0.0050`), test AP (`+0.0105`), and test recall (`+2` frauds at `0.5%`, `+3` at `1.0%`), whereas naive 50/50 ECDF rank blending diluted the supervised ranking (`test AP = 0.0787`). Default `HistGradientBoostingClassifier(max_iter=200, early_stopping=False)` overfit the extreme `0.17%` imbalance and became severely overconfident (top-`0.5%` mean probability `0.677` vs `0.092` precision; ties at `1.0` disabled the `0.1%` cutoff).

### Track 2 · Unsupervised Isolation Forest Tuning (`outputs/track2-iforest-tuning`, `57.6 s`)

- **Run ID:** `track2-iforest-tuning-20261119T074958Z`.
- **One-change-at-a-time selection:** Kept the best CV-improving change per dimension (`if_drop_amount`, `if_max_samples_2048`, `if_n_estimators_300`, `if_max_features_0.75`) and stacked them into **`if_combined`**, which won on CV AP (`0.2149`) and exploratory test AP (`0.0773`).

| Candidate | CV AP | Val AP | Test AP | Test `0.5%` cut (alerts / TP / P / R) | Test `1.0%` cut (alerts / TP / P / R) | Test `R@top1%` | Top-`0.5%` FP `Amount >= $1k` | Mean fit |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `if_baseline` | `0.1528 ± 0.1375` | `0.0216` | `0.0332` | `228` / `2` / `0.9%` / `2.7%` | `418` / `16` / `3.8%` / `21.3%` | `38.7%` | `27.5%` | `0.29 s` |
| `if_log1p` / `_robust` / `_standard` | `0.1594 ± 0.1371` | `0.0228` | `0.0338` | `232` / `4` / `1.7%` / `5.3%` | `422` / `16` / `3.8%` / `21.3%` | `38.7%` | `19.4%` | `0.34–0.57 s` |
| `if_drop_amount` | `0.1617 ± 0.1419` | `0.0282` | `0.0491` | `206` / `11` / `5.3%` / `14.7%` | `418` / `30` / `7.2%` / `40.0%` | `48.0%` | `19.3%` | `0.31 s` |
| `if_max_samples_2048` | `0.1836 ± 0.1822` | `0.0271` | `0.0534` | `184` / `14` / `7.6%` / `18.7%` | `391` / `32` / `8.2%` / `42.7%` | `54.7%` | `28.5%` | `0.51 s` |
| `if_max_samples_4096` | `0.1716 ± 0.1651` | `0.0286` | `0.0631` | `178` / `16` / `9.0%` / `21.3%` | `386` / `37` / `9.6%` / `49.3%` | `57.3%` | `29.1%` | `0.55 s` |
| `if_n_estimators_300` | `0.1742 ± 0.1493` | `0.0234` | `0.0400` | `194` / `9` / `4.6%` / `12.0%` | `387` / `22` / `5.7%` / `29.3%` | `46.7%` | `29.6%` | `0.85 s` |
| `if_max_features_0.75` | `0.1748 ± 0.1467` | `0.0233` | `0.0433` | `180` / `10` / `5.6%` / `13.3%` | `351` / `22` / `6.3%` / `29.3%` | `48.0%` | `38.3%` | `0.54 s` |
| `if_max_features_0.5` | `0.1497 ± 0.1366` | `0.0161` | `0.0294` | `165` / `4` / `2.4%` / `5.3%` | `354` / `12` / `3.4%` / `16.0%` | `34.7%` | `33.2%` | `0.44 s` |
| **`if_combined`** | **`0.2149 ± 0.1991`** | **`0.0329`** | **`0.0773`** | **`179` / `23` / `12.8%` / `30.7%`** | **`380` / `39` / `10.3%` / `52.0%`** | **`58.7%`** | `23.9%` | `2.16 s` |

- **Findings:** Linear scaling after `log1p(Amount)` (`RobustScaler`, `StandardScaler`) gave identical rankings to `log1p` alone (axis-aligned trees are invariant to affine feature scaling) and cut `>= $1,000` false positives from `27.5%` to `19.4%`. Dropping `Amount` altogether beat `log1p(Amount)` on every fold, and increasing `max_samples` from `256` to `2048` delivered the largest single unsupervised jump. Stacking the four winning edits (`if_combined`) remained 100% unsupervised (`2.2 s` fit), more than doubled exploratory test AP (`0.0332 → 0.0773`), and lifted frozen-`0.5%` recall `11.5×` (`2.7% → 30.7%`, `23` vs `2` frauds) with fewer alerts (`179` vs `228`).

### Track 3 · Unsupervised Local Outlier Factor (`outputs/track3-lof`, `579.2 s`)

- **Run ID:** `track3-lof-20261119T080014Z` (peak RSS `1,051.2 MB`).
- **Timed pilot sizing:** Pilot timings (`n = 10,000`: fit `0.39 s`, score `5,000` rows `0.15 s`; `n = 20,000`: fit `1.18 s`, score `0.30 s`) extrapolated `115.4 s` per full fold (`<= 180 s` budget), so LOF ran on **all `170,888` training rows** (`max_fit_rows = None`, seed `42`, `novelty = True`, `n_neighbors = 20`).

| Candidate | CV AP | Val AP | Test AP | Test `0.5%` cut (alerts / TP / R) | Test `1.0%` cut (alerts / TP / R) | `fold3` fit / score | `fold3` peak traced mem (fit / score) | Test `P99.9` / `Max` score |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `if_baseline` | `0.1528` | `0.0216` | `0.0332` | `228` / `2` / `2.7%` | `418` / `16` / `21.3%` | `0.4 s` / `0.3 s` | — | `0.6715` / `0.7889` |
| `lof_standard_k20` | `0.0024` | `0.0015` | `0.0017` | `294` / `1` / `1.3%` | `659` / `1` / `1.3%` | `92.5 s` / `28.5 s` | `225.2 MB` / `60.8 MB` | `5.12×10⁹` / `2.01×10¹⁰` |
| `lof_log1p_standard_k20` | `0.0037` | `0.0016` | `0.0017` | `337` / `0` / `0.0%` | `725` / `0` / `0.0%` | `113.9 s` / `51.5 s` | `225.2 MB` / `60.8 MB` | `5.57×10⁹` / `2.51×10¹⁰` |

- **Findings (negative result):** LOF did **not** improve over baseline—it degraded to near-chance (`test AP = 0.0017` vs `0.0013` prevalence) at >100× the runtime. Diagnosis: `creditcard.csv` contains `29` identical 29-feature clusters of size `>= 20` (up to `77` duplicates), causing `k = 20` local reachability density to hit scikit-learn's `1e-10` numerical floor and exploding LOF scores to `~10¹⁰` for legitimate points near duplicate clusters, completely crowding out true frauds in the top tail.

### Track 4 · Semi-Supervised Autoencoder (`outputs/track4-autoencoder`, `1877.0 s`)

- **Run ID:** `track4-autoencoder-20261119T083248Z`.
- **Label use & architecture:** Trained `MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu", solver="adam", batch_size=256, max_iter=100, early_stopping=True, validation_fraction=0.1)` with linear output strictly on **`Class = 0` training rows** (`170,528` rows on `fold3`, excluding `360` labeled frauds; internal 10% early-stopping split inside those legitimate rows, keeping `validation` untouched for cutoff derivation). Scored by per-row MSE reconstruction error in scaled space. Every run reached `max_iter = 100` (`ConvergenceWarning`; internal validation `R² = 0.601–0.610`).

| Candidate | CV AP | Val AP | Test AP | Test `0.5%` cut (alerts / TP / P / R) | Test `1.0%` cut (alerts / TP / P / R) | Test `R@top1%` | Mean fit |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `if_baseline` | `0.1528 ± 0.1375` | `0.0216` | `0.0332` | `228` / `2` / `0.9%` / `2.7%` | `418` / `16` / `3.8%` / `21.3%` | `38.7%` | `1.0 s` |
| **`autoencoder_standard_seed0`** *(CV pick)* | **`0.2797 ± 0.2554`** | `0.0599` | `0.1438` | `154` / `33` / `21.4%` / `44.0%` | `335` / `52` / `15.5%` / `69.3%` | `78.7%` | `154.9 s` |
| `autoencoder_log1p_standard_seed0` | `0.2663 ± 0.2574` | `0.0698` | `0.1634` | `194` / `38` / `19.6%` / `50.7%` | `370` / `56` / `15.1%` / `74.7%` | `81.3%` | `184.7 s` |
| `autoencoder_log1p_standard_seed1` | `0.2556 ± 0.1825` | `0.0574` | `0.1329` | `174` / `32` / `18.4%` / `42.7%` | `347` / `55` / `15.9%` / `73.3%` | `73.3%` | `146.0 s` |
| `autoencoder_log1p_standard_seed2` | `0.2448 ± 0.2174` | `0.0805` | `0.1833` | `171` / `37` / `21.6%` / `49.3%` | `348` / `57` / `16.4%` / `76.0%` | `82.7%` | `108.9 s` |

- **Findings & seed sensitivity:** Every autoencoder seed substantially outperformed both `if_baseline` and tuned `if_combined` without using fraud examples during training. Across seeds `0, 1, 2` (`log1p_standard`), mean CV AP was stable (`0.2556 ± 0.0088`) while exploratory test AP spanned `0.1329–0.1833` (`mean = 0.1599 ± 0.0207`, `4.0×–5.5×` baseline) with `42.7%–50.7%` test recall at the frozen `0.5%` cutoff and `73.3%–76.0%` at `1.0%`. Like Track 2, no autoencoder caught fraud inside the `0.1%` budget (`0 / 75`).

### Track 5 · Queue-Volume Stabilization (`outputs/track5-queue-stability`, `1010.4 s`)

- **Run ID:** `track5-queue-stability-20261119T085034Z`.
- **Setup:** Compared `static_cutoff` (first-release frozen cutoff), `batch_top_k_proportional` (`K = floor(rate × hour_rows)` per hourly batch), `batch_top_k_fixed_capacity` (`K = floor(rate × ref_rows / ref_hours)` per hourly batch), and `rolling_quantile` (`(1 - rate)` quantile of the previous `20,000` unlabeled scores, strictly causal) across two development windows (`dev_window1`, `dev_window2`) and exploratory `test` for `if_baseline`, `logistic_regression`, and `random_forest` at `0.1%`, `0.5%`, and `1.0%`.

| Model & Window (Budget) | `static_cutoff` (alerts / target · R · hrMAE) | `batch_top_k_proportional` | `batch_top_k_fixed_capacity` | `rolling_quantile` (`W = 20k`) |
| --- | --- | --- | --- | --- |
| `if_baseline` test (`0.5%`) | `228/284` (`0.80×`) · `2.7%` · `0.0021` | **`282/284` (`0.99×`) · `10.7%` · `0.0000`** | `280/284` (`0.99×`) · `5.3%` · `0.0008` | `290/284` (`1.02×`) · `8.0%` · `0.0016` |
| `if_baseline` test (`1.0%`) | `418/569` (`0.73×`) · `21.3%` · `0.0038` | **`565/569` (`0.99×`) · `41.3%` · `0.0001`** | `568/569` (`1.00×`) · `37.3%` · `0.0015` | `546/569` (`0.96×`) · `36.0%` · `0.0023` |
| `logistic_regression` test (`0.5%`) | `290/284` (`1.02×`) · `82.7%` · `0.0006` | `282/284` (`0.99×`) · `82.7%` · `0.0000` | `280/284` (`0.99×`) · `82.7%` · `0.0008` | `282/284` (`0.99×`) · `82.7%` · `0.0005` |
| `random_forest` `dev_window1` (`0.5%`) | `940/284` (`3.31×`) · `90.8%` · `0.0115` | `278/284` (`0.98×`) · `54.6%` · `0.0001` | `490/284` (`1.73×`) · `89.1%` · `0.0045` | **`417/284` (`1.47×`) · `89.9%` · `0.0043`** |
| `random_forest` test (`0.5%`) | `225/284` (`0.79×`) · `81.3%` · `0.0010` | **`282/284` (`0.99×`) · `82.7%` · `0.0000`** | `280/284` (`0.99×`) · `82.7%` · `0.0008` | `295/284` (`1.04×`) · `82.7%` · `0.0003` |
| `random_forest` test (`1.0%`) | `473/569` (`0.83×`) · `82.7%` · `0.0022` | **`565/569` (`0.99×`) · `84.0%` · `0.0001`** | `568/569` (`1.00×`) · `84.0%` · `0.0015` | `643/569` (`1.13×`) · `84.0%` · `0.0016` |

- **Findings:** Both `batch_top_k_proportional` and `rolling_quantile` eliminated the baseline's test-set under-alerting and substantially improved `if_baseline` test recall (`0.5%`: `2.7% → 10.7%` / `8.0%`; `1.0%`: `21.3% → 41.3%` / `36.0%`), while also stabilizing `random_forest` volume on test (`0.79× → 0.99× / 1.04×`, `+1` fraud). However, development windows revealed an important operational trade-off: strict per-hour `batch_top_k_proportional` locks hourly alert rate (`hrMAE <= 0.0001`) at the cost of clipping recall during bursty fraud hours (`dev_window1` RF `0.5%`: `54.6%` vs `89.9%` for `rolling_quantile`), whereas `rolling_quantile` absorbs smooth score drift while still allowing surge alerts during genuine fraud spikes. Fixed per-hour capacity (`batch_top_k_fixed_capacity`) drifted whenever hourly traffic volume changed (`1.73×` on `dev_window1`, `0.56×` on `dev_window2`).

### Overall Post-Baseline Conclusion

Four of the five tracks improved substantially over `baseline-20261119T053308Z`, while one (Track 3 LOF) regressed:
- **When historical fraud labels are available:** `random_forest+if_feature` (Track 1) is the clear top performer (`test AP = 0.7960`, `84.0%` recall at `0.5%`, `76.0%` recall at `0.1%` with `79.2%` precision), with `logistic_regression` (`test AP = 0.6965`, `1.2 s` fit, `1.5 µs/row`) as an ultra-fast linear alternative.
- **When only legitimate-class labels are available:** Track 4's `Class = 0` autoencoder (`test AP = 0.1438–0.1833`, `44.0%–50.7%` recall at `0.5%`) delivers `4.3×–5.5×` baseline AP.
- **When no labels are available at all:** Track 2's `if_combined` (`drop_amount`, `max_samples=2048`, `n_estimators=300`, `max_features=0.75`) delivers `2.33×` baseline AP (`0.0773`) and `11.5×` baseline `0.5%` recall (`30.7%`) in `2.2 s`.
- **For queue operations:** Pairing a detector with `rolling_quantile` (streaming, burst-tolerant) or `batch_top_k_proportional` (strict analyst capacity) prevents static-cutoff alert drift.
- **Exploratory caveat:** Because all five tracks were designed after inspecting the first-release test report on this 48-hour dataset, test metrics above are exploratory; production adoption requires re-validation on a new independent future holdout with entity-level features.
