"""Regenerate the small synthetic fixtures in tests/fixtures/.

The labeled fixture has repeated timestamps and both classes in every
chronological partition. The unlabeled copy is the same file without Class.
Values are synthetic; they are not drawn from the public dataset.

    python tests/make_fixtures.py
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
FEATURES = [f"V{i}" for i in range(1, 29)]


def make_labeled(rows: int = 2000, fraud_rate: float = 0.03, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 700, size=rows))  # about 3 rows per timestamp
    data = {"Time": times}
    values = rng.normal(0.0, 1.0, size=(rows, len(FEATURES)))
    labels = (rng.random(rows) < fraud_rate).astype(int)
    values[labels == 1, :5] += rng.normal(4.0, 1.0, size=(int(labels.sum()), 5))
    for i, name in enumerate(FEATURES):
        data[name] = np.round(values[:, i], 6)
    amount = np.round(rng.lognormal(3.0, 1.2, size=rows), 2)
    amount[labels == 1] = np.round(amount[labels == 1] * rng.uniform(0.2, 3.0, size=int(labels.sum())), 2)
    data["Amount"] = amount
    data["Class"] = labels
    frame = pd.DataFrame(data)
    # Shuffle file order so the split must sort by Time itself.
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    labeled = make_labeled()
    labeled.to_csv(os.path.join(FIXTURES, "labeled_small.csv"), index=False)
    labeled.drop(columns=["Class"]).to_csv(os.path.join(FIXTURES, "unlabeled_small.csv"), index=False)
    print(f"wrote {len(labeled)} rows, {int(labeled['Class'].sum())} frauds, "
          f"{labeled['Time'].nunique()} distinct timestamps to {FIXTURES}")


if __name__ == "__main__":
    main()
