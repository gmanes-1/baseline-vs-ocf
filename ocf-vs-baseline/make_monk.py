import os, pandas as pd
root = os.environ["KRML_DATASETS"]
cols = ["y"] + [f"X_{i}" for i in range(1, 7)] + ["id"]
for n in (1, 2, 3):
    name = f"900{n}-MONK-{n}_432r_6f"
    d = f"{root}/{name}"
    os.makedirs(d, exist_ok=True)
    df = pd.read_csv(f"{root}/monks-{n}.test", sep=r"\s+", header=None, names=cols).drop(columns="id")
    df.insert(0, "ID", range(len(df)))
    df.to_csv(f"{d}/{name}.csv", index=False)
    pd.DataFrame({"FEATURE_NAME": [f"X_{i}" for i in range(1, 7)],
                  "FEATURE_TYPE": ["Integer"] * 6}).to_csv(f"{d}/{name}-type.csv", index=False)
    print(name, len(df), "rows, positives:", int(df.y.sum()))
