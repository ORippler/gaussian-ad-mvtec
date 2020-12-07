import os
import argparse

job_dict = {
    "categories": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "folds": [0, 1, 2, 3, 4],
    "mode": ["--pca"],
    "thresholds": [0.95, 0.99],
    "args": "--model gaussian --max_nb_epochs 0 --batch_size 16 --extract_blocks 0 1 2 3 4 5 6 7 8 --arch efficientnet-b4",
}


def load_jobs(grid, logfolder_path, gpu):
    # Read Arguments from File

    global job_version  # for some reason nonlocal does not work
    job_version = 0

    def job_category_fold(category, fold, mode, threshold):
        job = grid.copy()
        global job_version
        job["args"] = (
            " "
            + job["args"]
            + " --category {} --logpath {} --version {} {} --variance_threshold {}".format(
                category, logfolder_path, job_version, mode, threshold
            )
        )
        if gpu:
            job["args"] = job["args"] + " --gpus 0"
        job_version += 1
        if fold is not None:
            job["args"] += " --fold {}".format(fold)
        return job

    # cross-product the grid
    jobs = [
        job_category_fold(category, fold, mode, threshold)
        for category in grid["categories"]
        for fold in grid["folds"]
        for mode in grid["mode"]
        for threshold in grid["thresholds"]
    ]

    grid.update({"mode": ["--npca"], "thresholds": [0.99, 0.999, 0.9999]})

    jobs.extend(
        [
            job_category_fold(category, fold, mode, threshold)
            for category in grid["categories"]
            for fold in grid["folds"]
            for mode in grid["mode"]
            for threshold in grid["thresholds"]
        ]
    )

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logpath",
        type=str,
        default=os.getcwd(),
        help="The path where logs should go",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether or not to use GPU acceleration",
    )

    args = parser.parse_args()

    jobs = load_jobs(job_dict, args.logpath, args.gpu)

    # Perform individual runs
    for job in jobs:
        os.system("python -m src.common.trainer" + job["args"])

    # aggregate results
    from .results import aggregate_results
    import pandas as pd
    import numpy as np

    def enumerate(xs, start=0, step=1):
        for x in xs:
            yield (start, x)
            start += step

    df = aggregate_results(args.logpath)

    df["mode"] = ""
    str2bool = {"True": True, "False": False}
    df.loc[df["npca"].map(str2bool), "mode"] = "npca"
    df.loc[df["pca"].map(str2bool), "mode"] = "pca"

    df = df.groupby(["mode", "variance_threshold", "category"]).mean()
    df = df.reset_index(level=2)

    table = np.empty(shape=(10, 10))

    for idx, df_index in enumerate(df.index.unique(), step=2):
        for level in range(9):
            table[level, idx] = df.loc[df_index][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].mean()
            table[level, idx + 1] = df.loc[df_index][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].sem()
        table[9, idx] = df.loc[df_index][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].mean()
        table[9, idx + 1] = df.loc[df_index][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].sem()

    column_index = [
        [str(index) for index in df.index.unique() for _ in (0, 1)],
        ["mean", "sem"] * 8,
    ]
    tuples = list(zip(*column_index))
    column_index = pd.MultiIndex.from_tuples(tuples)

    index = [f"level_{level}" for level in range(9)] + ["sum"]

    table = pd.DataFrame(table, index=index, columns=column_index)
    table = table * 100
    pd.options.display.float_format = "{:,.1f}".format
    table = table.reindex(
        columns=[
            "('pca', '0.99')",
            "('pca', '0.95')",
            "('npca', '0.99')",
            "('npca', '0.999')",
            "('npca', '0.9999')",
        ],
        level=0,
    )

    table.to_csv("table3.csv")
    print(table)
