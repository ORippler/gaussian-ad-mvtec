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
    "mode": ["--pca", "--npca"],
    "thresholds": [0.99],
    "archs": ["efficientnet-b0"],
    "args": "--model gaussian --max_nb_epochs 0 --batch_size 16 --extract_blocks 6 --augment",
}


def load_jobs(grid, logfolder_path, gpu):
    # Read Arguments from File

    global job_version  # for some reason nonlocal does not work
    job_version = 0

    def job_category_fold(category, fold, mode, threshold, arch):
        job = grid.copy()
        global job_version
        if mode is not None and threshold is not None:
            job["args"] = (
                " "
                + job["args"]
                + " --category {} --logpath {} --version {} {} --variance_threshold {} --arch {}".format(
                    category,
                    logfolder_path,
                    job_version,
                    mode,
                    threshold,
                    arch,
                )
            )
        else:
            job["args"] = (
                " "
                + job["args"]
                + " --category {} --logpath {} --version {} --arch {}".format(
                    category, logfolder_path, job_version, arch
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
        job_category_fold(category, fold, mode, threshold, arch)
        for category in grid["categories"]
        for fold in grid["folds"]
        for mode in grid["mode"]
        for threshold in grid["thresholds"]
        for arch in grid["archs"]
    ]

    grid.update(
        {
            "mode": [None],
            "thresholds": [None],
            "archs": ["efficientnet-b4", "efficientnet-b0"],
        }
    )

    jobs.extend(
        [
            job_category_fold(category, fold, mode, threshold, arch)
            for category in grid["categories"]
            for fold in grid["folds"]
            for mode in grid["mode"]
            for threshold in grid["thresholds"]
            for arch in grid["archs"]
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

    df["mode"] = " "
    str2bool = {"True": True, "False": False}
    df.loc[df["npca"].map(str2bool), "mode"] = "npca"
    df.loc[df["pca"].map(str2bool), "mode"] = "pca"

    df = df.groupby(["arch", "mode", "variance_threshold", "category"]).mean()
    df = df.reset_index(level=(2, 3))

    table = np.empty(shape=(6, 8))

    for idx, df_index in enumerate(df.index.unique(), step=2):
        for level in range(5):
            table[level, idx] = df.loc[df_index][
                f"latent_maha/level_6/sigma_{level + 1}/fpr/test_mean"
            ].mean()
            table[level, idx + 1] = df.loc[df_index][
                f"latent_maha/level_6/sigma_{level + 1}/tpr/test_mean"
            ].mean()
        table[5, idx] = df.loc[df_index][
            "latent_maha/level_6/auc/test/ROC/anomalies_mean"
        ].mean()
        table[5, idx + 1] = np.NaN

    column_index = [
        [str(index) for index in df.index.unique() for _ in (0, 1)],
        ["FPR", "TPR"] * 8,
    ]
    tuples = list(zip(*column_index))
    column_index = pd.MultiIndex.from_tuples(tuples)

    index = [f"sigma_{level + 1}" for level in range(5)] + ["AUROC"]

    table = pd.DataFrame(table, index=index, columns=column_index)
    table = table * 100
    pd.options.display.float_format = "{:,.1f}".format

    table = table.reindex(
        columns=[
            "('efficientnet-b0', 'pca')",
            "('efficientnet-b0', ' ')",
            "('efficientnet-b0', 'npca')",
            "('efficientnet-b4', ' ')",
        ],
        level=0,
    )

    table.to_csv("table4.csv")
    print(table)
