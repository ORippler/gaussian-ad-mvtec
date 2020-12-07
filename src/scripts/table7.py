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
    "archs": ["resnet18", "resnet34", "resnet50"],
    "args": "--model gaussian --max_nb_epochs 0 --batch_size 16 --extract_blocks 0 1 2 3 4",
}


def load_jobs(grid, logfolder_path, gpu):
    # Read Arguments from File

    global job_version  # for some reason nonlocal does not work
    job_version = 0

    def job_category_fold(category, fold, arch):
        job = grid.copy()
        global job_version
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
        job_category_fold(category, fold, arch)
        for category in grid["categories"]
        for fold in grid["folds"]
        for arch in grid["archs"]
    ]

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
        # print("python -m src.common.trainer" + job["args"])
        os.system("python -m src.common.trainer" + job["args"])

    # aggregate results
    from .results import aggregate_results
    import pandas as pd
    import numpy as np

    df = aggregate_results(args.logpath)

    def enumerate(xs, start=0, step=1):
        for x in xs:
            yield (start, x)
            start += step

    df = df.groupby(["arch", "category"]).mean()
    df = df.reset_index(level=1)

    table = np.empty(shape=(6, 6))

    for idx, df_index in enumerate(df.index.unique(), step=2):
        for level in range(5):
            table[level, idx] = df.loc[df_index][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].mean()
            table[level, idx + 1] = df.loc[df_index][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].sem()
        table[5, idx] = df.loc[df_index][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].mean()
        table[5, idx + 1] = df.loc[df_index][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].sem()

    index = [f"level_{level + 1}" for level in range(5)] + ["Sum"]

    column_index = [
        [str(index) for index in df.index.unique() for _ in (0, 1)],
        ["Mean", "SEM"] * 4,
    ]
    tuples = list(zip(*column_index))
    column_index = pd.MultiIndex.from_tuples(tuples)

    table = pd.DataFrame(table, index=index, columns=column_index)
    table = table * 100
    pd.options.display.float_format = "{:,.1f}".format

    table.to_csv("table7.csv")
    print(table)
