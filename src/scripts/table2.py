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
    "archs": [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ],
    "args": "--model gaussian --max_nb_epochs 0 --batch_size 16 --extract_blocks 0 1 2 3 4 5 6 7 8",
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
    df = df.groupby(["arch", "category"]).mean()

    def enumerate(xs, start=0, step=1):
        for x in xs:
            yield (start, x)
            start += step

    table = np.empty(shape=(10, 16))

    for idx, arch in enumerate(
        [f"efficientnet-b{level}" for level in range(8)], step=2
    ):
        print(idx)
        for level in range(9):
            table[level, idx] = df.loc[arch][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].mean()
            table[level, idx + 1] = df.loc[arch][
                f"latent_maha/level_{level}/auc/test/ROC/anomalies_mean"
            ].sem()
        table[9, idx] = df.loc[arch][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].mean()
        table[9, idx + 1] = df.loc[arch][
            "maha/full_auc/test/ROC/anomalies_mean"
        ].sem()

    column_index = [
        [f"efficientnet-b{level}" for level in range(8) for _ in (0, 1)],
        ["mean", "sem"] * 8,
    ]
    tuples = list(zip(*column_index))
    column_index = pd.MultiIndex.from_tuples(tuples)

    index = [f"level_{level}" for level in range(9)] + ["sum"]

    table = pd.DataFrame(table, index=index, columns=column_index)
    table = table * 100
    pd.options.display.float_format = "{:,.1f}".format

    table.to_csv("table2.csv")
    print(table)
