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
    "args": "--model gaussian --max_nb_epochs 0 --arch efficientnet-b4 --batch_size 16 --extract_blocks 0 1 2 3 4 5 6 7 8 --l2 --sed",
}


def load_jobs(grid, logfolder_path, gpu):
    # Read Arguments from File

    global job_version  # for some reason nonlocal does not work
    job_version = 0

    def job_category_fold(category, fold):
        job = grid.copy()
        global job_version
        job["args"] = (
            " "
            + job["args"]
            + " --category {} --logpath {} --version {}".format(
                category, logfolder_path, job_version
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
        job_category_fold(category, fold)
        for category in grid["categories"]
        for fold in grid["folds"]
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

    table = np.empty(shape=(10, 6))

    for idx, method in enumerate(("l2", "sed", "maha"), step=2):
        for level in range(9):
            table[level, idx] = df[
                f"latent_{method}/level_{level}/auc/test/ROC/anomalies_mean"
            ].mean()
            table[level, idx + 1] = df[
                f"latent_{method}/level_{level}/auc/test/ROC/anomalies_mean"
            ].sem()
        table[9, idx] = df[f"{method}/full_auc/test/ROC/anomalies_mean"].mean()
        table[9, idx + 1] = df[
            f"{method}/full_auc/test/ROC/anomalies_mean"
        ].sem()

    column_index = [
        ["l2", "l2", "sed", "sed", "maha", "maha"],
        ["mean", "sem", "mean", "sem", "mean", "sem"],
    ]

    index = [f"level_{level}" for level in range(9)] + ["sum"]

    table = pd.DataFrame(table, index=index, columns=column_index)
    table = table * 100
    pd.options.display.float_format = "{:,.1f}".format

    table.to_csv("table1.csv")
    print(table)
