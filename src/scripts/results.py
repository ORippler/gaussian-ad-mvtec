#!/usr/bin/env python3
"""Aggregate results script"""

import argparse
import os

import numpy as np
import pandas


def aggregate_version(path):
    """Return a pandas series for this version."""
    try:
        meta_tags = pandas.read_csv(
            os.path.join(path, "meta_tags.csv"), index_col="key"
        )
        metrics = pandas.read_csv(os.path.join(path, "metrics.csv"))
    except (FileNotFoundError, pandas.errors.EmptyDataError):
        print("skipping {}".format(path))
        return None

    # remove empty epoch start rows
    metrics = metrics.groupby("epoch").first()
    tags = pandas.Series(meta_tags["value"])
    if "auroc/val" in metrics.columns:
        tags["auroc/val_mean"] = metrics["auroc/val"].mean()
        tags["auroc/val_max"] = metrics["auroc/val"].max()
        tags["auroc/val_100"] = metrics["auroc/val"].get(100, np.NaN)
        tags["auroc/val_120"] = metrics["auroc/val"].get(120, np.NaN)
        tags["auroc/val_140"] = metrics["auroc/val"].get(140, np.NaN)
        tags["auroc/val_160"] = metrics["auroc/val"].get(160, np.NaN)
        tags["auroc/val_180"] = metrics["auroc/val"].get(180, np.NaN)
        tags["auroc/val_200"] = metrics["auroc/val"].get(200, np.NaN)
        tags["auroc/val_last"] = metrics.loc[len(metrics) - 1, "auroc/val"]
    if "loss/val" in metrics.columns:
        tags["loss/val_last"] = metrics.loc[len(metrics) - 1, "loss/val"]
    # Test auc scores have different names.
    auc_names = ["auc", "full_auc", "cc_auc", "weak_auc", "seg_auc"]
    for column in filter(
        lambda c: "/test" in c
        and c.split("/test", 1)[0]
        in [
            name + auc_name
            for name in ("maha/", "sed/", "l2/", "")
            for auc_name in auc_names
        ],
        metrics.columns,
    ):
        assert metrics[column].count() <= 1
        tags[column] = metrics[column].max()

    for column in filter(lambda c: c.startswith("npca_"), metrics.columns):
        assert metrics[column].count() <= 1
        tags[column] = metrics[column].max()
    for column in filter(
        lambda c: "/test" in c and c.startswith("latent_"), metrics.columns
    ):
        assert metrics[column].count() <= 1
        tags[column] = metrics[column].max()

    tags["epoch_last"] = metrics.index[-1]
    return tags


def aggregate(df, groups, fns=["mean", "sem"]):
    """Group and aggregate df into _mean and _sem columns.

    Keeps the grouping in the index (call .reset_index() to remove it).
    """
    # Fix groups that can be NaN so that pandas can sort them in groupby.
    str_columns = ["attach_blocks", "pca_file", "min_nb_iters", "max_nb_iters"]
    df = df.astype(
        {c: str for c in str_columns if c in df.columns}, copy=False
    )

    # df.groupby skips NaN values until version 1.1.0, which is why we
    # need this work around
    df = df.fillna("dummy")
    df = df.groupby([g for g in groups if g in df.columns]).agg(fns)
    df = df.replace("dummy", np.NaN)
    df.columns = df.columns.map("_".join)
    return df


def aggregate_folds(df, fns=["mean", "sem"]):
    """Group and aggregate the k-fold runs into _mean and _sem columns."""
    groups = [
        "model",
        "arch",
        "pretrained",
        "batch_size",
        "lr",
        "momentum",
        "median_filter",
        "attach_blocks",
        "pca_file",
        "min_nb_iters",
        "max_nb_iters",
        "category",
        "variance_threshold",
        "pca",
        "npca",
    ]
    return aggregate(df, groups, fns=fns)


def aggregate_categories(df, fns=["mean", "sem"]):
    groups = [
        "model",
        "arch",
        "pretrained",
        "batch_size",
        "lr",
        "momentum",
        "median_filter",
        "attach_blocks",
        "pca_file",
        "min_nb_iters",
        "max_nb_iters",
    ]
    return aggregate(df, groups, fns=fns)


def aggregate_results(path):
    """Aggregate results from multiple MVTec runs into one excel sheet"""
    versions = os.scandir(os.path.join(path, "lightning_logs"))
    df = pandas.DataFrame(
        filter(
            lambda s: s is not None,
            (aggregate_version(v.path) for v in versions),
        )
    )
    df = aggregate_folds(df).reset_index()
    # df = aggregate_categories(df).reset_index()

    # df.to_csv('results.csv')
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=".")
    args = parser.parse_args()

    df = aggregate_results(args.path)
    df.to_csv("./results.csv")
