#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Artifact download start")
    artifact_path = run.use_artifact(args.input_artifact).file()
    art_df = pd.read_csv(artifact_path)

    logger.info("Outliers being dropped")
    drop_indexes = art_df['price'].between(args.min_price, args.max_price)
    art_df = art_df[drop_indexes].copy()

    idx = art_df['longitude'].between(-74.25, -73.50) & art_df['latitude'].between(40.5, 41.2)
    art_df = art_df[idx].copy()


    # Save the cleaned dataset
    logger.info("Saving cleaned Artifact")
    file_name = "clean_sample.csv"
    art_df.to_csv(file_name, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file_name)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type= str,
        help= 'input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type= str,

        required=True
    )

    parser.add_argument(
        "--output_type",
        type= str,

        required=True
    )

    parser.add_argument(
        "--output_description",
        type= str,

        required=True
    )

    parser.add_argument(
        "--min_price",
        type= float,

        required=True
    )

    parser.add_argument(
        "--max_price",
        type= float,

        required=True
    )


    args = parser.parse_args()

    go(args)
