import logging
from typing import Dict, List, Tuple
import os

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator, ReverseGPTEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    VLMFeatureProposer,
    VLMProposer,
)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker


def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name="eval_h",
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]
    print(f"Name: {data_args['name']}")
    print(f"{data_args['root']}/{data_args['name']}.csv")
    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv", sep=";")
    print(f"Row One {df.keys()}")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )
    print(f"Group name {df['group_name'].unique()}")
    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names



def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses_" + args["data"]["name"]: table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    print(f"Args {args}")

    # get all folder names
    csv_names = os.listdir('../data')
    folder_names = []
    for csv_name in csv_names:
        if csv_name.endswith('.csv'):
            folder_names.append(csv_name.split('.')[0])

    for name in folder_names:
        print()
        print()
        logging.info("Processing name " + name)
        args["data"]["name"] = name
        args["data"]["group1"] = "nature_" + name.split('_')[1]
        args["data"]["group2"] = "ai_" + name.split('_')[1]


        logging.info("Loading data...")
        dataset1, dataset2, group_names = load_data(args)
        # print(dataset1, dataset2, group_names)

        logging.info("Loading ranked hypotheses...")
        ranked_hypotheses = []
        with open("mydata/ranked_hypotheses_" + args["data"]["name"] + ".txt") as file:
            for line in file:
                ranked_hypotheses.append(line.rstrip())
        print(ranked_hypotheses)

        logging.info("Evaluating hypotheses...")
        metrics = evaluate(args, ranked_hypotheses, group_names)
        print(metrics)

        logging.info("Saving evaluation...")
        with open("mydata/reverse_eval_" + args["data"]["name"] + ".txt", "w") as f:
            for key in metrics:
                f.write(key + " : " + str(metrics[key]) + "\n")


if __name__ == "__main__":
    main()
