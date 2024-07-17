import logging
from typing import Dict, List, Tuple
import os

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
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
            name="rank_h",
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



def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses_" + args["data"]["name"]: table_hypotheses})
        for i in range(5):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    if args["wandb"]:
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth_" + args["data"]["name"]: table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses]



@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    print(f"Args {args}")

    split = "train"
    args["data"]["root"] = "../" + split


    # get all folder names
    csv_names = os.listdir('../' + split)
    folder_names = []
    for csv_name in csv_names:
        folder_names.append(csv_name.split('.')[0])

    for name in folder_names:
        print()
        print()
        logging.info("Processing name " + name)
        args["data"]["name"] = name
        args["data"]["group1"] = "nature" + name[name.find('_'):]
        args["data"]["group2"] = "ai" + name[name.find('_'):]

        logging.info("Loading data...")
        dataset1, dataset2, group_names = load_data(args)
        print(group_names, len(dataset1), len(dataset2))

        for prefix in ["ai", "nature"]:

            logging.info("Loading hypotheses...")
            hypotheses = []
            with open(split + "_results/hypotheses_" + prefix + "_" + args["data"]["name"] + ".txt") as file:
                for line in file:
                    hypotheses.append(line.rstrip())
            print(hypotheses)

            logging.info("Ranking hypotheses...")
            if prefix == "ai":
                ranked_hypotheses = rank(args, hypotheses, dataset2, dataset1, group_names)
            else:
                ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
            print(ranked_hypotheses)

            logging.info("Saving rankedhypotheses...")
            with open(split + "_results/ranked_hypotheses_" + prefix + "_" + args["data"]["name"] + ".txt", "w") as f:
                for ranked_hypothesis in ranked_hypotheses:
                    f.write(ranked_hypothesis + "\n")



if __name__ == "__main__":
    main()
