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
            name="gen_h",
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


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict], name_str) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs_"+ name_str + "_" + args["data"]["name"]: wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses




@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    print(f"Args {args}")

    split = "test"
    args["data"]["root"] = "../group_k/" + split

    # get all folder names
    csv_names = os.listdir('../group_k/' + split)
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

        logging.info("Proposing hypotheses...")
        nature_hypotheses = propose(args, dataset1, dataset2, "nature")
        ai_hypotheses = propose(args, dataset2, dataset1, "ai")

        logging.info("Saving hypotheses...")
        with open(split + "_results/hypotheses_ai_" + args["data"]["name"] + ".txt", "w") as f:
            for hypothesis in ai_hypotheses:
                if hypothesis.startswith('"') and hypothesis.endswith('"'):
                    hypothesis = hypothesis[1:-1]
                f.write(hypothesis + "\n")

        with open(split + "_results/hypotheses_nature_" + args["data"]["name"] + ".txt", "w") as f:
            for hypothesis in nature_hypotheses:
                if hypothesis.startswith('"') and hypothesis.endswith('"'):
                    hypothesis = hypothesis[1:-1]
                f.write(hypothesis + "\n")


if __name__ == "__main__":
    main()
