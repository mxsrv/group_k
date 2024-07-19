import logging
from typing import Dict, List, Tuple
import os
import numpy as np

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
from components.captioner import Captioner
from components.classificator import GPTClassificator


def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name="test",
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]
    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv", sep=";")

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


def caption(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    captioner_args = args["proposer"]
    captioner_args["seed"] = args["seed"]
    captioner_args["captioner"] = args["captioner"]

    captioner = Captioner(captioner_args)
    captions1 = captioner.propose(dataset1)
    captions2 = captioner.propose(dataset2)
    return captions1, captions2


def classify(args: Dict, test_captions: List[str], train_ai_hypotheses: List[str], train_nature_hypotheses: List[str], group_name: str) -> Dict:
    evaluator_args = args["evaluator"]

    classificator = GPTClassificator(evaluator_args)

    group, score, evaluated_hypotheses = classificator.evaluate_seperately(
        args, test_captions, train_ai_hypotheses, train_nature_hypotheses
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses " + group_name + " " + args["data"]["name"]: table_evaluated_hypotheses}, commit=False)
    return group, score



@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    print(f"Args {args}")

    tryLoadCaptions = True
    saveCaptions = True

    split = "test"
    args["data"]["root"] = "../" + split

    # get all folder names
    csv_names = os.listdir('../' + split)
    folder_names = []
    for csv_name in csv_names:
        folder_names.append(csv_name.split('.')[0])

    #prepare classwise classification-accuracy scores
    count_classified = 0
    correctly_classified = 0
    correctly_ai_classified = 0
    count_ai_classified = 0
    correctly_nature_classified = 0
    count_nature_classified = 0

    scores_ai = {}
    scores_nature = {}
    false_classes = []

    for name in folder_names:
        print()
        print()
        logging.info("Processing name " + name)

        args["data"]["name"] = name
        args["data"]["group1"] = "nature" + name[name.find('_'):]
        args["data"]["group2"] = "ai" + name[name.find('_'):]

        ai_captions = []
        nature_captions = []
        if tryLoadCaptions:
            logging.info("Loading captions...")
            try:
                with open(split + "_results/captions_ai_" + args["data"]["name"] + ".txt", "r") as file:
                    for line in file:
                        ai_captions.append(line.rstrip())
                with open(split + "_results/captions_nature_" + args["data"]["name"] + ".txt", "r") as file:
                    for line in file:
                        nature_captions.append(line.rstrip())
            except:
                pass

        if ai_captions == [] or nature_captions == []:
            logging.info("Loading data...")
            dataset1, dataset2, group_names = load_data(args)
            # print(dataset1, dataset2, group_names)

            logging.info("Proposing captions...")
            nature_captions, ai_captions = caption(args, dataset1, dataset2)

            if saveCaptions:
                logging.info("Saving captions...")
                with open(split + "_results/captions_ai_" + args["data"]["name"] + ".txt", "w") as f:
                    for hypothesis in ai_captions:
                        if hypothesis.startswith('"') and hypothesis.endswith('"'):
                            hypothesis = hypothesis[1:-1]
                        f.write(hypothesis + "\n")

                with open(split + "_results/captions_nature_" + args["data"]["name"] + ".txt", "w") as f:
                    for hypothesis in nature_captions:
                        if hypothesis.startswith('"') and hypothesis.endswith('"'):
                            hypothesis = hypothesis[1:-1]
                        f.write(hypothesis + "\n")

        logging.info("Loading training hypotheses...")        
        train_ai_hypotheses = []
        train_nature_hypotheses = []
        try:
            with open("train_results/ranked_hypotheses_ai_" + args["data"]["name"] + ".txt") as file:
                for line in file:
                    train_ai_hypotheses.append(line.rstrip())
            with open("train_results/ranked_hypotheses_nature_" + args["data"]["name"] + ".txt") as file:
                for line in file:
                    train_nature_hypotheses.append(line.rstrip())
        except:
            pass

        if train_ai_hypotheses == [] or train_nature_hypotheses == []:
            logging.info("no training results loaded, abort")
            continue

        logging.info("Classify captions...")

        group_ai, score_ai = classify(args, ai_captions, train_ai_hypotheses, train_nature_hypotheses, "ai")
        scores_ai[name] = (group_ai, score_ai)
        print("gt 1:", group_ai, score_ai)
        if group_ai != -1:
            count_classified += 1
            count_ai_classified += 1
            if group_ai == 1:
                correctly_classified += 1
                correctly_ai_classified += 1

        group_nature, score_nature = classify(args, nature_captions, train_ai_hypotheses, train_nature_hypotheses, "nature")
        scores_nature[name] = (group_nature, score_nature)
        print("gt 0:", group_nature, score_nature)
        if group_nature != -1:
            count_classified += 1
            count_nature_classified += 1
            if group_nature == 0:
                correctly_classified += 1
                correctly_nature_classified += 1

        if group_ai == 0 or group_nature == 1:
            false_classes.append(name)

        if args["wandb"]:
            wandb.log({
                "ai_score": score_ai, 
                "nature_score" : score_nature, 
                })

    # print classification results
    print(false_classes)
                
    print()
    print("Correctly classified:", correctly_classified, "out of", count_classified)
    print("Correctly ai classified:", correctly_ai_classified, "out of", count_ai_classified)
    print("Correctly nature classified:", correctly_nature_classified, "out of", count_nature_classified)
    if count_classified > 0:
        print("Accuracy:", correctly_classified / count_classified)
        print("Scores ai:", np.mean(list(scores_ai.values())), np.std(list(scores_ai.values())), "nature:", np.mean(list(scores_nature.values())), np.std(list(scores_nature.values())))
        print("ai", np.unique(list(scores_ai.values()), return_counts=True))
        print("nature", np.unique(list(scores_nature.values()), return_counts=True))

        model_accuracy = {}
        for model_name in ["adm", "biggan", "stablediffusion"]:
            model_count_classified = 0
            model_correctly_classified = 0
            for key in scores_ai.keys():
                if model_name in key:
                    model_count_classified += 1
                    if scores_ai[key][0] == 1:
                        model_correctly_classified += 1
            for key in scores_nature.keys():
                if model_name in key:
                    model_count_classified += 1
                    if scores_nature[key][0] == 0:
                        model_correctly_classified += 1
            if model_count_classified > 0:
                model_accuracy[model_name] = model_correctly_classified / model_count_classified

        print(model_accuracy)

        if args["wandb"]:
            print()
            wandb.log({
                "count_classified": count_classified, 
                "count_ai_classified": count_ai_classified, 
                "count_nature_classified": count_nature_classified, 
                "correctly_classified" : correctly_classified, 
                "correctly_ai_classified" : correctly_ai_classified, 
                "correctly_nature_classified" : correctly_nature_classified, 
                "accuracy": correctly_classified / count_classified,
                "accuracy_ai": correctly_ai_classified / count_ai_classified,
                "accuracy_nature": correctly_nature_classified / count_nature_classified,
                "adm_accuracy": model_accuracy["adm"],
                "biggan_accuracy": model_accuracy["biggan"],
                "stablediffusion_accuracy": model_accuracy["stablediffusion"],
                })

if __name__ == "__main__":
    main()
