import hashlib
import json
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

import components.prompts as prompts
import wandb
from serve.utils_general import save_data_diff_image
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_embed_caption_blip, get_vlm_output


class Captioner:
    def __init__(self, args: Dict):
        self.args = args

    def propose(
        self, dataset1: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_captions = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            captions = self.get_captions(sampled_dataset1)
            all_captions.extend(captions)
        return all_captions

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        return random.sample(dataset, n)

    def visualize(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Dict:
        images1 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset1
        ]
        images2 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset2
        ]
        images = {"images_group_1": images1, "images_group_2": images2}
        return images

    def captioning(self, dataset: List[Dict]):
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                self.args["captioner"]["prompt"],
                self.args["captioner"]["model"],
            )

    def get_captions(
        self, sampled_dataset1: List[Dict]
    ) -> Tuple[List[str], Dict]:
        self.captioning(sampled_dataset1)
        captions1 = [
            f"{item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset1
        ]
        return captions1

