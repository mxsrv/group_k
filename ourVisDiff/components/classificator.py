import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from serve.utils_llm import get_llm_output



class GPTClassificator:
    """
    Ask GPT if the hypothesis is true or false.
    """

    prompt = """I am a machine learning reseracher summarizing differences in groups of images. 

Given a description of concepts being present in Group A and Group B, output whether a given set of captions aligns better with the concepts of Group A or with those concepts of Group B. 
Answer with a 1 (more aligned with Group A), or 0 (more aligned with Group B). 
A score of 1 should be given if the captions are more true for the concepts of A than B. A score of 0 should be given if the captions are more true for the concepts of B than A.

Here is the descriptions
Group A: {concepts_a}.
Group B: {concepts_b}. 
Captions: {captions}

Again, output either a 1, or 0. Response:"""

    def __init__(self, args: Dict):
        self.args = args
        self.decision_boundary = 0.35

    def decide_group(self, score):
        if score >= self.decision_boundary:
            return 1 # ai
        else:
            return 0 # nature

    def calculate_score(self, answer):
        score = -1
        try:
            score = int(answer)
        except ValueError:
            amount_1 = answer.count("1")
            amount_0 = answer.count("0")
            if amount_1 == 1 and amount_0 == 0:
                score = 1
            if amount_0 == 1 and amount_1 == 0:
                score = 0
        return score


    def evaluate_seperately(
        self, args: Dict, test_captions: List[str], train_ai_hypotheses: List[str], train_nature_hypotheses: List[str]
    ) -> Tuple[Dict, List[Dict]]:
        # varify that the hypothesis is true or false
        
        ai_hypotheses_string = ""
        for i, caption in enumerate(train_ai_hypotheses):
            if i < self.args["n_hypotheses"]:
                ai_hypotheses_string += caption + ", "
        
        nature_hypotheses_string = ""
        for i, caption in enumerate(train_nature_hypotheses):
            if i < self.args["n_hypotheses"]:
                nature_hypotheses_string += caption + ", "

        scores = []
        evaluated_hypotheses = []

        for i, caption in enumerate(test_captions):
            if i < self.args["n_captions"]:

                prompt = self.prompt.format(captions=caption, concepts_a=ai_hypotheses_string, concepts_b=nature_hypotheses_string)
                answer = get_llm_output(prompt, self.args["model"])
                score = self.calculate_score(answer)
                
                evaluated_hypotheses.append({"prompt": prompt, "score": score, "response": answer})
                    
                if score == 0 or score == 1:  
                    scores.append(score)

        score = np.mean(scores)
        group = self.decide_group(score)
        return group, score, evaluated_hypotheses

    def evaluate_combined(
        self, args: Dict, test_captions: List[str], train_ai_hypotheses: List[str], train_nature_hypotheses: List[str]
    ) -> Tuple[Dict, List[Dict]]:
        # varify that the hypothesis is true or false
        test_caption_string = ""
        for i, caption in enumerate(test_captions):
            if i < self.args["n_captions"]:
                test_caption_string += caption + ", "

        ai_hypotheses_string = ""
        for i, caption in enumerate(train_ai_hypotheses):
            if i < self.args["n_hypotheses"]:
                ai_hypotheses_string += caption + ", "
        
        nature_hypotheses_string = ""
        for i, caption in enumerate(train_nature_hypotheses):
            if i < self.args["n_hypotheses"]:
                nature_hypotheses_string += caption + ", "

        prompt = self.prompt.format(captions=test_caption_string, concepts_a=ai_hypotheses_string, concepts_b=nature_hypotheses_string)
        answer = get_llm_output(prompt, self.args["model"])
        score = self.calculate_score(answer)
        group = self.decide_group(score)

        evaluated_hypotheses = [{"prompt": prompt, "score": score, "response": answer}]

        return group, score, evaluated_hypotheses


class NullEvaluator:
    def __init__(self, args: Dict):
        self.args = args

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
        return {}, [{}]


def test_evaluator():
    args = {
        "model": "gpt-4",
        "n_hypotheses": 20,
    }
    evaluator = GPTEvaluator(args)
    hypotheses = [
        "dogs in the snow",
        "golden retrivers on a ski slope",
        "animals in the snow",
        "dogs in winter time",
    ]
    gt_a = "images of dogs in the snow"
    gt_b = "images of dogs next to cats"
    metrics, evaluated_hypotheses = evaluator.evaluate(hypotheses, gt_a, gt_b)
    print(metrics)
    print(evaluated_hypotheses)


if __name__ == "__main__":
    test_evaluator()
