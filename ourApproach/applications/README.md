# Multimodal AI Project Group K
## Detecting Fake Images through Semantic Analysis

In this repository we describe the steps to reproduce our approach for detecting fake images through semantic analysis.
In the test and train folders there are csv files enumerating the images of the respective splits.
The respective dataset was not uploaded to the git but can be uploaded elsewhere if requested or be found in the lightning-ai studio in the subset_evaluation folder.

In the folder ourApproach lies our adaptation of the https://github.com/Understanding-Visual-Datasets/VisDiff repository to distinguish fake images.
We created the following files: components/captioner.py, components/classificator.py, generate_hypothesis.py, rank_hypothesis.py, test.py.
For preprocessing the dataset and creating subsets, we use csv_generator.py and data_subset_extraction.py.
We decided to keep original VisDiff files inside the project to be able to try out different combinations of proposer and ranker.

In the subfolder train_results are the generated and ranked image hypotheses of the trainset, our "trained model".
In the subfolder test_results are buffered captions of the testset so that BLIP doesnt need to be used at every test run.

Valid keys for gpt/openai (and wandb) need to be used to reproduce our results.

### Training
For retraining first new hypotheses need to be generated. Therefore BLIP and generate_hypothesis.py needs to be started with the following commands 
```bash
python serve/vlm_server_blip.py 
python generate_hypothesis.py --config configs/base.yaml
```
Next the hypotheses needs to be ranked using CLIP and rank_hypotheses with
```bash
python serve/clip_server.py 
python rank_hypothesis.py --config configs/base.yaml
```
Those models can't be started simultaneously on one gpu with max 24GB VRAM, thats why the script is splitted as it is.

### Testing
For running interference on a test image set the test.py script needs to be run with the BLIP model.
```bash
python serve/vlm_server_blip.py 
python test.py --config configs/base.yaml
```

