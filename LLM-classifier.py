import json
import pandas as pd
from tqdm.notebook import tqdm
import torch

from MoP.llm import LLM
from MoP.classifier import LLMClassifier
from MoP.experiment import runClassification

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str)
parser.add_argument("--split", type=str)
args = parser.parse_args()

        
modelID = 'meta-llama/Meta-Llama-3-8B-Instruct'
llm = LLM(modelID)

runClassification(args.task, args.split, llm)


