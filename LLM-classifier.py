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
parser.add_argument("--logDir", type=str, default='./log')
parser.add_argument("--resultDir", type=str, default='./exp_result')
parser.add_argument("--taskFile", type=str, default='tasks.json')
parser.add_argument("--format", type=str, default='explicit')
parser.add_argument("--customizedPrompt", type=bool, default=False)

args = parser.parse_args()
print('List of arguments:', args)
        
modelID = 'meta-llama/Meta-Llama-3-8B-Instruct'
llm = LLM(modelID)

runClassification(
    args.task, args.split, llm,
    resFormat=args.format,
    logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
    correctionModes=[],
    customizePrompt=args.customizedPrompt
)
