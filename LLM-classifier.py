import json
import pandas as pd
from tqdm.notebook import tqdm
import torch

from MoP.llm import LLM
from MoP.classifier import LLMClassifier
from MoP.experiment import runClassification
from MoP.dataloader import loadMMLU

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
parser.add_argument("--customizedPrompt", action="store_true")

parser.add_argument("--sample", action="store_true")
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=1)


args = parser.parse_args()
print('List of arguments:', args)
        
modelID = 'meta-llama/Meta-Llama-3-8B-Instruct'
llm = LLM(
    modelID,
    do_sample=args.sample,
    num_beams=args.num_beams,
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p
)

if args.task == 'mmlu':
    subTasks = list(loadMMLU().keys())

    for subtask in subTasks:

        print(f'Running MMLU task: mmlu-{subtask} ...')

        runClassification(
            f'mmlu-{subtask}', args.split, llm,
            resFormat=args.format,
            logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
            sampleSize=200
        )
        print('\n\n\n\n')
else:

    runClassification(
        args.task, args.split, llm,
        resFormat=args.format,
        logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
        sampleSize=200
    )
