import json
import pandas as pd
from tqdm.notebook import tqdm
import torch
from itertools import product
import random
random.seed(42)

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

parser.add_argument("--successive_halving", action="store_true")
parser.add_argument("--halving_round", type=int, default=0)

parser.add_argument("--experiment_mode", action="store_true")


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

if args.successive_halving:

    assert(args.task == 'mmlu')

    halvRound = args.halving_round

    if halvRound > 0:
        params = list(pd.read_csv(f'./succ_halv_results/round{halvRound-1}.csv').Param.values)
        print(f'Round {halvRound}: {len(params)} in this round')

    beams = [1, 4, 5, 10]
    temps = [0.4, 0.5, 0.6, 0.7]
    ks = [10, 30, 50]
    ps = [0.3, 0.5, 0.7, 1]
    sampleSize = 2**halvRound*10

    subTasks = list(loadMMLU().keys())
    sampledTasks = random.sample(subTasks, 5)

    print("Sampled tasks", sampledTasks)
    print('Number of samples from each task', sampleSize, flush=True)

    for b, t, k, p in product(beams, temps, ks, ps):

        llm.set_params(b, t, k, p)

        if halvRound > 0:
            paramStr = f'{llm.do_sample}_{llm.num_beams}_{llm.temperature}_{llm.top_k}_{llm.top_p}'
            if paramStr not in params:
                print(f'Skipping parameter combination {paramStr}')
                continue

        for subtask in sampledTasks:

            print(f'Running MMLU task: mmlu-{subtask} {llm.num_beams}__{llm.temperature}__{llm.top_k}__{llm.top_p} ...', flush=True)

            runClassification(
                f'mmlu-{subtask}', args.split, llm,
                resFormat=args.format,
                logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
                sampleSize=sampleSize,
                correctionModes=['vote', 'single']
            )
            print('\n\n\n', flush=True)
    
else:

    if args.experiment_mode:
        correctionModes = [f'experiment-{i}' for i in range(12)]
        print('All correction modes', '\t'.join(correctionModes), flush=True)
    else:
        correctionModes = ['vote', 'single', 'array'] 

    if args.task == 'mmlu' or args.task == 'mmlu1000':
        subTasks = list(loadMMLU().keys())

        for subtask in subTasks:

            print(f'Running MMLU task: mmlu-{subtask} ...')

            runClassification(
                f'{args.task}-{subtask}', args.split, llm,
                resFormat=args.format,
                logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
                sampleSize=200, correctionModes=correctionModes
            )
            print('\n\n\n', flush=True)
    else:
        runClassification(
            args.task, args.split, llm,
            resFormat=args.format,
            logDir=args.logDir, resultDir=args.resultDir, taskFile=args.taskFile,
            sampleSize=200, correctionModes=correctionModes
        )
