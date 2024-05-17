import random
import json

import pandas as pd
from tqdm.notebook import tqdm

from MoP.classifier import LLMClassifier
from MoP.llm import LLM

def runClassification(
    task, split, llm, resFormat, logDir, resultDir, taskFile,
    correctionModes=['array', 'vote', 'single', 'reason'], customizePrompt=False, sampleSize=200
):
    
    with open(taskFile, 'r') as f:
        dataCards = json.load(f)
        
    clf = LLMClassifier(
        task=task, split=split, resFormat=resFormat, taskJson=dataCards[task],
        llm=llm, numSeq=7, logDir=logDir, resultDir=resultDir, customizePrompt=customizePrompt
    )
    
    print(task, split, resFormat, clf.dataset[split], flush=True)
    
    for idx in range(min(sampleSize, len(clf.dataset[split]))):
        
        clf.classify(idx)
        
        for mode in correctionModes: 
            clf.correction(idx, mode)
            
        print(flush=True)
    print(flush=True)
        



def splitExampleTrainTest(df: pd.DataFrame, on: str, num: int):
    
    fewShot = []
    train = [] # TODO
    test = []
    
    for val in df[on].unique():
        
        subset = df[df[on] == val]
        if subset.shape[0] == 1:
            print(f'{on} col only has one row of {val}')
            
        fewShot.append(subset.head(num))
        
    fewShot = pd.concat(fewShot)
    test = df[~df.index.isin(fewShot.index)]
    
    return fewShot, test

def convertData(df: pd.DataFrame, inCol: str, outCols: str):
    
    return [convertRow(row, inCol, outCols) for ind, row in df.iterrows()]

def convertRow(row, inCol: str, outCols: str):
    
    outJson = {}
    for col in outCols:
        outJson[col] = row[col]
        
    return (row[inCol].lower(), json.dumps(outJson))

def classifyDataframe(
    task: str, outcomeCol: str, fewShot, inputDf: pd.DataFrame, modelID: str, outDir: str,
    llm=None, numSeq=7, correction=False, promptMixture=False, chunkIndex=None
):

    if llm is None:
        llm = LLM(modelID)

    assert(chunkIndex is not None)
    
    outDir = f"{outDir}/{task}_{outcomeCol}_{modelID.replace('/', '__')}_{correction}_{promptMixture}"
    outFile = f"{outDir}/{chunkIndex}.csv"

    if len(glob.glob(outFile)) != 0:
        resultDf = pd.read_csv(outFile)
    else:
        resultDf = pd.DataFrame()

    print(f'{resultDf.shape[0]} are processed')

    for ind, row in inputDf.iterrows():

        print('ID is:', row['ID'])

        if row['ID'] in resultDf.ID:
            print(f'{row["ID"]} exists ...', flush=True)
            continue

        res = {'ID': row['ID']}

        classifier = LLMClassifier(task=task, llm=llm, text=row['Abstract'], numSeq=numSeq)
        ret = classifier.classify(fewShot, maxNewToken=200)

        if ret==0:
            res[outcomeCol] = 'ERROR'
            continue
        
        if correction:
            classifier.correction(outcome=outcomeCol, promptMixture=promptMixture)

        try:
            res[outcomeCol] = classifier.response_agg[outcomeCol]
        except Exception as e:
            res[outcomeCol] = 'ERROR'

        resultDf = pd.concat([resultDf, pd.DataFrame(res)], ignore_index=True, sort=False)
        resultDf.to_csv(outFile, index=False)

        print(flush=True)

def runTest(
    task, fewShot, testData, modelID, outcomeCol,
    llm=None, numSeq=1, correction=False, promptMixture=False, promptArray=False, verbose=True
):

    if llm is None:
        llm = LLM(modelID)
    
    resultDf = []
    
    vanillaCorrect = 0 # without correction
    correct = 0
    total = 0
    for ind, (content, output) in tqdm(enumerate(testData)):
        print(ind, end='\t')
        total += 1

        output = json.loads(output)

        classifier = LLMClassifier(task=task, llm=llm, text=content, label=output, numSeq=numSeq)
        ret = classifier.classify(fewShot, maxNewToken=200)

        if ret==0:
            print("ERROR: failed to classifiy")
            
            result = {'Content': content, 'Label': output, modelID: {}}
            result[f'{modelID}_{task}'] = "ERROR"
            resultDf.append(result)

            continue

        if classifier.verify():
            vanillaCorrect += 1
        
        if correction:
            classifier.correction(outcome=outcomeCol, promptMixture=promptMixture, promptArray=promptArray, verbose=verbose)
            
        pred = {outcomeCol: classifier.response_agg[outcomeCol]}
        result = {'Content': content, 'Label': output, modelID: pred}

        try:

            if classifier.verify():
                print(f'Correct    {pred[outcomeCol]}')
                
                correct += 1
                result[f'{modelID}_{task}'] = "Correct"
            else:
                print('Wrong', classifier.response_agg)
                print('Model:', random.sample(classifier.response, min(len(classifier.response), 2)))
                print('Label:', output)
                print(content[:100])
                
                result[f'{modelID}_{task}'] = "Wrong"

            print(flush=True)
        except Exception as e:
            print("ERROR", e)
            print(pred)

            result[f'{modelID}_{task}'] = "ERROR"
            
        resultDf.append(result)

        print(f'Out of {total} classified, {vanillaCorrect} are correct before correction, {correct} are correct after correction.\n\n', flush=True)

    print(f'{task} {modelID} {correct} correct out of {len(testData)}')
    
    return pd.DataFrame(resultDf)