import glob
import json
import numpy as np
import pandas as pd

def loadIdeology(
    trainDir='/scratch/fl1092/code_ideology_of_science/data',
    testDir='/scratch/fl1092/altmetrics_political_leaning',
    classifiedDir='/scratch/fl1092/code_ideology_of_science/res_ideology/gemini-1.5-flash__True__1__0.6__10__0.3',
    subtask=None
):
    
    data = {'train': [], 'test': []}

    ## load train ##
    if subtask == 'climate':
        columns=['Abstract', 'ClimateRationale', 'IsClimate']
        colMap = {'ClimateRationale': 'rationale', 'IsClimate': 'label'}
    elif subtask == 'climateSolution':
        columns=['Abstract', 'SolutionRationale', 'IsClimate', 'IsSolution']
        colMap = {'SolutionRationale': 'rationale', 'IsSolution': 'label'}
    elif subtask == 'climateSolutionEmission':
        columns=['Abstract', 'SolutionRationale', 'IsSolution', 'IsEmission']
        colMap = {'SolutionRationale': 'rationale', 'IsEmission': 'label'}
    elif subtask == 'climateImpact':
        columns=['Abstract', 'ImpactRationale', 'IsClimate', 'IsImpact']
        colMap = {'ImpactRationale': 'rationale', 'IsImpact': 'label'}
    elif subtask == 'climateImpactNegative':
        columns=['Abstract', 'ImpactRationale', 'IsImpact', 'IsNegative']
        colMap = {'ImpactRationale': 'rationale', 'IsNegative': 'label'}
    elif subtask == 'climateImpactAlarmism':
        columns=['Abstract', 'ImpactRationale', 'IsImpact', 'IsAlarmism']
        colMap = {'ImpactRationale': 'rationale', 'IsAlarmism': 'label'}
    else:
        raise Exception('Undefined task')

    df = (
        pd.read_csv(f'{trainDir}/Manual-ClimatePaper-Paper-Classification-May8.csv', usecols=columns)
        .head(50).fillna('nan')
        .rename(columns=colMap)
    )

    if subtask == 'climateSolution' or subtask == 'climateImpact':
        df = df.query('IsClimate == "Yes"')
    elif subtask == 'climateSolutionEmission':
        df = df.query('IsSolution == "Yes"')
    elif subtask == 'climateImpactNegative' or subtask == 'climateImpactAlarmism':
        df = df.query('IsImpact == "Yes"')

    print("Shape of training data: ", df.shape)

    for ind, row in df.iterrows():
        data['train'].append({'idx': ind, 'text': row['Abstract'], 'rationale': row['rationale'], 'label': row['label'].lower()})
    ## load train finished ##


    ## load test ##
    df = (
        pd.read_csv(f'{testDir}/ClimatePapersAbstractJan18.csv')
        .dropna(subset=['DOI','Title','Abstract'])
        .query('Abstract != "UNKNWON ABSTRACT"')
        .query('Abstract != "MORE THAN ONE MATCH"')
        .query('Abstract != ""')
    )

    if subtask == 'climateSolution' or subtask == 'climateImpact':
        classified = pd.read_csv(f'{classifiedDir}/ideology_climate_test/classify.csv',index_col=0).rename(columns={'idx':'ID'})
        df = df.merge(classified, on='ID').query('classify == "yes"')

    elif subtask == 'climateSolutionEmission':
        classified = pd.read_csv(f'{classifiedDir}/ideology_climateSolution_test/classify.csv',index_col=0).rename(columns={'idx':'ID'})
        df = df.merge(classified, on='ID').query('classify == "yes"')

    elif subtask == 'climateImpactNegative' or subtask == 'climateImpactAlarmism':
        classified = pd.read_csv(f'{classifiedDir}/ideology_climateImpact_test/classify.csv',index_col=0).rename(columns={'idx':'ID'})
        df = df.merge(classified, on='ID').query('classify == "yes"')

    # load result from previous steps to #

    print("Shape of test data: ", df.shape)

    for ind, row in df.iterrows():
        data['test'].append({'idx': row['ID'], 'text': row['Abstract'], 'label': ''})

    return data


def loadGSM8k(rootDir = '/scratch/fl1092/data_common/grade-school-math/grade_school_math/data', subtask=None, returnTask=None):
    
    data = {}
    for task in ['train', 'test']:

        data[task] = []
        with open(f'{rootDir}/{task}.jsonl') as f:
            for ind, line in enumerate(f):
                m = json.loads(line)
                data[task].append({'question': m['question'], 'answer': m['answer'], 'idx': ind, 'label': m['answer'].split('####')[-1].strip()})

    if subtask is not None:
        data['test'] = np.array_split(data['test'], 8)[int(subtask)]

    if returnTask is None: return data
    else: return data[returnTask]


def loadCSQA(rootDir = '/scratch/fl1092/data_common/CommonSenseQA', subtask=None, returnTask=None):

    inputTemplate='''\
Question: {q}
Choices:
a. {a}
b. {b}
c. {c}
d. {d}
e. {e}
    '''
    
    data = {}
    for task in ['train', 'dev']:

        data[task] = []
        with open(f'{rootDir}/{task}_rand_split.jsonl') as f:
            for ind, line in enumerate(f):
                m = json.loads(line)
                data[task].append({
                    'question': inputTemplate.format(
                        q=m['question']['stem'],
                        a=m['question']['choices'][0]['text'],
                        b=m['question']['choices'][1]['text'],
                        c=m['question']['choices'][2]['text'],
                        d=m['question']['choices'][3]['text'],
                        e=m['question']['choices'][4]['text'],
                    ),
                    'idx': ind,
                    'label': m['answerKey'].lower()
                })

    if subtask is not None:
        data['dev'] = np.array_split(data['dev'], 8)[int(subtask)]

    if returnTask is None: return data
    else: return data[returnTask]


def loadMMLU(rootDir = '/scratch/fl1092/data_common/MMLU', task=None):
    
    inputTemplate='''\
Question: {q}
Choices:
a. {a}
b. {b}
c. {c}
d. {d}
    '''
    
    data = {}
    count = 0
    
    for file in glob.glob(f'{rootDir}/*/*.csv'):
        
        split = file.split('/')[-2]
        if split == 'auxiliary_train':
            continue
            
        fileName = file.replace('.csv', '').split('/')[-1].split('_')
        subtask, taskSplit = '_'.join(fileName[:-1]), fileName[-1]
        
        assert(split == taskSplit)
        
        if subtask not in data:
            data[subtask] = {
                'dev': [], # dev is for few-shot
                'test': [],
                'val': []
            }
        
        df = pd.read_csv(file, names=['question','A','B','C','D','label'])
        
        for ind, row in df.iterrows():
            data[subtask][split].append(
                {'question': inputTemplate.format(
                    q=row['question'],
                    a=row['A'],
                    b=row['B'],
                    c=row['C'],
                    d=row['D']
                ), 'label': row['label'].lower(), 'idx': ind}
            )
            count += 1
            
    for key, v in data.items():
        assert(len(v['dev'])!=0)
        assert(len(v['test'])!=0)
        assert(len(v['val'])!=0)
        
    print(count, 'data points', 'test:', len(v['test']))
    
    if task is None: return data
    else: return data[task]