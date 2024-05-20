import glob
import pandas as pd

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
        
    print(count, 'data points')
    
    if task is None: return data
    else: return data[task]