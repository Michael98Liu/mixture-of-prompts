import os
import glob
import json
import random

from datasets import load_dataset
import pandas as pd

from promptarray.generate import arrayGenerate
from promptarray.generator import PromptArrayGenerator

from .llm import LLM
from .dataloader import loadMMLU, loadGSM8k, loadCSQA

# MODES = ['classify', 'exp'] # 'correct', 'mistake', 'vote', 'array', 'single', 'reason',


class LLMClassifier:

    def __init__(
        self, task, taskJson, split, llm, allModes,
        resFormat='default', logDir='./log', resultDir='./exp_result', numSeq=1,
        customizePrompt=False
    ):
        taskList = task.split('-')
        if len(taskList) == 1:
            self.task = taskList[0]
            self.subtask = None
        elif len(taskList) == 2:
            self.task, self.subtask = taskList
        else:
            print(task)
            raise ValueError('format of task is wrong')

        self.task_obj = taskJson
        self.split = split
        self.inputKeys = self.task_obj['input_keys']
        self.llm = llm
        self.format = resFormat
        self.allModes = [x for x in allModes]
        self.allModes.extend(['classify'])
        if 'vote' in self.allModes:
            self.allModes.extend(['correct','mistake'])

        print("Task ID:", self.task_obj['task_id'], self.task, self.subtask)
        assert(self.split in ['train', 'validation', 'test', 'val', 'dev'])
        assert(self.format in ['default', 'json', 'explicit'])

        self.outDir = f'{resultDir}/{self.llm.model_id.replace("/", "__")}__{llm.do_sample}__{llm.num_beams}__{llm.temperature}__{llm.top_k}__{llm.top_p}/{task}_{self.split}'
        self.logDir = f'{logDir}/{self.llm.model_id.replace("/", "__")}__{llm.do_sample}__{llm.num_beams}__{llm.temperature}__{llm.top_k}__{llm.top_p}'
        self.logFile = f'{self.logDir}/{task}_{self.split}.jsonl'

        os.makedirs(self.outDir, exist_ok=True)
        os.makedirs(self.logDir, exist_ok=True)

        self._initializeResults() # if does not exist, initialize; if exists, load
        
        self.response = None # classification responses
        self.response_agg = '' # response as a result of majority vote
        self.score = -1 # confidence score associated with response_agg
        self.isCorrect = None
        self.reflections = []
        self.numSeq = numSeq # number of sequences to generate

        if 'labels' in self.task_obj:
            self.outcomes = {
                'label': [x.replace('_', ' ') for x in list(self.task_obj['labels'].keys())]
            }
            self.outcomeMap = {k.replace('_', ' '): v for k, v in self.task_obj['labels'].items()} # map from text to integer class label
            self.reverseMap = {v: k for k, v in self.outcomeMap.items()} # map from integer class label to text

        if self.task == 'mmlu' or self.task == 'mmlu1000' or self.task == 'csqa':
            from .MMLU_prompt import (
                FORMAT_FINAL_DECISION,
                FORMAT_DEFAULT,
                FORMAT_JSON,

                EXPLICIT_FEWSHOT_TEMPLATE,
                JSON_FEWSHOT_TEMPLATE,

                CLASSIFICATION_TEMPLATE,
                CORRECTION_CORRECT_TEMPLATE,
                CORRECTION_MISTAKE_TEMPLATE,
                CORRECTION_ARRAY_TEMPLATE,
                CORRECTION_OR_TEMPLATE,
                CORRECTION_REASON_TEMPLATE,
                EXP_PROMPTS
            )

            if self.task == 'mmlu1000':
                dataset = loadMMLU(task=self.subtask, rootDir = '/scratch/fl1092/data_common/MMLU_1000')
            elif self.task == 'mmlu':
                dataset = loadMMLU(task=self.subtask)
            elif self.task == 'csqa':
                dataset = loadCSQA(subtask=self.subtask)
            else:
                print("ERROR loading multiple choice dataset")

        elif self.task == "gsm8k":
            from .GSM8K_prompt import (
                FORMAT_DEFAULT,
                CLASSIFICATION_TEMPLATE,
                EXPLICIT_FEWSHOT_TEMPLATE, # placeholder
                CORRECTION_CORRECT_TEMPLATE,
                CORRECTION_MISTAKE_TEMPLATE,
                CORRECTION_ARRAY_TEMPLATE,
                CORRECTION_OR_TEMPLATE,
            )

            dataset = loadGSM8k(subtask=self.subtask)
            print("First data entry:", dataset['test'][0])
            
            if self.format != 'default':
                print('Setting output format to default')
                self.format = 'default'

        else:
            from .glue_prompt import (
                FORMAT_FINAL_DECISION,
                FORMAT_DEFAULT,
                FORMAT_JSON,

                EXPLICIT_FEWSHOT_TEMPLATE,
                JSON_FEWSHOT_TEMPLATE,

                CLASSIFICATION_TEMPLATE,
                USER_PROMPT_TWO_SENT,
                CORRECTION_CORRECT_TEMPLATE,
                CORRECTION_MISTAKE_TEMPLATE,
                CORRECTION_ARRAY_TEMPLATE,
                CORRECTION_OR_TEMPLATE,
                CORRECTION_REASON_TEMPLATE,

                MRPC_CLASSIFY,
                MRPC_ADDITIONAL_GUIDELINE,
                MNLI_CLASSIFY
            )

            dataset = load_dataset("nyu-mll/glue", task)

        # reindex the dataset because the index of data might not start from 0 #
        self.dataset = {}
        for key, v in dataset.items():
            self.dataset[key] = {}
            for x in v:
                self.dataset[key][x['idx']] = x
        # finishing re-indexing #

        # initializing prompts for classification and fewshot #
        if self.format == 'default':
            self.outcomePrompt = FORMAT_DEFAULT
            self.fewShotTemplate = EXPLICIT_FEWSHOT_TEMPLATE
        elif self.format == 'json':
            self.outcomePrompt = FORMAT_JSON.format(choices=self._formatChoices(self.task_obj['labels'].keys()))
            self.fewShotTemplate = JSON_FEWSHOT_TEMPLATE
        elif self.format == 'explicit':
            self.outcomePrompt = FORMAT_FINAL_DECISION.format(choices=self._formatChoices(self.task_obj['labels'].keys()))
            self.fewShotTemplate = EXPLICIT_FEWSHOT_TEMPLATE
        # initializing prompts end #

        if self.task == 'mmlu' or self.task == 'mmlu1000' or self.task == 'csqa':
            self.fewShotExamples = self._buildFewShotMMLU(split=self.task_obj['few_shot_split'])
            self._parseClassification = self._classificationParser
        elif self.task == 'gsm8k':
            self.fewShotExamples = self._buildFewShotGSM(ids=self.task_obj['few_shot_ids'], split=self.task_obj['few_shot_split'])
            self._parseClassification = self._GSMParser
        else:
            self.fewShotExamples = self._buildFewShotGLUE(self.task_obj['few_shot_ids'], self.task_obj['few_shot_rationale'], self.task_obj['few_shot_split'])
            self._parseClassification = self._classificationParser

        if customizePrompt==False:
            self.classifyPrompt = '\n'.join([CLASSIFICATION_TEMPLATE.format(task=self.task_obj['task_description']), self.outcomePrompt])
        else:
            # using customized prompt
            if self.task == 'mrpc':
                self.classifyPrompt = '\n'.join([MRPC_ADDITIONAL_GUIDELINE, self.outcomePrompt])
            elif self.task == 'mnli':
                self.classifyPrompt = '\n'.join([MNLI_CLASSIFY, self.outcomePrompt])
            else:
                raise ValueError('Customized prompt undefined for task')

        self.corrCorrectPrompt = '\n'.join([CORRECTION_CORRECT_TEMPLATE.format(task=self.task_obj['task_description']), self.outcomePrompt])
        self.corrMistakePrompt = '\n'.join([CORRECTION_MISTAKE_TEMPLATE.format(task=self.task_obj['task_description']), self.outcomePrompt])
        self.corrArrayPrompt = '\n'.join([CORRECTION_ARRAY_TEMPLATE.format(task=self.task_obj['task_description']), self.outcomePrompt])
        self.corrOrPrompt = '\n'.join([CORRECTION_OR_TEMPLATE.format(task=self.task_obj['task_description']), self.outcomePrompt])

        try:
            self.corrReasonPrompt = '\n'.join([CORRECTION_REASON_TEMPLATE.format(
                task=self.task_obj['task_description'], choices=self._formatChoices(self.task_obj['labels'].keys())), self.outcomePrompt])
        except Exception as e:
            print("Reason mode prompt ERROR", str(e))

        try:
            self.experimentPrompt = ['\n'.join([x, 'You will be provided with the input and your previous response. Read them carefully and explain your rationale.', self.outcomePrompt]) for x in EXP_PROMPTS]
        except Exception as e:
            print("Experiment mode prompt ERROR", str(e))
        
        print('Classification prompt:', self.classifyPrompt, end='\n\n')

    def _formatChoices(self, choices):

        choices = [x.replace('_', ' ') for x in choices]

        if len(choices) == 2:
            return ' or '.join(choices)
        elif len(choices) > 2:
            return ', '.join(choices[:-1]) + ', or ' + choices[-1]
        else:
            raise ValueError("Length of choices must be at least 2.")

    def _formatInput(self, idx, split=None):

        if split is None: split=self.split

        if len(self.inputKeys) == 1:
            return self.dataset[split][idx][self.inputKeys[0]]
        elif len(self.inputKeys) == 2:
            return USER_PROMPT_TWO_SENT.format(
                key1= self.inputKeys[0], sent1= self.dataset[split][idx][self.inputKeys[0]],
                key2= self.inputKeys[1], sent2= self.dataset[split][idx][self.inputKeys[1]]
            )
        else:
            raise ValueError('Too many input features')

    def _mostFrequent(self, l):

        m = {}
        maxCount = 0
        for ele in l:
            if ele not in m: m[ele] = 0
            m[ele] += 1
            maxCount = max(maxCount, m[ele])

        res = []
        for k, v in m.items():
            if v == maxCount:
                res.append(k)

        if len(res) != 1:
            print('Warning: ties exist in majority voting')

        print("Voting", m)

        return res[0], round(maxCount/len(l), 3), len(res) != 1 # response, score, and whether a tie exists

    def _classificationParser(self, responses):

        # update self.response_agg to be the parsed result
        # return wether there is a tie or not

        # print('Responses:', responses)

        mapResult = lambda x: self.outcomeMap[x.strip()]

        if self.format == 'explicit':
            responses = [self._cleanResponse(x) for x in responses]
        elif self.format == 'json':
            responses = [json.loads(x)['label'].strip() for x in responses]
        elif self.format == 'default':
            pass
        else:
            raise ValueError('Unrecognized format')

        res = []
        for x in responses:
            try:
                res.append(mapResult(x)) # ensure outcome is one of the potential outcomes
            except Exception as e:
                print("Error is", e, "Response was", x)
                continue

        print(f'{len(res)} out of {len(responses)} has the correct format')
        if len(res) <= len(responses)/2:
            raise ValueError('Less than half of responses has the correct format')

        self.response_agg, self.score, tied = self._mostFrequent(res) # aggregated as a result of majority voting

        print('Classification:', self.response_agg, 'and tie exists' if tied else 'and no tie')

        return tied

    def _GSMParser(self, responses):

        responses = [self._cleanResponse(x, splitOn='####') for x in responses]
        self.response_agg, self.score, tied = self._mostFrequent(responses) # aggregated as a result of majority voting

        print('Classification:', self.response_agg, 'and tie exists' if tied else 'and no tie')

        return tied

    def _buildFewShotMMLU(self, split='dev'):

        examples = []

        for idx, data in self.dataset[split].items():
            if self.task_obj['few_shot_ids'] != []:
                if idx not in self.task_obj['few_shot_ids']:
                    continue
            
            examples.append([{'role':'user', 'content': data['question']}, {'role':'assistant', 'content': data['label']}])

        print(f'{len(examples)} few-shot examples prepared')

        return examples

    def _buildFewShotGLUE(self, ids, reasons, split='train'):

        examples = []

        for seq, idx in enumerate(ids):
            
            text = self._formatInput(idx, split=split)
            response = self.fewShotTemplate.format(
                rationale=self.task_obj['few_shot_rationale'][seq],
                label=self.reverseMap[self.dataset[split][idx]['label']]
            )

            examples.append([{'role':'user', 'content': text}, {'role':'assistant', 'content': response}])
        
        return examples

    def _buildFewShotGSM(self, ids, split='train'):

        examples = []

        for seq, idx in enumerate(ids):
            
            text = self.dataset[split][idx]['question']
            response = self.dataset[split][idx]['answer']

            examples.append([{'role':'user', 'content': text}, {'role':'assistant', 'content': response}])
        
        return examples

    def _initializeResults(self):

        self.results = {}
        for mode in self.allModes:
            fileName = f'{self.outDir}/{mode}.csv'

            if len(glob.glob(fileName)) != 0:
                # if exists load
                print(f'Loading {mode} results ...', flush=True)
                self.results[mode] = pd.read_csv(fileName, index_col=0)
            else:
                # if does not exist, initialize and save
                print(f'Initializing {mode} results ...', flush=True)
                self.results[mode] = pd.DataFrame(columns=['idx', mode, 'label', 'score'])
                self._dumpResult(mode)

    def _addResult(self, idx, mode):

        self.results[mode] = pd.concat(
            [self.results[mode], pd.DataFrame(
                {
                    'idx': idx,
                    'label': self.dataset[self.split][idx]['label'],
                    mode: self.response_agg,
                    'score': self.score
                }, index=[idx])
            ], ignore_index=False
        )
        self._dumpResult(mode)

    def _dumpResult(self, mode):

        self.results[mode].to_csv(f'{self.outDir}/{mode}.csv', index=True)

    def _recordLog(self, idx, mode, prompt, response, trial, classification, success):

        with open(self.logFile, 'a') as f:
            f.write(json.dumps({
                'idx': idx,
                'Mode': mode,
                'Prompt': prompt,
                'Response': response,
                'Classification': classification,
                'Trial': trial,
                'Success': success
            }))
            f.write('\n')

    def _arrayCorrection(self, idx, userPrompt, maxTries=3, verbose=False):

        tries=0
        response=''
        while tries<maxTries:
            try:

                generator = PromptArrayGenerator(
                    self.llm.model,
                    self.llm.tokenizer,
                    eos_token_id=self.llm.terminators,
                )

                choiceText = 'the answer using a single number.' if self.task == 'gsm8k' else f'{self._formatChoices(self.task_obj["labels"].keys())}.'
                sepText = '####' if self.task == 'gsm8k' else 'Therefore, my final decision is: '

                jointPrompt = ''.join([
                    self.corrArrayPrompt, '\n\n',
                    userPrompt.replace('{', ' ').replace('}', ' ').replace('/','-').replace('|',' ').replace('&', '-').replace('~', '-'),
                    '\n\n',
                    f'First explain your rationale and then output your final decision starting with "{sepText}", followed by ' + choiceText
                ])

                response = arrayGenerate(
                    generator,
                    jointPrompt,
                    bad_words=[], verbose=verbose,
                    do_sample=self.llm.do_sample,
                    temperature=self.llm.temperature,
                    k=self.llm.top_k,
                    p=self.llm.top_p
                )

                self._parseClassification(response)
                self._recordLog(idx=idx, prompt=jointPrompt, mode='array', response=response, trial=tries, classification=self.response_agg, success=True)

                return response

            except Exception as e:
                self._recordLog(idx=idx, prompt=jointPrompt, mode='array', response=response, trial=tries, classification=None, success=False)

                tries += 1
                print(f"ERROR array correction on try {tries}:", e)

        if tries==maxTries:
            print('WARNING: Maximum retries reached with promptarray. Did not correct.')
            return []

    def _cleanResponse(self, res, splitOn='my final decision is:'):
        
        res = res.lower().replace('<|eot_id|>','').split(splitOn)[-1].strip(',.; \n\t\v\r\f')
        res = res.split('\n')[0].strip(',.;* \n\t\v\r\f')

        if self.task=='mmlu' and len(res) != 1:
            res = res.split('.')[0].strip()

        return res

    def _getOneResponse(self, classification, splitOn='my final decision is:'):

        for res in self.response:
            
            if self.task == 'gsm8k':
                splitOn = '####'
            else:
                splitOn = 'my final decision is:'

            cleaned = self._cleanResponse(res, splitOn=splitOn)

            if str(classification).strip().lower() == str(cleaned):
                return res

        print('Warning: cannot find the classification in any responses. Returning the first one.')

        return self.response[0]

    def _loadClassificationLog(self, idx):
        print('Loading previous classification from log ...')

        with open(self.logFile, 'r') as f:
            for line in f:
                m = json.loads(line)
                if int(m['idx']) == idx and m['Mode'] == 'classify':
                    self.response = m['Response']
                    self.response_agg = m['Classification']
                    return

        print('ERROR. Cannot load previous classification from log.')

    def classify(self, idx, maxTries=3, **kwargs):

        ### reset classifier ###
        self.response = None
        self.response_agg = ''
        self.score = -1
        self.isCorrect = None
        ### reset classifier ###

        print(f'Classifying {idx} ...')

        if idx in self.results['classify'].idx:
            print(f'skipping {idx} classify; exists')
            return 1

        try:

            self.response = self.promptLLM(
                idx=idx,
                mode='classify',
                systemPrompt=self.classifyPrompt,
                fewShots=self.fewShotExamples,
                userPrompt=self._formatInput(idx),
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )

            self._addResult(idx, mode='classify')

            return 1 # successfully classified

        except ValueError:
            print('ERROR: classification failed')
            self._addResult(idx, mode='classify')

            return 0 # failed

    def correction(self, idx: str, mode: str, maxTries=2, verbose=True, **kwargs):

        if mode not in ['vote', 'array', 'single', 'reason'] and 'exp' not in mode:
            print(f'mode is {mode}. Continuing ...')
            return

        if idx in self.results[mode].idx:
            print(f'skipping {idx} {mode}; exists')
            return

        if self.response is None:
            self._loadClassificationLog(idx)

        prevClassification = self.results['classify'].query(f'idx == {idx}').classify.values[0]

        if prevClassification=='' or pd.isna(prevClassification):
            print(f'skipping {idx} {mode}; was not able to classify')
            return

        prevClassification = str(prevClassification)

        print(f'Mode: {mode} | Correcting {idx} ...', end='\t')
        print(f'Previous class was {prevClassification}', end=' ')
        try:
            print(f'(i.e., {self.reverseMap[prevClassification]})')
        except Exception as e:
            self.reverseMap = {}
            self.reverseMap[prevClassification] = prevClassification

            print(f'(i.e., {self.reverseMap[prevClassification]})')
        
        prevResponse = self._getOneResponse(classification=self.reverseMap[prevClassification])
        userPrompt = f"The input was: {self._formatInput(idx)} \nYour decision was: {self.reverseMap[prevClassification]}\nYour response was: {prevResponse}\n"
        
        if mode=='array':

            responses = self._arrayCorrection(idx=idx, userPrompt=userPrompt, maxTries=maxTries)

        elif mode=='vote':

            responses = []
            try:
                correctResponse = self.promptLLM(
                    idx=idx,
                    mode='correct',
                    systemPrompt=self.corrCorrectPrompt,
                    fewShots=[],
                    userPrompt=userPrompt,
                    maxTries=maxTries,
                    parser=self._parseClassification,
                    **kwargs
                )
                responses.extend(correctResponse)
                self._addResult(idx, 'correct')

                mistakeResponse = self.promptLLM(
                    idx=idx,
                    mode='mistake',
                    systemPrompt=self.corrMistakePrompt,
                    fewShots=[],
                    userPrompt=userPrompt,
                    maxTries=maxTries,
                    parser=self._parseClassification,
                    **kwargs
                )
                responses.extend(mistakeResponse)
                self._addResult(idx, 'mistake')
            except ValueError:
                responses = []

        elif mode=='single':
            
            try:
                responses = self.promptLLM(
                    idx=idx,
                    mode=mode,
                    systemPrompt=self.corrOrPrompt,
                    fewShots=[],
                    userPrompt=userPrompt,
                    maxTries=maxTries,
                    parser=self._parseClassification,
                    **kwargs
                )
            except ValueError:
                responses = []

        elif 'experiment' in mode:
            expIndex = int(mode.split('-')[-1])
            
            try:
                responses = self.promptLLM(
                    idx=idx,
                    mode=mode,
                    systemPrompt=self.experimentPrompt[expIndex],
                    fewShots=[],
                    userPrompt=userPrompt,
                    maxTries=maxTries,
                    parser=self._parseClassification,
                    **kwargs
                )
            except ValueError:
                responses = []

        elif mode=='reason':
            
            try:
                responses = self.promptLLM(
                    idx=idx,
                    mode=mode,
                    systemPrompt=self.corrReasonPrompt,
                    fewShots=[],
                    userPrompt=userPrompt,
                    maxTries=maxTries,
                    parser=self._parseClassification,
                    **kwargs
                )
            except ValueError:
                responses = []

        try:
            tied = self._parseClassification(responses)
            self._addResult(idx, mode)

            if tied is True:
                self.response_agg = prevClassification
                print(f'Tie exists. Did not correct. Class is {self.response_agg}')
            else:
                print(f'Corrected classification is {self.response_agg}')
        except Exception as e:
            print(f"{self.task} {self.split} {idx} {mode} ERROR. Did not correct. Error is {str(e)}")
        

    def promptLLM(self, idx, mode, systemPrompt, fewShots, userPrompt, parser, maxTries=3, **kwargs):

        messages = [{'role':'system', 'content': systemPrompt}]

        for exp in fewShots:
            # TODO: make sure examples do not exceed model token length
            messages.extend(exp)

        messages.append({'role':'user', 'content': userPrompt})

        formatted = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # print(formatted)

        tries = 0
        while tries < maxTries:
            try:
                response = self.llm.prompt(formatted, numSeq=self.numSeq, **kwargs)
                parser(response) # ensure that the outcome is the right format

                self._recordLog(idx=idx, mode=mode, prompt=formatted, response=response, trial=tries, classification=self.response_agg, success=True)
                return response

            except Exception as e:
                self._recordLog(idx=idx, mode=mode, prompt=formatted, response=f'ERROR: {e}', trial=tries, classification=None, success=False)

                tries += 1
                print(f"Tried {tries} times. ERROR", e)
                
        print(f"Tried {maxTries} times and failed. ERROR")
        raise ValueError(f"Tried {maxTries} times and failed")