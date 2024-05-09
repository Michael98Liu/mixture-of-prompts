import os
import glob
import json
import random

from datasets import load_dataset
import pandas as pd

from promptarray.generate import arrayGenerate
from promptarray.generator import PromptArrayGenerator

from .llm import LLM

from .glue_prompt import (
    CLASSIFICATION_TEMPLATE,
    FEWSHOT_TEMPLATE,
    USER_PROMPT_TWO_SENT,
    CORRECTION_CORRECT_TEMPLATE,
    CORRECTION_MISTAKE_TEMPLATE,
    CORRECTION_ARRAY_TEMPLATE,
    CORRECTION_OR_TEMPLATE,
    CORRECTION_REASON_TEMPLATE
)

class LLMClassifier:

    def __init__(self, task, taskJson, split, llm, numSeq=1):

        self.task = task
        self.task_obj = taskJson
        self.split = split
        self.inputKeys = self.task_obj['input_keys']
        self.llm = llm
        assert(self.task_obj['task_id'] == self.task)
        assert(self.split in ['train', 'validation', 'test'])

        self.dataset = load_dataset("nyu-mll/glue", task)
        self.outDir = f'./exp_result/{self.llm.model_id.replace("/", "__")}'
        self.logDir = f'./log/{self.llm.model_id.replace("/", "__")}/'
        self.outFile = f'{self.outDir}/{task}_{self.split}.csv'
        self.logFile = f'{self.logDir}/{task}_{self.split}.jsonl'

        os.makedirs(self.outDir, exist_ok=True)
        os.makedirs(self.logDir, exist_ok=True)

        if len(glob.glob(self.outFile)) != 0:
            # outfile exists
            self.results = pd.read_csv(self.outFile)
        else:
            self.results = pd.DataFrame([], columns=['idx','Classification','correct','mistake','vote','array','single', 'reason'])
        
        self.response = None
        self.response_agg = None # response as a result of majority vote
        self.isCorrect = None
        self.reflections = []
        self.numSeq = numSeq # number of sequences to generate

        self.outcomes = {
            'label': list(self.task_obj['labels'].keys())
        }
        self.outcomeMap = self.task_obj['labels'] # map from text to integer class label
        self.reverseMap = {v: k for k, v in self.outcomeMap.items()} # map from integer class label to text

        self.fewShotExamples = self._buildFewShot(self.task_obj['few_shot_ids'], self.task_obj['few_shot_rationale'], self.task_obj['few_shot_split'])

        # initializing prompts #
        self.classifyPrompt = CLASSIFICATION_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )
        self.corrCorrectPrompt = CORRECTION_CORRECT_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )
        self.corrMistakePrompt = CORRECTION_MISTAKE_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )
        self.corrArrayPrompt = CORRECTION_ARRAY_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )
        self.corrOrPrompt = CORRECTION_OR_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )
        self.corrReasonPrompt = CORRECTION_REASON_TEMPLATE.format(
            task=self.task_obj['task_description'],
            choices=self._formatChoices(self.task_obj['labels'].keys())
        )

        # print( self.classifyPrompt, end='\n\n')
        # print( self.corrReasonPrompt)

    def _formatChoices(self, choices):

        choices = list(choices)

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

        return res[0], len(res) != 1 # response, and whether a tie exists

    def _parseClassification(self, responses):

        # print('Responses:', responses)

        mapResult = lambda x: self.outcomeMap[x.strip()]

        responses = [x.lower().replace('<|eot_id|>','').split('my final decision is:')[-1].strip(',.; \n\t\v\r\f') for x in responses]
        responses = [x.split('\n')[0].strip(',.; \n\t\v\r\f') for x in responses]

        res = []
        for x in responses:
            try:
                res.append(mapResult(x)) # ensure outcome is one of the potential outcomes
            except Exception as e:
                print(e, x)
                continue

        print(f'{len(res)} out of {len(responses)} has the correct format')
        if len(res) <= len(responses)/2:
            raise ValueError('Less than half of responses has the correct format')

        self.response_agg, tied = self._mostFrequent(res) # aggregated as a result of majority voting

        print('Classification:', self.response_agg, 'and tie exists' if tied else 'and no tie')

        return tied
        
    def _buildFewShot(self, ids, reasons, split='train'):

        examples = []

        for seq, idx in enumerate(ids):
            
            text = self._formatInput(idx, split=split)
            response = FEWSHOT_TEMPLATE.format(
                rationale=self.task_obj['few_shot_rationale'][seq],
                label=self.reverseMap[self.dataset[split][idx]['label']]
            )

            examples.append([{'role':'user', 'content': text}, {'role':'assistant', 'content': response}])
        
        return examples

    def _addResult(self, idx):

        self.results = pd.concat(
            [self.results, pd.DataFrame(
                {
                    'idx': idx,
                    'label': self.dataset[self.split][idx]['label'],
                    'Classification': self.response_agg
                }, index=[idx])
            ], ignore_index=False
        )
        self._dumpResult()

    def _addCorrection(self, idx, mode):
        self.results.loc[idx, mode] = self.response_agg
        self._dumpResult()

    def _dumpResult(self):
        self.results.to_csv(self.outFile, index=False)

    def _recordLog(self, prompt, response, trial, classification, success):

        with open(self.logFile, 'a') as f:
            f.write(json.dumps({
                'Prompt': prompt,
                'Response': response,
                'Classification': classification,
                'Trial': trial,
                'Success': success
            }))

    def _arrayCorrection(self, userPrompt, maxTries=3, verbose=True):

        tries=0
        response=''
        while tries<maxTries:
            try:

                generator = PromptArrayGenerator(
                    self.llm.model,
                    self.llm.tokenizer,
                    eos_token_id=self.llm.terminators,
                )
                
                jointPrompt = ''.join([
                    self.corrArrayPrompt, '\n\n',
                    userPrompt.replace('{', ' ').replace('}', ' ').replace('/','-').replace('|',' ').replace('&', '-').replace('~', '-'),
                    '\n\n',
                    f'Briefly explain your rationale briefly and output your final decision starting with "Therefore, my final decision is: ", followed by {self._formatChoices(self.task_obj["labels"].keys())}.'
                ])

                response = arrayGenerate(
                    generator,
                    jointPrompt,
                    bad_words=[], verbose=verbose
                )

                self._parseClassification(response)
                self._recordLog(prompt=jointPrompt, response=response, trial=tries, classification=self.response_agg, success=True)

                return response

            except Exception as e:
                self._recordLog(prompt=jointPrompt, response=response, trial=tries, classification=None, success=False)

                tries += 1
                print(f"ERROR array correction on try {tries}:", e)

        if tries==maxTries:
            print('WARNING: Maximum retries reached with promptarray. Did not correct.')
            return []

    def classify(self, idx, maxTries=3, **kwargs):

        if idx in self.results.idx: return 1

        try:

            self.response = self.promptLLM(
                systemPrompt=self.classifyPrompt,
                fewShots=self.fewShotExamples,
                userPrompt=self._formatInput(idx),
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )

            self._addResult(idx)

            return 1 # successfully classified

        except Exception as e:
            print('ERROR: classification failed', e)

            return 0 # failed

    def correction(self, idx, mode, maxTries=3, verbose=True, **kwargs):

        assert(mode in ['vote', 'array', 'single', 'reason'])

        if not pd.isna(self.results.loc[idx, mode]):
            print(f'skipping {idx} {mode}')
            return

        prevResponse = self.results.query(f'idx == {idx}').Classification.values[0]

        print(f'Mode: {mode} | Correcting {idx} ...', end='\t')
        print(f'Previous class was {prevResponse} (i.e., {self.reverseMap[prevResponse]})')

        userPrompt = f"The input was: {self._formatInput(idx)} \nYour decision was: {self.reverseMap[prevResponse]}\n"
        
        if mode=='array':

            responses = self._arrayCorrection(userPrompt=userPrompt)

        elif mode=='vote':

            responses = []

            correctResponse = self.promptLLM(
                systemPrompt=self.corrCorrectPrompt,
                fewShots=[],
                userPrompt=userPrompt,
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )
            responses.extend(correctResponse)
            self._addCorrection(idx, 'correct')

            mistakeResponse = self.promptLLM(
                systemPrompt=self.corrMistakePrompt,
                fewShots=[],
                userPrompt=userPrompt,
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )
            responses.extend(mistakeResponse)
            self._addCorrection(idx, 'mistake')

        elif mode=='single':

            responses = self.promptLLM(
                systemPrompt=self.corrOrPrompt,
                fewShots=[],
                userPrompt=userPrompt,
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )

        elif mode=='reason':

            responses = self.promptLLM(
                systemPrompt=self.corrReasonPrompt,
                fewShots=[],
                userPrompt=userPrompt,
                maxTries=maxTries,
                parser=self._parseClassification,
                **kwargs
            )

        try:
            tied = self._parseClassification(responses)
            self._addCorrection(idx, mode)

            if tied is True:
                self.response_agg = prevResponse
                print(f'Tie exists. Did not correct. Class is {self.response_agg}')
            else:
                print(f'Corrected classification is {self.response_agg}')
        except Exception as e:
            print(f"{self.task} {self.split} {idx} {mode} ERROR. Did not correct")
        

    def promptLLM(self, systemPrompt, fewShots, userPrompt, parser, maxTries=3, **kwargs):

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

                self._recordLog(prompt=formatted, response=response, trial=tries, classification=self.response_agg, success=True)
                return response

            except Exception as e:
                self._recordLog(prompt=formatted, response=response, trial=tries, classification=None, success=False)

                tries += 1
                print(f"Tried {tries} times. ERROR", e)
                
        print(f"Tried {maxTries} times and failed. ERROR")
        raise ValueError(f"Tried {maxTries} times and failed")


    # def verify(self):
    #     # check whether the classification is correct

    #     if self.label is None:
    #         print('Does not know ground-truth label.')
    #         return None

    #     for outcome, _ in self.outcomes.items():

    #         if self.response_agg[outcome] != self.label[outcome].lower().strip():
    #             self.isCorrect = False
    #             return self.isCorrect

    #         if self.response_agg[outcome] == 'no':
    #             # for multiple binary classifications, only keep comparing if earlier classification is "yes"
    #             break

    #     self.isCorrect = True
    #     return self.isCorrect