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

MODES = ['classify', 'correct', 'mistake', 'vote', 'array', 'single', 'reason']

class LLMClassifier:

    def __init__(
        self, task, taskJson, split, llm,
        resFormat='default', logDir='./log', resultDir='./exp_result', numSeq=1,
        customizePrompt=False
    ):

        self.task = task
        self.task_obj = taskJson
        self.split = split
        self.inputKeys = self.task_obj['input_keys']
        self.llm = llm
        self.format = resFormat
        assert(self.task_obj['task_id'] == self.task)
        assert(self.split in ['train', 'validation', 'test'])
        assert(self.format in ['default', 'json', 'explicit'])

        self.dataset = load_dataset("nyu-mll/glue", task)
        self.outDir = f'{resultDir}/{self.llm.model_id.replace("/", "__")}__{llm.do_sample}__{llm.num_beams}__{llm.temperature}__{llm.top_k}__{llm.top_p}/{task}_{self.split}'
        self.logDir = f'{logDir}/{self.llm.model_id.replace("/", "__")}__{llm.do_sample}__{llm.num_beams}__{llm.temperature}__{llm.top_k}__{llm.top_p}'
        self.logFile = f'{self.logDir}/{task}_{self.split}.jsonl'

        os.makedirs(self.outDir, exist_ok=True)
        os.makedirs(self.logDir, exist_ok=True)

        if len(glob.glob(f'{self.outDir}/*.csv')) != 0:
            # outfile exists
            self._loadResults()
        else:
            self._initializeResults()
        
        self.response = None # classification responses
        self.response_agg = '' # response as a result of majority vote
        self.score = -1 # confidence score associated with response_agg
        self.isCorrect = None
        self.reflections = []
        self.numSeq = numSeq # number of sequences to generate

        self.outcomes = {
            'label': [x.replace('_', ' ') for x in list(self.task_obj['labels'].keys())]
        }
        self.outcomeMap = {k.replace('_', ' '): v for k, v in self.task_obj['labels'].items()} # map from text to integer class label
        self.reverseMap = {v: k for k, v in self.outcomeMap.items()} # map from integer class label to text

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

        self.fewShotExamples = self._buildFewShot(self.task_obj['few_shot_ids'], self.task_obj['few_shot_rationale'], self.task_obj['few_shot_split'])

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
        self.corrReasonPrompt = '\n'.join([CORRECTION_REASON_TEMPLATE.format(
            task=self.task_obj['task_description'], choices=self._formatChoices(self.task_obj['labels'].keys())), self.outcomePrompt])


        print('Classification prompt:', self.classifyPrompt, end='\n\n')
        # print( self.corrReasonPrompt)

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

    def _parseClassification(self, responses):

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
        
    def _buildFewShot(self, ids, reasons, split='train'):

        examples = []

        for seq, idx in enumerate(ids):
            
            text = self._formatInput(idx, split=split)
            response = self.fewShotTemplate.format(
                rationale=self.task_obj['few_shot_rationale'][seq],
                label=self.reverseMap[self.dataset[split][idx]['label']]
            )

            examples.append([{'role':'user', 'content': text}, {'role':'assistant', 'content': response}])
        
        return examples

    def _loadResults(self):

        self.results = {}
        for mode in MODES:
            self.results[mode] = pd.read_csv(f'{self.outDir}/{mode}.csv', index_col=0)

    def _initializeResults(self):

        self.results = {}
        for mode in MODES:
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
                
                jointPrompt = ''.join([
                    self.corrArrayPrompt, '\n\n',
                    userPrompt.replace('{', ' ').replace('}', ' ').replace('/','-').replace('|',' ').replace('&', '-').replace('~', '-'),
                    '\n\n',
                    f'First explain your rationale and then output your final decision starting with "Therefore, my final decision is: ", followed by {self._formatChoices(self.task_obj["labels"].keys())}.'
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

    def _cleanResponse(self, res):
        res = res.lower().replace('<|eot_id|>','').split('my final decision is:')[-1].strip(',.; \n\t\v\r\f')
        res = res.split('\n')[0].strip(',.; \n\t\v\r\f')
        return res

    def _getOneResponse(self, classification):

        for res in self.response:

            cleaned = self._cleanResponse(res)

            if classification.strip().lower() == cleaned:
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

        print(f'Classifying {idx} ...')

        if idx in self.results['classify'].idx: return 1

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

        assert(mode in ['vote', 'array', 'single', 'reason'])

        if idx in self.results[mode].index and not pd.isna(self.results[mode].loc[idx, mode]):
            print(f'skipping {idx} {mode}; exists')
            return

        if self.response is None:
            self._loadClassificationLog(idx)

        prevClassification = self.results['classify'].query(f'idx == {idx}').classify.values[0]

        if prevClassification=='' or pd.isna(prevClassification):
            print(f'skipping {idx} {mode}; was not able to classify')
            return

        print(f'Mode: {mode} | Correcting {idx} ...', end='\t')
        print(f'Previous class was {prevClassification} (i.e., {self.reverseMap[prevClassification]})')
        
        prevResponse = self._getOneResponse(classification=self.reverseMap[prevClassification])
        userPrompt = f"The input was: {self._formatInput(idx)} \nYour decision was: {self.reverseMap[prevClassification]}\nYour response was: {prevResponse}\n"
        
        if mode=='array':

            responses = self._arrayCorrection(idx=idx, userPrompt=userPrompt)

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
            print(f"{self.task} {self.split} {idx} {mode} ERROR. Did not correct")
        

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
                self._recordLog(idx=idx, mode=mode, prompt=formatted, response=response, trial=tries, classification=None, success=False)

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