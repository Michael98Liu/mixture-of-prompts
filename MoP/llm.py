from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast, GemmaTokenizerFast,
    BitsAndBytesConfig, GenerationConfig, set_seed, pipeline
)
set_seed(42)

import google.generativeai as genai
GOOGLE_API_KEY=''
genai.configure(api_key=GOOGLE_API_KEY)

import torch
from peft import PeftModel

from tqdm.notebook import tqdm

import os
import time
import glob
import json
import datetime

class DummyTokenizer():

    def __init__(self):
        pass

    def apply_chat_template(self, msg, *args, **kwargs):
        res = ''
        for m in msg:
            if m['role'] == 'system':
                res += m['content']
            else:
                res += m['role']
                res += ': '
                res += m['content']
            res += '\n\n'
        res += 'assistant: '
        
        return res

class LLM:
    
    def __init__(self, model_id,
                 model_cache_dir='/scratch/fl1092/huggingface/hub/',
                 result_cache_dir='./log/{model}/',
                 do_sample=True,
                 num_beams=1,
                 temperature=1,
                 top_k=50,
                 top_p=1
                ):

        self.do_sample = do_sample

        if self.do_sample==False:
            # using default values
            self.temperature = 1
            self.top_k = 50
            self.top_p = 1
        else:
            self.set_params(num_beams,temperature,top_k,top_p)
        
        self.model_id = model_id
        if 'gemini' in self.model_id:
            self.model = genai.GenerativeModel(self.model_id)
            self.config = genai.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k
            )
            self.tokenizer = DummyTokenizer() # placeholder

            return

        self.cache_dir = result_cache_dir.format(model=self.model_id.replace('/', '_'))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
        self._load_basemodel(model_cache_dir)
        print(f'Model {self.model_id} loaded to', self.model.device, flush=True)
       
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            do_sample=self.do_sample,
            batch_size=4
        )

    def set_params(self, num_beams,temperature,top_k,top_p):

        def useInt(x): # use integer value if possible
            if int(x) == x: return int(x)
            else: return x

        self.num_beams = useInt(num_beams)
        self.temperature = useInt(temperature)
        self.top_k = useInt(top_k)
        self.top_p = useInt(top_p)
        
        
    def _load_basemodel(self, model_cache_dir):

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=model_cache_dir, device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=model_cache_dir, device_map="auto", quantization_config=bnb_config
        )

        if 'Llama-3' in self.model_id:
            self.terminators = GenerationConfig.from_pretrained(
                "chat_templates/generation_configs/", "llama-3-chat.json").stop_token_ids

        elif 'vicuna' in self.model_id:
            chat_template = open('./chat_templates/chat_templates/vicuna.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
            self.terminators = GenerationConfig.from_pretrained(
                "chat_templates/generation_configs/", "vicuna.json").stop_token_ids

        elif 'mistralai' in self.model_id:
            chat_template = open('./chat_templates/chat_templates/mistral-instruct.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
            self.terminators = GenerationConfig.from_pretrained(
                "chat_templates/generation_configs/", "mistral-instruct.json").stop_token_ids

        elif 'phi' in self.model_id:
            chat_template = open('./chat_templates/chat_templates/zephyr.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
            self.terminators = GenerationConfig.from_pretrained(
                "chat_templates/generation_configs/", "zephyr.json").stop_token_ids

        elif 'qwen' in self.model_id:
            chat_template = open('./chat_templates/chat_templates/chatml.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
            self.terminators = GenerationConfig.from_pretrained(
                "chat_templates/generation_configs/", "qwen2-chat.json").stop_token_ids
        
        else:
            self.terminators = None
    
    def _load_finetuned(self, model_cache_dir):

        ### Not using this at the moment ###
        
        base_model = "NousResearch/Llama-2-13b-hf" 

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            base_model, cache_dir=model_cache_dir, device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(
            base_model, cache_dir=model_cache_dir,
            device_map = "auto", load_in_8bit = True)

        self.model = PeftModel.from_pretrained(self.model, self.model_id)
        self.model = self.model.eval()

        
    def prompt(self, messages, numSeq=1, maxNewToken=1000, cache=False):

        # print(maxNewToken, messages)

        # if sample and num_beams is 1 (not beam search), numSeq can be any value
        # else if not sample and num beams is 1 (greedy), numSeq has to be 1
        # else (here, beams is greater than 1) numSeq can be at most num_beams

        if 'gemini' in self.model_id:
            
            try:
                response = self.model.generate_content(
                    messages,
                    generation_config=self.config
                )
            except Exception as e:
                print(e, flush=True)

                if str(e) == '429 Resource has been exhausted (e.g. check quota).':
                    time.sleep(36000)

            time.sleep(20)
            response = [response.text]


        else:

            inputLen = len(self.pipeline.tokenizer(messages))
            if inputLen > 4000:
                print(f'WARNING! Input length {inputLen} exceeds 4000.')
                print(messages[:50], end='\t')
                print('...', end='\t')
                print(messages[-50:])

            sequences = self.pipeline(
                messages,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=numSeq if self.do_sample and self.num_beams==1 else 1 if not self.do_sample and self.num_beams==1 else min(numSeq, self.num_beams),
                eos_token_id=self.terminators,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                max_new_tokens=maxNewToken,
            )

            response = [seq['generated_text'][len(messages):].strip() for seq in sequences] # sequences[0]['generated_text'][-1]['content'] 

        if cache:
            with open(f'{self.cache_dir}{str(datetime.datetime.now())}.json', 'w+') as f:
                json.dump({
                    'Prompt': messages,
                    'Response': response,
                    'MaxNewToken': maxNewToken,
                    'Model': self.model_id
                }, f)

        return response
