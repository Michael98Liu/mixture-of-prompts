from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast,
    BitsAndBytesConfig, GenerationConfig
)
import transformers
import torch
from peft import PeftModel

from tqdm.notebook import tqdm

import os
import glob
import json
import datetime

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

        def useInt(x): # use integer value if possible
            if int(x) == x: return int(x)
            else: return x
        
        self.model_id = model_id
        self.cache_dir = result_cache_dir.format(model=self.model_id.replace('/', '_'))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
        self._load_basemodel(model_cache_dir)
        print(f'Model {self.model_id} loaded to', self.model.device, flush=True)

        self.do_sample = do_sample
        self.num_beams = useInt(num_beams)
        self.temperature = useInt(temperature)
        self.top_k = useInt(top_k)
        self.top_p = useInt(top_p)

        if self.do_sample==False:
            # using default values
            self.temperature = 1
            self.top_k = 50
            self.top_p = 1
       
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            do_sample=self.do_sample,
            batch_size=4
        )

        
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
            num_return_sequences=numSeq if self.do_sample or self.num_beams>=numSeq else 1,
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