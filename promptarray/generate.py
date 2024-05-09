# coding=utf-8
# Copyright (c) 2021 Jeffrey M. Binder.  All rights reserved.

from .generator import PromptArrayGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import argparse

def arrayGenerate(
    generator, prompt_text, length = 256,
    do_sample = True,
    temperature = 0.6,
    k = 5,
    p = 0.5,
    repetition_penalty = 1.5,
    bad_words = ["the"],
    num_return_sequences = 1,
    overlap_factor = 0.25,
    verbose=False
):

    import time
    start_time = time.time()
    outputs = generator(
        prompt=prompt_text,
        num_return_sequences=num_return_sequences,
        max_length=length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        bad_words=bad_words,
        overlap_factor=overlap_factor,
        verbose=verbose
    )

    if verbose:
        print(f"Time: {time.time() - start_time}s")

        for i, output in enumerate(outputs):
            if num_return_sequences > 1:
                print(f'Generated sequence {i}:')
            print(output)

    return outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generating')

    parser.add_argument("--model_cache_dir", type=str, default='/scratch/fl1092/huggingface/hub/')
    parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--prompt_text", type=str, default='Scientists recently discovered a new species of {serpent~snake}. Here is a description of it:')

    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_cache_dir, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.model_cache_dir, device_map="auto")
    model.eval()

    if 'Llama-3' in args.model_name:
        terminators = GenerationConfig.from_pretrained(
            "chat_templates/generation_configs/", "llama-3-chat.json").stop_token_ids
    else:
        raise ValueError('Undefined EOS tokens')

    # Initialize PromptArray
    generator = PromptArrayGenerator(
        model,
        tokenizer,
        eos_token_id=terminators
    )

    # generate
    arrayGenerate(generator, args.prompt_text, verbose=True)