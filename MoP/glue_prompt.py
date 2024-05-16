FORMAT_FINAL_DECISION = '''Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

FORMAT_JSON = '''Finally output a JSON object with the following keys: "rationale" and "label". \
Value of "label" can only be one of the following values: {choices}. Only return a json object and nothing else.'''

FORMAT_DEFAULT = ''''''

CLASSIFICATION_TEMPLATE = '''You are a classifier tasked with classifying {task}. Let's think step by step. 
Firstly, summarize the information contained in the input, given the classification task.
Secondly, classify {task}.'''

USER_PROMPT_TWO_SENT = '''{key1} is "{sent1}"
{key2} is "{sent2}"'''

EXPLICIT_FEWSHOT_TEMPLATE = '''{rationale} Therefore, my final decision is: {label}'''

JSON_FEWSHOT_TEMPLATE = '''{{"rationale": "{rationale}", "label": "{label}"}}'''

CORRECTION_CORRECT_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of classifications. \
You were asked to classify {task}. Let's think step by step.
Given the input and the classification, verify if the classification is correct and explain your rationale.'''

CORRECTION_MISTAKE_TEMPLATE = '''You are an advanced reasoning agent that can identify mistakes in classification. \
You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong and explain your rationale.'''

CORRECTION_ARRAY_TEMPLATE = '''You are an advanced reasoning agent that can {{\
identify mistakes in classification. You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong &\
verify the correctness of classifications. You were asked to classify {task}. Let's think step by step.
Given the input and the a classification, verify if the classification is correct}}.'''

CORRECTION_OR_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of, \
or identify mistakes in, classifications. You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong or correct and explain your rationale.'''

CORRECTION_REASON_TEMPLATE = '''You are an advanced reasoning agent that can assess the consistency of classifications and identify the \
superior one. You were asked to classify {task}. Let's think step by step.
Firstly, given an input, generate different rationales for each of the following classification labels {choices}.
Secondly, reflect on the provided rationales and assess how logically sound they are.
Thirdly, identify the most logical rationale and the corresponding label.'''


LLM_PARSER = '''The following paragraph contains a classification and rationale.
Explain what the rationale is, and then output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''


### the following prompts are generated by LLaMA ###
MRPC_CLASSIFY = '''Classify the semantic equivalence of two sentences. \
Two sentences are considered equivalent if they convey the same meaning, ignoring minor grammatical or lexical differences. \
Please analyze the pair of sentences and label them as 1 (equivalent) or 0 (not equivalent).'''

MRPC_ADDITIONAL_GUIDELINE = '''Classify the semantic equivalence of two sentences. \
Two sentences are considered equivalent if they convey the same meaning, ignoring minor grammatical or lexical differences. \
Please analyze the pair of sentences and label them as 1 (equivalent) or 0 (not equivalent).

**Additional Guidelines:**

1. Ignore minor differences in sentence structure, word order, and punctuation.
2. Focus on the core meaning and content of each sentence.
3. Consider the context in which the sentences appear, but do not rely solely on context to make a decision.
4. Be cautious of sentences that use pronouns or other ambiguous references; instead, try to identify the specific entities or concepts being referred to.
5. When in doubt, err on the side of caution and label the pair as "not equivalent" (0).'''

MNLI_CLASSIFY = '''Classify the relationship between the premise and hypothesis as follows:

* 0 (Entailment): The premise logically supports or implies the hypothesis.
* 1 (Neutral): The premise and hypothesis are unrelated or lack a clear logical connection.
* 2 (Contradiction): The premise contradicts or negates the hypothesis.

Please evaluate the relationship between the provided premise and hypothesis using the above criteria and assign the corresponding label.'''