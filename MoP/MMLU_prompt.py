FORMAT_FINAL_DECISION = '''Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

FORMAT_JSON = '''Finally output a JSON object with the following keys: "rationale" and "label". \
Value of "label" can only be one of the following values: {choices}. Only return a json object and nothing else.'''

FORMAT_DEFAULT = ''''''

CLASSIFICATION_TEMPLATE = '''You are an advanced reasoning agent tasked with {task}. Let's think step by step. 
Firstly, provide an explanation of why some answers are wrong and why a particular answer is correct.
Secondly, identify the correct answer.'''

EXPLICIT_FEWSHOT_TEMPLATE = '''Therefore, my final decision is: {label}'''

JSON_FEWSHOT_TEMPLATE = '''{{"rationale": "Explanation goes here.", "label": "{label}"}}'''

CORRECTION_CORRECT_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of multiple choice answers. \
You were asked to {task}. Let's think step by step.
Given the input and the answer, verify if the answer is correct and explain your rationale.'''

CORRECTION_MISTAKE_TEMPLATE = '''You are an advanced reasoning agent that can identify mistakes in multiple choice answers. \
You were asked to {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong and explain your rationale.'''

CORRECTION_ARRAY_TEMPLATE = '''You are an advanced reasoning agent that can {{\
identify mistakes in multiple choice answers. You were asked to {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong &\
verify the correctness of multiple choice answers. You were asked to {task}. Let's think step by step.
Given the input and the answer, verify if the answer is correct}}.'''

CORRECTION_OR_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of, \
or identify mistakes in, multiple choice answers. You were asked to {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong or correct and explain your rationale.'''

CORRECTION_REASON_TEMPLATE = '''You are an advanced reasoning agent that can assess the consistency of multiple choice answers and identify the \
superior one. You were asked to {task}. Let's think step by step.
Firstly, given an input, generate different rationales for each of the following choices {choices}.
Secondly, reflect on the provided rationales and assess how logically sound they are.
Thirdly, identify the most logical rationale and the corresponding choice.'''

EXP_PROMPTS = [
    '''What’s the problem with the response below?''',
    '''Verify that the response below is correct.''',
    '''Assume that this answer could be either correct or incorrect. Review the answer carefully and report any serious problems you find.''',
    '''Assume that this answer could be either correct or incorrect. Review the answer carefully and report if the answer is correct.''',
    '''Please carefully examine the previous responses to verify their correctness, and provide detailed feedback.''',
    '''Please carefully examine the previous responses to identify any errors, and provide detailed feedback.''',
    '''Do you think the previous response is wrong or not, and if so please point out what is wrong.''',
    '''Do you think the previous response is correct or not, and if so please point out what is correct.''',
    '''Please review and critique your previous response.''', # new
    '''Please review and appraise your previous response.''', # new
    '''Please carefully examine the previous responses, and provide detailed feedback.''', # new
    '''Please carefully examine the previous responses to identify any errors, and provide detailed feedback.''', # new,
]
