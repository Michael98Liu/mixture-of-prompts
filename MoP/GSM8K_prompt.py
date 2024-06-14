FORMAT_DEFAULT = '''Finally output your final decision starting with "####", followed by the answer to the math problem using a single number.'''

EXPLICIT_FEWSHOT_TEMPLATE = ''''''

CLASSIFICATION_TEMPLATE = '''You are an advanced reasoning agent tasked with solving {task}. Let's think step by step.'''

CORRECTION_CORRECT_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of solutions to math problems. \
You were asked to solve {task}. Let's think step by step.
Given the input and the answer, verify if the answer is correct and explain your rationale.'''

CORRECTION_MISTAKE_TEMPLATE = '''You are an advanced reasoning agent that can identify mistakes in solutions to math problems. \
You were asked to solve {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong and explain your rationale.'''

CORRECTION_ARRAY_TEMPLATE = '''You are an advanced reasoning agent that can {{\
identify mistakes in solutions to math problems. You were asked to solve {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong &\
verify the correctness of solutions to math problems. You were asked to solve {task}. Let's think step by step.
Given the input and the answer, verify if the answer is correct}}.'''

CORRECTION_OR_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of, \
or identify mistakes in, solutions to math problems. You were asked to solve {task}. Let's think step by step.
Given the input and the answer, check if the answer is wrong or correct and explain your rationale.'''
