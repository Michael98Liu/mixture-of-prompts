# CLASSIFICATION_TEMPLATE = '''You are a classifier tasked with classifying {task}. Let's think step by step. 
# Firstly, summarize the information contained in the input, given the classification task ("Rationale").
# Secondly, classify {task} ("label"): {choices}.
# When you finish, output a json object with the following keys: "label", "Rationale". Only return a json object and nothing else.'''

CLASSIFICATION_TEMPLATE = '''You are a classifier tasked with classifying {task}. Let's think step by step. 
Firstly, summarize the information contained in the input, given the classification task.
Secondly, classify {task}.
Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

USER_PROMPT_TWO_SENT = '''{key1} is "{sent1}"
{key2} is "{sent2}"'''

FEWSHOT_TEMPLATE = '''{rationale} Therefore, my final decision is: {label}'''

CORRECTION_CORRECT_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of classifications. \
You were asked to classify {task}. Let's think step by step.
Given the input and the classification, verify if the classification is correct and explain your rationale.
Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

CORRECTION_MISTAKE_TEMPLATE = '''You are an advanced reasoning agent that can identify mistakes in classification. \
You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong and explain your rationale.
Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

CORRECTION_ARRAY_TEMPLATE = '''You are an advanced reasoning agent that can {{\
identify mistakes in classification. You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong &\
verify the correctness of classifications. You were asked to classify {task}. Let's think step by step.
Given the input and the a classification, verify if the classification is correct}}.
'''

CORRECTION_OR_TEMPLATE = '''You are an advanced reasoning agent that can verify the correctness of, \
or identify mistakes in, classifications. You were asked to classify {task}. Let's think step by step.
Given the input and the classification, check if the classification is wrong or correct and explain your rationale.
Finally output your final decision starting with "Therefore, my final decision is: ", followed by {choices}.'''

CORRECTION_REASON_TEMPLATE = '''You are an advanced reasoning agent that can assess the consistency of classifications and identify the \
superior one. You were asked to classify {task}. Let's think step by step.
Firstly, given an input, generate different rationales for each of the following classification labels {choices}. \
Secondly, reflect on the provided rationales and assess how logically sound they are.
Finally, identify the classification label with the most logical rationale and output your final decision starting with \
"Therefore, my final decision is: ", followed by {choices}.'''