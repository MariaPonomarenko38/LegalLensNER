import datasets
from datasets import load_dataset
import ast
import pandas as pd

CONTEXT = """You are given a text that may contain legal entities and violation indicators. You are also given a list of entity types representing 
legal entities and violation indicators. Your task is to detect and identify 
all instances of the supplied entity types in the user utterance. 
The output must have the same content as the input. Only the tokens that match the entities in the list should be enclosed within XML tags. The XML tag comes from the entities described in the list below. For example, a law should be enclosed within <LAW></LAW> tags. Ensure that all entities are identified. Do not perform false identifications.

List Of Entities
LAW: Specific law or regulation breached.
VIOLATION: Content describing the violation.
VIOLATED BY: Entity committing the violation.
VIOLATED ON: Victim or affected party.

"""

TRAINING_PROMPT = """USER: {input} ASSISTANT: {output}"""

def get_label_list(dataset):
    label_set = set()
    for data in dataset:
        labels = data[
            "ner_tags"
        ]  # Adjust this field name based on your dataset structure
        label_set.update(labels)
    return list(label_set)

def space(token):
    if token not in [',', '.', '?']:
        return " "
    return ""

def transform_to_tagged_string(tokens, tags):
    tagged_string = ""
    current_tag = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tag:
                tagged_string += f" </{current_tag}>"
                current_tag = None
            tagged_string += space(token) + token
        else:
            entity = tag.split('-')[1]
            if current_tag is None:
                tagged_string += f" <{entity}>" + space(token) + token 
                current_tag = entity
            elif current_tag == entity:
                tagged_string += space(token) + token
            else:
                tagged_string += f" </{current_tag}> <{entity}>" + space(token) + token 
                current_tag = entity

    if current_tag:
        tagged_string += f" </{current_tag}>"
    
    return tagged_string.strip()

def transform_to_tagged_string1(tokens, tags):
    tagged_string = ""
    current_tag = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tag:
                current_tag = None
            #tagged_string += space(token) + token
        else:
            entity = tag.split('-')[1]
            if current_tag is None:
                tagged_string += f"; {entity} -" + space(token) + token 
                current_tag = entity
            elif current_tag == entity:
                tagged_string += space(token) + token
            else:
                tagged_string += f"; {entity} -"# + space(token) + token 
                current_tag = entity

    if current_tag:
        tagged_string += f"{current_tag}"
    
    return tagged_string.lstrip("; ")

def prepare_instructions(dataset):
    
    list_tokens = dataset["tokens"]
    list_ner_tags = dataset["ner_tags"]
    instructions = []

    for tokens, ner_tags in zip(list_tokens, list_ner_tags):
        input = ""#CONTEXT
        for token in tokens:
            input += space(token) + token

        input = input.lstrip()

        output = transform_to_tagged_string1(tokens, ner_tags)
        example = TRAINING_PROMPT.format(
            input=input,
            output=output,
        )
        instructions.append(example)
        
    return instructions


def prepare_dataset(dataset_repo):
    dataset = load_dataset(dataset_repo)
    dataset = dataset.map(
        lambda x: {
            "tokens": ast.literal_eval(x["tokens"]),
            "ner_tags": ast.literal_eval(x["ner_tags"]),
        }
    )
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    train_prompt_question = prepare_instructions(train_dataset)
    valid_prompt_question = prepare_instructions(val_dataset)

    train_prompt_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_prompt_question})
    )

    valid_prompt_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": valid_prompt_question})
    )

    return train_prompt_dataset, valid_prompt_dataset

if __name__ == '__main__':
    
    dataset_name = "darrow-ai/LegalLensNER"
    train_prompt_dataset,val_prompt_dataset = prepare_dataset(dataset_name)

    print(len(val_prompt_dataset), len(train_prompt_dataset))