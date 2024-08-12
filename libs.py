import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import itertools
import pandas as pd


from datasets import load_dataset, Dataset
from datasets import DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from sklearn.metrics import confusion_matrix, classification_report


import evaluate
import os
import argparse
from tqdm import tqdm
from tabulate import tabulate

import datetime
import pyperclip

import ast


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # # Debugging lines to understand the structure
    # print(f"Predictions type: {type(predictions)}")
    # print(f"Predictions content: {predictions}")
    
    # print(f"Labels type: {type(labels)}")
    # print(f"Labels content: {labels}")
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Assuming the first element contains the logits
    

    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    try:
        predictions = np.argmax(predictions, axis=1)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Predictions array has inconsistent shapes. Debugging...")
        print(predictions)
        raise e

    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

def compute_top_k_accuracy(preds, labels, k=1):
    
    if isinstance(preds, tuple):
        preds = preds[0]
        
    top_k_preds = np.argsort(preds, axis=1)[:, -k:]
    top_k_accuracy = np.any(top_k_preds == np.expand_dims(labels, axis=1), axis=1).mean()
    return top_k_accuracy

def preprocess_function(examples, tokenizer, max_length=512):
    return tokenizer(examples["Question"], truncation=True, padding = 'max_length', max_length=max_length)






class KC_Chain_Of_Thought_Classification:

    def __init__(self, question, solution, step, model):
        self.question = question
        self.solution = solution
        self.step = step
        self.model = model

    def get_list_kcs(self):
        sys_message = f"""
            You are a professional teacher adhering to the Common Core standards, teaching Mathematics to students from Grade 1 to Grade 6. \
            Your task is to carefully analyze the problem presented, including the question, solution, and steps. Based on this analysis, \
            identify the most suitable Knowledge Components (KCs) from Common Core standards required to solve the problem. \
            You must provide each KC in the exact code format and then explain, step-by-step, why it was chosen. \
            Ensure that the explanation shows how the problem's elements relate to the specific concepts or skills defined by the KC.
            """

        content = f"""
        Analyze the following problem to identify the required Knowledge Components:

        **Question:** "{self.question}"
        **Solution:** "{self.solution}"
        **Steps Taken:** "{self.step}"

        Please provide:
        1. A detailed analysis of the problem.
        2. The identified Knowledge Components
        3. A step-by-step explanation of why each KC is relevant, clearly linking the problem's elements to the KC.
        """



        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": content}
        ]
        pipe = pipeline("text-generation", model= model)
        response = pipe(messages, max_new_tokens=200)  # Limit output to 50 new tokens
        reply = response[0]['generated_text'][2]['content']

        # Extract all unique KCs from the reply
        kcs = re.findall(r'\d+\.\w+\.\w+\.\d+', reply)
        kcs = list(set(kcs))

        return reply, kcs















def iou_evaluation(kc_label, notation_df):
    KC_question = {}
    for i in tqdm(range(len(kc_label))):
    
        ques = kc_label.loc[i, 'Question']
        
        kcs = kc_label.loc[i, 'KCs applied']
        
        try:
            kcs = ast.literal_eval(kcs)
        except:
            kcs = [kcs]
            
        grade_applied = kc_label.loc[i, 'Grade applied']
        
        grade = kc_label.loc[i, 'Grade']
        
        kcs_pred = ast.literal_eval(kc_label.loc[i, 'KCs predicted'])
        
        KC_question[i] = {'Question': ques, 'KCs applied': kcs, 'Grade applied': grade_applied, 'Grade': grade, 'KCs predicted': kcs_pred}
        
    
    kc_df = pd.DataFrame(KC_question)
    kc_df = kc_df.T
    kc_df = kc_df.reset_index(drop=True)
    
    # Change the column name from sub_code to full_code
    for i in range(0, kc_df.shape[0]):
        for j in range(0, len(kc_df.iloc[i,:]['KCs applied'])):
            kc = kc_df.iloc[i,:]['KCs applied'][j]
            if kc in notation_df['Sub Code'].values:
                kc_df.iloc[i,:]['KCs applied'][j] = notation_df[notation_df['Sub Code'] == kc ]['Full Code'].values[0]
                
    kc_df['IOU'] = None
    # Calculate IOU for each question
    for i in range(kc_df.shape[0]):
        kc1 = kc_df.iloc[i]['KCs applied']
        kc2 = kc_df.iloc[i]['KCs predicted']
        
        # Ensure kc1 and kc2 are lists or sets
        kc1 = set(kc1) if isinstance(kc1, (list, set)) else set([kc1])
        kc2 = set(kc2) if isinstance(kc2, (list, set)) else set([kc2])
        
        intersection = len(kc1.intersection(kc2))
        union = len(kc1.union(kc2))
        iou = intersection / union if union != 0 else 0
        iou_formatted = format(iou, '.2f')
        
        kc_df.loc[i, 'IOU'] = float(iou_formatted)
        kc_df['IOU'] = pd.to_numeric(kc_df['IOU'], errors='coerce')
        
    kc_df_sorted = kc_df.sort_values(by='IOU', ascending=False).reset_index(drop=True)
    iou_total_sum = kc_df_sorted['IOU'].sum()
    
    return iou_total_sum, len(kc_df_sorted)
                
    