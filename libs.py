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



def change_kc_label(kc_label, question_number):
    kc_list = []
    for kc in kc_label[kc_label['Question Number'] == question_number]['KC applied'].values:
        a = kc.split(',')
        a = [x.strip() for x in a]
        kc_list.append(a)
    return kc_list

def change_step(kc_label, question_number):
    step_list = []
    for step in kc_label[kc_label['Question Number'] == question_number]['Step'].values:
        step_list.append(step)
    return step_list




















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
                
    