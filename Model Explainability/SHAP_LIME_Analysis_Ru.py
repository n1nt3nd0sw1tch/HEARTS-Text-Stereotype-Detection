from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

# Select the random sample of observations to use methods on
def sample_observations(file_path, k, seed):
    data = pd.read_csv(file_path)
    
    combinations = data.groupby(['dataset_name', 'categorisation'])
    
    sampled_data = pd.DataFrame(columns=data.columns)
    
    for name, group in combinations:
        same_label = group[group['predicted_label'] == group['actual_label']]
        diff_label = group[group['predicted_label'] != group['actual_label']]
        
        if len(same_label) >= k:
            same_sample = same_label.sample(n=k, random_state=seed)
        else:
            same_sample = same_label
        
        if len(diff_label) >= k:
            diff_sample = diff_label.sample(n=k, random_state=seed)
        else:
            diff_sample = diff_label
        
        sampled_data = pd.concat([sampled_data, same_sample, diff_sample], axis=0)
    
    sampled_data.reset_index(drop=True, inplace=True)
    
    print(sampled_data)
    
    return sampled_data


# Define function to compute SHAP values
def shap_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)

    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    for index, row in sampled_data.iterrows():
        text_input = [row['text']]
        shap_values = explainer(text_input)
        
        print(f"Dataset: {row['dataset_name']} - Categorisation: {row['categorisation']} - Predicted Label: {row['predicted_label']} - Actual Label: {row['actual_label']}")
        label_index = class_names.index("LABEL_1")  
        
        specific_shap_values = shap_values[:, :, label_index].values
        
        tokens = re.findall(r'\w+', row['text'])
        for token, value in zip(tokens, specific_shap_values[0]):
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_shap': value,
                'sentence': row['text'],
                'dataset': row['dataset_name'],
                'categorisation': row['categorisation'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })
                
    return pd.DataFrame(results)


# Define function to compute LIME values 
def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    
    def predict_proba(texts):
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        print("Probabilities shape:", probabilities.shape)
        return probabilities    
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=lambda x: custom_tokenizer(x))  
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
        
        print(f"Dataset: {row['dataset_name']} - Categorisation: {row['categorisation']} - Predicted Label: {row['predicted_label']} - Actual Label: {row['actual_label']}")

        explanation_list = exp.as_list(label=1)
        
        token_value_dict = {token: value for token, value in explanation_list}

        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_lime': value,
                'sentence': text_input,
                'dataset': row['dataset_name'],
                'categorisation': row['categorisation'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })

    return pd.DataFrame(results)

# Define helper functions
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

def compute_cosine_similarity(vector1, vector2):
    v1 = np.asarray(vector1, dtype=float).reshape(1, -1)
    v2 = np.asarray(vector2, dtype=float).reshape(1, -1)
    return cosine_similarity(v1, v2)[0, 0]

def compute_pearson_correlation(vector1, vector2):
    v1 = np.asarray(vector1, dtype=float)
    v2 = np.asarray(vector2, dtype=float)
    if v1.size < 2 or v2.size < 2:
        return np.nan  
    correlation, _ = pearsonr(v1, v2)
    return correlation

def to_probability_distribution(values):
    vals = np.asarray(values, dtype=float)
    min_val = np.min(vals)
    if min_val < 0:
        vals = vals + abs(min_val)
    total = np.sum(vals)
    if total > 0:
        vals = vals / total
    return vals

def compute_js_divergence(vector1, vector2):
    prob1 = to_probability_distribution(vector1)
    prob2 = to_probability_distribution(vector2)
    return jensenshannon(prob1, prob2)