import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd


# Data Loading
def load_data(filename):

    filename = [filename]
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename[0],usecols=[0], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename[0],usecols=[2], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename[0],usecols=[1], encoding='ISO-8859-1')
    label = pd.DataFrame.replace(raw_label,['FAVOR','NONE','AGAINST'], [1,2,0])
    concat_text = pd.concat([raw_text, label, raw_target], axis=1)
    concat_text = concat_text[concat_text.Stance != 2]
    
    return(concat_text)


# Data Cleaning
def data_clean(strings, norm_dict):
    
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL, emoji...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i][0] = norm_dict[clean_data[i][0]]
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # split compound hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


# Clean All Data
def clean_all(filename, norm_dict):
    
    concat_text = load_data(filename)
    raw_data = concat_text['Tweet'].values.tolist() 
    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict)
        x_target[i] = data_clean(x_target[i], norm_dict)
    
    return clean_data,label,x_target

