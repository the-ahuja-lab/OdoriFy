#!/usr/bin/env python
# coding: utf-8

# 
# Given smile -> Predict top k sequences
# 
# 1). Make one hot representation of input smile
# 
# 2). And make one-hot representations of all the unique smiles in db
# 
# 3). Now using Tatinomo coefficient check the similarity with each smile and set a threshold value and take only those smiles.
# 
# 4). Now only take those sequences that are there with these smiles in ground truth.
# 
# 5). Apply M4 with the sequences, find the top k predictions.
# 
# 6). With each sequence-smile pair a smmile interpretibility needed.
# 

# In[2]:

import io
import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import random
import pickle
from captum.attr import IntegratedGradients
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import sys
import configparser

if(len (sys.argv) != 2):
    print("Usage: ", sys.argv[0], "config.cfg")
    sys.exit(0)

config = configparser.ConfigParser()
config.read(sys.argv[1])

def getConfig(section, attribute, default=""):
    try:
        return config[section][attribute]
    except:
        return default

TRAIN_DATA_FILE= getConfig("Task","train_data_file")
apply_data_file= getConfig("Task","apply_data_file")
result_file= getConfig("Task","result_file")
smile_l=int(getConfig("Task","smile_length","75"))
seq_l=int(getConfig("Task","sequence_length","315"))
filename=getConfig("Task","filename")
model_filename = getConfig("Task","model_file")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )


class BLSTM(nn.Module):
    def __init__(self, input_smile_dim, hidden_smile_dim, layer_smile_dim, input_seq_dim, hidden_seq_dim, layer_seq_dim, output_dim):
        super(BLSTM, self).__init__()
        self.hidden_smile_dim = hidden_smile_dim
        self.layer_smile_dim = layer_smile_dim
        self.hidden_seq_dim = hidden_seq_dim
        self.layer_seq_dim = layer_seq_dim
        self.output_dim = output_dim
        self.smile_len = smile_l
        self.seq_len = seq_l
        self.num_smile_dir=2
        self.num_seq_dir=2
        
        self.lstm_smile = nn.LSTM(input_smile_dim, hidden_smile_dim, layer_smile_dim,bidirectional=True)
        self.lstm_seq = nn.LSTM(input_seq_dim, hidden_seq_dim, layer_seq_dim,bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        
        self.fc_seq= nn.Linear(self.seq_len*hidden_seq_dim*self.num_seq_dir,smile_o)
        self.fc_smile= nn.Linear(self.smile_len*hidden_smile_dim*self.num_smile_dir,seq_o)
        self.batch_norm_combined = nn.BatchNorm1d(smile_o+seq_o, affine = False)
        # self.fc_combined = nn.Sequential(nn.Linear(1000,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,10),nn.ReLU(),nn.Linear(10,output_dim))
        # self.fc_combined = nn.Sequential(nn.Linear(smile_o+seq_o,100),nn.ReLU(),nn.BatchNorm1d(100, affine = False),nn.Dropout(.5),nn.Linear(100,10),nn.ReLU(),nn.Linear(10,output_dim))
        # self.fc_combined = nn.Sequential(nn.Linear(smile_o+seq_o,10),nn.ReLU(),nn.Linear(10,output_dim))
        self.fc_combined = nn.Sequential(nn.Linear(smile_o+seq_o,100),nn.ReLU(),nn.Linear(100,10),nn.ReLU(),nn.Linear(10,output_dim))
        
    def forward(self, x1,x2):
        h0_smile = torch.zeros(self.layer_smile_dim*self.num_smile_dir, x1.size(1), self.hidden_smile_dim).requires_grad_()
        c0_smile = torch.zeros(self.layer_smile_dim*self.num_smile_dir, x1.size(1), self.hidden_smile_dim).requires_grad_()
        h0_seq = torch.zeros(self.layer_seq_dim*self.num_seq_dir, x2.size(1), self.hidden_seq_dim).requires_grad_()
        c0_seq = torch.zeros(self.layer_seq_dim*self.num_seq_dir, x2.size(1), self.hidden_seq_dim).requires_grad_()
 
        h0_smile=h0_smile.to(device)
        c0_smile=c0_smile.to(device)
        h0_seq=h0_seq.to(device)
        c0_seq=c0_seq.to(device)
 
        out_smile, (hn_smile, cn_smile) = self.lstm_smile(x1, (h0_smile, c0_smile))
        out_seq, (hn_seq, cn_seq) = self.lstm_seq(x2, (h0_seq, c0_seq))
        out_smile = self.dropout(out_smile)
        out_seq = self.dropout(out_seq)
        out_seq=self.fc_seq(out_seq.view(-1,self.seq_len*self.hidden_seq_dim*self.num_seq_dir))
        out_seq = self.dropout(out_seq)
        out_smile=self.fc_smile(out_smile.view(-1,self.smile_len*self.hidden_smile_dim*self.num_smile_dir))
        out_smile = self.dropout(out_smile)
 
        out_combined=torch.cat((out_smile,out_seq), dim=1)
        out_combined = self.batch_norm_combined(out_combined)
        out_combined=self.fc_combined(out_combined)
 
        prob=nn.Softmax(dim=1)(out_combined)
        pred=nn.LogSoftmax(dim=1)(out_combined)
        return pred

# In[3]:


def one_hot_smile(smile):
    key="()+–./-0123456789=#@$ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]abcdefghijklmnopqrstuvwxyz^"
    test_list=list(key)
    res = {val : idx  for idx, val in enumerate(test_list)}
    threshold=smile_l

    if len(smile)<=threshold:
        smile=smile+("^"*(threshold-len(smile)))
    else:
        smile=smile[0:threshold]
    array=[[0 for j in range(len(key))] for i in range(threshold)]
    for i in range(len(smile)):
        array[i][res[smile[i]]]=1
    array=torch.Tensor(array)

    return array

def one_hot_seq(seq):
    key="ABCDEFGHIJKLMNOPQRSTUVWXYZ^"
    seq=seq.upper()
    test_list=list(key)
    res = {val : idx  for idx, val in enumerate(test_list)}
    threshold=seq_l

    if len(seq)<=threshold:
        seq=seq+("^"*(threshold-len(seq)))
    else:
        seq=seq[0:threshold]
    array=[[0 for j in range(len(key))] for i in range(threshold)]
    for i in range(len(seq)):
      array[i][res[seq[i]]]=1
    array=torch.Tensor(array)

    return array


def prediction(model, x_input_smile, x_input_seq):
    x_user_smile=one_hot_smile(x_input_smile)
    x_user_smile=list(x_user_smile)
    x_user_smile=torch.stack(x_user_smile)
    x_user_smile=x_user_smile.view(1,smile_l,77)

    x_user_seq=one_hot_seq(x_input_seq)
    x_user_seq=list(x_user_seq)
    x_user_seq=torch.stack(x_user_seq)
    x_user_seq=x_user_seq.view(1,seq_l,27)
    model.eval()
    scores = model(x_user_smile.to(device),x_user_seq.to(device))
    _, predictions = scores.max(1)

    prob=torch.exp(scores)
    prob=prob.tolist()    
    return float(str(prob[0][predictions.item()])[:5]), predictions.item()


# In[18]:

def combined_user_predict(model, x_input_smile, x_input_seq, count,filename):
    mol = Chem.MolFromSmiles(x_input_smile)
    Chem.Kekulize(mol)
    x_input_smile=Chem.MolToSmiles(mol, kekuleSmiles=True)
    ax=plt.figure()
    x_user_smile=one_hot_smile(x_input_smile)
    x_user_smile=list(x_user_smile)
    x_user_smile=torch.stack(x_user_smile)
    x_user_smile=x_user_smile.view(1,smile_l,77)

    x_user_seq=one_hot_seq(x_input_seq)
    x_user_seq=list(x_user_seq)
    x_user_seq=torch.stack(x_user_seq)
    x_user_seq=x_user_seq.view(1,seq_l,27)
    model.eval()
    torch.backends.cudnn.enabled=False
    ig = IntegratedGradients(model)
    baseline = torch.zeros(1, smile_l, 77)
    for i in baseline[0]:
        i[-1]=1

    attr,delta= ig.attribute((x_user_smile.to(device),x_user_seq.to(device)), target=1,return_convergence_delta=True)
    attr=attr[0].view(smile_l,77)
    maxattr,_=torch.max(attr,dim=1)
    minattr,_=torch.min(attr,dim=1)
    relevance=maxattr+minattr
    relevance=relevance.cpu().detach().numpy()
    data_relevance=pd.DataFrame()
    data_relevance["values"]=relevance

    len_smile=min(len(x_input_smile), smile_l)
    cropped_smile_relevance=data_relevance.iloc[0:len_smile]
    x_smile_labels=pd.Series(list(x_input_smile[:len_smile]))
    cropped_smile_relevance['smile_char']=x_smile_labels
    impacts=[]
    
    
    cropped_smile_relevance['positive']=['']*len_smile
    cropped_smile_relevance['negative']=['']*len_smile
    for row in range(len_smile):
        if (ord(cropped_smile_relevance['smile_char'][row])<65 or ord(cropped_smile_relevance['smile_char'][row])>90):
            cropped_smile_relevance['values'][row]=0
            cropped_smile_relevance['positive'][row]=0
            cropped_smile_relevance['negative'][row]=0
        else:
            if(cropped_smile_relevance['values'][row]>0):
                cropped_smile_relevance['positive'][row]=cropped_smile_relevance['values'][row]
                cropped_smile_relevance['negative'][row]=0
            elif(cropped_smile_relevance['values'][row]<0):
                cropped_smile_relevance['negative'][row]=cropped_smile_relevance['values'][row]
                cropped_smile_relevance['positive'][row]=0
            else:
                cropped_smile_relevance['positive'][row]=0
                cropped_smile_relevance['negative'][row]=0
             
            impacts.append(cropped_smile_relevance['values'][row])
    
    impacts=np.array(impacts)
    ax=cropped_smile_relevance.plot( y=["positive", "negative"], color=['green', 'red'], kind="bar", figsize=(25,15))
    ax.legend(['Contribution to Binding', 'Contribution to Non-Binding'],prop={'size': 16})
    ax.set_xticklabels(cropped_smile_relevance['smile_char'],fontsize=15,rotation=0)
    ax.set_xlabel("SMILES", fontsize=15)
    ax.set_ylabel("Relevance", fontsize=15)
    ax.figure.savefig(f"{filename}_{count}_SmileInterpretability.pdf")
    #ax.close()

    mol=x_input_smile
    m = Chem.MolFromSmiles(mol)
    num_atoms = m.GetNumAtoms()
    labels = [ m.GetAtomWithIdx(i).GetSymbol().upper() for i in range(num_atoms) ]
    colors = {}
    i=0
    k=0
    y_max = np.max(impacts)
    y_min = np.min(impacts)
    dist = y_max - y_min
    while i < len(mol):
        c = mol[i]
        n = ""
        if c.upper() not in "CBONSPFIK":
            print(mol[i], 0.0, "0xFFFFFF")
        else:       
            if i + 1 < len(mol):
                n = mol[i+1]
            sym = c + n    
            sym = sym.strip()
            com = sym.upper()
            if com == "BR" or com == "CL" or com == "NA":
                i = i + 1
            else:
                com = c.upper()
                sym = c
            if com == labels[k]:
                color = "0xBBBBBB"
                triple = [0, 0 ,0]
                if impacts[k] > 0.0:
                    y = int(math.floor(255.0 - 155.0 * impacts[k]  / y_max))
                    color = "0x00" + hex(y)[-2:] + "00"
                    triple[1] = y /255.0
                if impacts[k] < 0.0:
                    y = int(math.floor(255.0 - 155.0 * impacts[k]  / y_min))
                    color = "0x" + hex(y)[-2:] + "0000"
                    triple[0] = y / 255.0
                colors[k]= tuple(triple)
                print(sym, impacts[k], color)
                k = k + 1   
        i = i + 1
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)

    drawer.DrawMolecule(m,highlightAtoms = [i for i in range(num_atoms)], highlightBonds=[], highlightAtomColors = colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')

    fp = open(f"{filename}_{count}_mol.svg", "w")
    print(svg, file=fp)
    fp.close()
    
    #sequence Interpretability:
    ax=plt.figure()
    baseline = torch.zeros(2, seq_l, 27)
    ig = IntegratedGradients(model)
    attr,delta= ig.attribute((x_user_smile.to(device),x_user_seq.to(device)), target=1,return_convergence_delta=True)
    smile_attr=attr[0].view(smile_l,77)
    seq_attr=attr[1].view(seq_l,27)
    maxattr,_=torch.max(seq_attr,dim=1)
    minattr,_=torch.min(seq_attr,dim=1)
    relevance=maxattr+minattr
    relevance=relevance.cpu().detach().numpy()
    data_relevance=pd.DataFrame()
    data_relevance["values"]=relevance

    len_seq=min(len(x_input_seq), seq_l)
    cropped_seq_relevance=data_relevance.iloc[0:len_seq]
    x_seq_labels=pd.Series(list(x_input_seq))
    cropped_seq_relevance['seq_char']=x_seq_labels
    cropped_seq_relevance['positive']=['']*len_seq
    cropped_seq_relevance['negative']=['']*len_seq
    
    for row in range(len_seq):
        if (ord(cropped_seq_relevance['seq_char'][row])<65 or ord(cropped_seq_relevance['seq_char'][row])>90):
            cropped_seq_relevance['values'][row]=0
            cropped_smile_relevance['positive'][row]=0
            cropped_smile_relevance['negative'][row]=0
        else:
            if(cropped_seq_relevance['values'][row]>0):
                cropped_seq_relevance['positive'][row]=cropped_seq_relevance['values'][row]
                cropped_seq_relevance['negative'][row]=0
            else:
                cropped_seq_relevance['negative'][row]=cropped_seq_relevance['values'][row]
                cropped_seq_relevance['positive'][row]=0
             
            
    ax=cropped_seq_relevance.plot( y=["positive", "negative"], color=['green', 'red'], kind="barh", figsize=(20, 70) )
    ax.legend(['Contribution to Binding', 'Contribution to Non-Binding'],prop={'size': 16})
    ax.set_yticklabels(cropped_seq_relevance['seq_char'],fontsize=12,rotation=0)
    ax.set_ylabel("Receptor Sequence",fontsize=15)
    ax.set_xlabel("Relevance",fontsize=15,rotation=0)
    ax.figure.savefig(f'{filename}_{count}_SequenceInterpretability.pdf')
    #ax.close()
    
    


# Read input file
df = pd.read_csv(TRAIN_DATA_FILE)
unique_smiles=df["SMILES"].unique().tolist()

f = pd.read_csv(apply_data_file)
threshold=f["threshhold"][0]
input_smile=f["smiles"][0]

mol_smiles=[Chem.MolFromSmiles(smile) for smile in unique_smiles]
fps = [Chem.RDKFingerprint(x) for x in mol_smiles]

input_mol_smile = Chem.MolFromSmiles(input_smile)
input_fps = Chem.RDKFingerprint(input_mol_smile)

similar_smiles = []
for i in range(len(fps)):
    sim = DataStructs.FingerprintSimilarity(fps[i],input_fps)
    if(sim > threshold):
        similar_smiles.append(unique_smiles[i])

# Finding seqs with Activation status with similar_smiles= 1 

# In[11]:


similar_seqs={}
for i in range(len(similar_smiles)):
    for j in range(len(df)):
        if( df['SMILES'][j]==similar_smiles[i] and df['Activation_Status'][j]==1 and str(df['Mutations'][j])=='nan'):
            similar_seqs[ df['Final_Sequence'][j] ]=1


# Run M4 on these smile, seq pair (find top-k sequences, and interpretability of those top-k)

# In[12]:

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

f=open(model_filename, 'rb')
loaded_model = CPU_Unpickler(f).load()
# loaded_model=pickle.load(f)


# meta-dataframe to match receptors and Receptor Sequence
databasedf=df[['Receptor','Final_Sequence']]
databasedf= databasedf.drop_duplicates()


k=0
df_top_seqs=pd.DataFrame(columns=['Final_Sequence', 'Probability'])
for seq in similar_seqs:
    prob,pred=prediction(loaded_model, input_smile, seq )
    if(pred==1):
        df_top_seqs.loc[k]=[seq,prob]
        k+=1


df_top_seqs=df_top_seqs.sort_values("Probability", ascending=False)



df_top_seqs=pd.merge(df_top_seqs, databasedf, on='Final_Sequence')


for i in range(len(df_top_seqs)):
    count=str(i+1)
    combined_user_predict(loaded_model, input_smile, df_top_seqs['Final_Sequence'][i] , count,filename)

if(len(df_top_seqs)==0):
        df_top_seqs.loc[0]=['Empty','Empty','Empty']

df_top_seqs.to_csv(result_file, index=False)





