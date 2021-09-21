import random
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,cohen_kappa_score,confusion_matrix,roc_curve,balanced_accuracy_score
import sys
import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle
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
smile_l=int(getConfig("Task","smile_length","75"))
seq_l=int(getConfig("Task","sequence_length","315"))
smile_h=int(getConfig("Task","smile_hidden","50")) #smile_hidden
seq_h=int(getConfig("Task","sequence_hidden","50")) #hidden_seq
smile_o=int(getConfig("Task","smile_o","50"))
seq_o=int(getConfig("Task","seq_o","50"))
bt=int(getConfig("Task","batch_size","32")) #batch
num_epochs=int(getConfig("Task","num_epochs","50")) #epoch
learning_rate= float(getConfig("Task","learning_rate", "0.0001"))
filename=getConfig("Task","filename")

df=pd.read_csv(TRAIN_DATA_FILE)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )

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
		return pred,prob

def check_accuracy_per_epoch(model,x_smile,x_seq,y_model, epoch_wise, epochno,  batch_size=bt):
	num_correct = 0
	num_samples = 0
	model.eval()
	count=1
	num_pred=[]
	num_prob=[]

	with torch.no_grad():
		for beg_i in range(0, x_smile.shape[0], batch_size):
			x_smile_batch = x_smile[beg_i:beg_i + batch_size]
			x_seq_batch = x_seq[beg_i:beg_i + batch_size]
			y_batch = y_model[beg_i:beg_i + batch_size]
			x_smile_batch=torch.FloatTensor(x_smile_batch)
			x_seq_batch=torch.FloatTensor(x_seq_batch)
			y_batch=torch.LongTensor(list(y_batch))
			x_smile_batch = Variable(x_smile_batch)
			x_seq_batch = Variable(x_seq_batch)
			y_batch = Variable(y_batch)

			x_smile_batch=x_smile_batch.to(device)
			x_seq_batch=x_seq_batch.to(device)
			y_batch=y_batch.to(device)

			scores = model(x_smile_batch,x_seq_batch)
			_, predictions = scores[0].max(1)
			probs = [i[1].item() for i in scores[1]]
			num_pred.extend(predictions.tolist())
			num_prob.extend(probs)
			num_correct += (predictions == y_batch).sum()
			num_samples += predictions.size(0)
			count+=1
		
		y_model=torch.Tensor.cpu(y_model)
		accuracy=accuracy_score(np.array(y_model),np.array(num_pred))
		precision=precision_score(np.array(y_model),np.array(num_pred))
		recall=recall_score(np.array(y_model),np.array(num_pred))
		f1=f1_score(np.array(y_model),np.array(num_pred))
		roc=roc_auc_score(np.array(y_model),np.array(num_pred))
		kappa=cohen_kappa_score(np.array(y_model),np.array(num_pred))
		conf_matrix=confusion_matrix(np.array(y_model),np.array(num_pred))
		bal_acc=balanced_accuracy_score(np.array(y_model),np.array(num_pred))
		epoch_wise[epochno]=[accuracy,precision,recall, kappa, bal_acc, f1, roc, conf_matrix]

				
def train_epoch(model,x_train_smile,x_train_seq,y_train,x_test_smile,x_test_seq,y_test,train_epochwise,test_epochwise, epochno, batch_size=bt):
    model.train()
    loss_train_array = []
    for beg_i in range(0, x_train_smile.shape[0], batch_size):
        x_train_smile_batch = x_train_smile[beg_i:beg_i + batch_size]
        x_train_seq_batch = x_train_seq[beg_i:beg_i + batch_size]
        y_train_batch = y_train[beg_i:beg_i + batch_size]
        x_train_smile_batch=list(x_train_smile_batch)
        x_train_seq_batch=list(x_train_seq_batch)
        x_train_smile_batch=torch.stack(x_train_smile_batch)
        x_train_smile_batch=torch.FloatTensor(x_train_smile_batch)
        x_train_seq_batch=torch.stack(x_train_seq_batch)
        x_train_seq_batch=torch.FloatTensor(x_train_seq_batch)
        y_train_batch=torch.LongTensor(list(y_train_batch))
        
        x_train_smile_batch = Variable(x_train_smile_batch)
        x_train_seq_batch = Variable(x_train_seq_batch)
        y_train_batch = Variable(y_train_batch)
 
        x_train_smile_batch=x_train_smile_batch.to(device)
        x_train_seq_batch=x_train_seq_batch.to(device)
        y_train_batch=y_train_batch.to(device)
 
        optimizer.zero_grad()
        # (1) Forward
        y_comb_train = model(x_train_smile_batch,x_train_seq_batch)
        y_hat_train=y_comb_train[0]
        # y_hat_train= model(x_train_smile_batch,x_train_seq_batch)
        # (2) Compute diff
        loss_train = criterion(y_hat_train, y_train_batch)
        # (3) Compute gradients
        loss_train.backward()
        # (4) update weights
        optimizer.step()        
        loss_train_array.append(torch.Tensor.cpu(loss_train).data.numpy())
    
    check_accuracy_per_epoch(model,x_train_smile,x_train_seq,y_train, train_epochwise, epochno,  batch_size=bt)
    
    loss_test=0
    loss_test_array=[]
 
    for beg_i in range(0, x_test_smile.shape[0], batch_size):
 
        x_test_smile_batch = x_test_smile[beg_i:beg_i + batch_size]
        x_test_seq_batch = x_test_seq[beg_i:beg_i + batch_size]
        y_test_batch = y_test[beg_i:beg_i + batch_size]
        x_test_smile_batch=list(x_test_smile_batch)
        x_test_smile_batch=torch.stack(x_test_smile_batch)
        x_test_smile_batch=torch.FloatTensor(x_test_smile_batch)
        x_test_seq_batch=list(x_test_seq_batch)
        x_test_seq_batch=torch.stack(x_test_seq_batch)
        x_test_seq_batch=torch.FloatTensor(x_test_seq_batch)
        y_test_batch=torch.LongTensor(list(y_test_batch))
        x_test_smile_batch = Variable(x_test_smile_batch)
        x_test_seq_batch = Variable(x_test_seq_batch)
        y_test_batch = Variable(y_test_batch)
 
        x_test_smile_batch=x_test_smile_batch.to(device)
        x_test_seq_batch=x_test_seq_batch.to(device)
        y_test_batch=y_test_batch.to(device)
 
        # (1) Forward
        y_comb_test = model(x_test_smile_batch,x_test_seq_batch)
        y_hat_test= y_comb_test[0]
        # y_hat_test= model(x_test_smile_batch,x_test_seq_batch)
        # (2) Compute diff
        loss_test = criterion(y_hat_test, y_test_batch)
        loss_test_array.append(loss_test.to(device))

    val_loss=sum(loss_test_array)/len(loss_test_array)
    train_loss=sum(loss_train_array)/len(loss_train_array)
 
    check_accuracy_per_epoch(model,x_test_smile,x_test_seq,y_test, test_epochwise, epochno,  batch_size=bt)


    print("Training loss is")
    print(train_loss)
    print("Validation loss is")
    print(val_loss)
 
    return train_loss,val_loss


def check_accuracy(model,x_smile,x_seq,y_model,filename, batch_size=bt):
		num_correct = 0
		num_samples = 0
		model.eval()
		count=1
		num_pred=[]
		num_prob=[]

		with torch.no_grad():
				for beg_i in range(0, x_smile.shape[0], batch_size):
						x_smile_batch = x_smile[beg_i:beg_i + batch_size]
						x_seq_batch = x_seq[beg_i:beg_i + batch_size]
						y_batch = y_model[beg_i:beg_i + batch_size]
						x_smile_batch=torch.FloatTensor(x_smile_batch)
						x_seq_batch=torch.FloatTensor(x_seq_batch)
						y_batch=y_batch.tolist()
						y_batch=torch.LongTensor(list(y_batch))
						x_smile_batch = Variable(x_smile_batch)
						x_seq_batch = Variable(x_seq_batch)
						y_batch = Variable(y_batch)

						x_smile_batch=x_smile_batch.to(device)
						x_seq_batch=x_seq_batch.to(device)
						y_batch=y_batch.to(device)

						scores = model(x_smile_batch,x_seq_batch)
						_, predictions = scores[0].max(1)
						probs = [i[1].item() for i in scores[1]]
						num_pred.extend(predictions.tolist())
						num_prob.extend(probs)
						num_correct += (predictions == y_batch).sum()
						num_samples += predictions.size(0)
						count+=1

				y_model=torch.Tensor.cpu(y_model)

				accuracy=accuracy_score(np.array(y_model),np.array(num_pred))
				precision=precision_score(np.array(y_model),np.array(num_pred))
				recall=recall_score(np.array(y_model),np.array(num_pred))
				f1=f1_score(np.array(y_model),np.array(num_pred))
				roc=roc_auc_score(np.array(y_model),np.array(num_pred))
				kappa=cohen_kappa_score(np.array(y_model),np.array(num_pred))
				conf_matrix=confusion_matrix(np.array(y_model),np.array(num_pred))
				bal_acc=balanced_accuracy_score(np.array(y_model),np.array(num_pred))

				lr_fpr, lr_tpr, _ = roc_curve(np.array(y_model),np.array(num_prob))
				plt.figure()
				plt.plot(lr_fpr, lr_tpr, marker='.', label='BLSTM')
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.legend()
				plt.savefig("ROC.pdf".format(filename))
				plt.close()
				print(accuracy, precision, recall, f1, roc, kappa, conf_matrix, bal_acc)



train_df,val_df=train_test_split(df,test_size=0.2, shuffle=True)
df_minority=train_df[train_df['Activation_Status']==1]
df_majority=train_df[train_df['Activation_Status']==0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0]) 
df_minority_upsampled = pd.concat([df_majority, df_minority_upsampled ])
train_df=df_minority_upsampled
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.to_csv("{}_train.csv".format(filename), index=False)
val_df.to_csv("{}_test.csv".format(filename), index=False)

#initialising new model

model=BLSTM(77,smile_h,1,27,seq_h,1,2)
model.to(device)
criterion= nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#defining X and Y for model
X_smile_onehot_train=train_df["SMILES"].apply(one_hot_smile)
X_seq_onehot_train=train_df["Final_Sequence"].apply(one_hot_seq)
Y_train=train_df["Activation_Status"]

X_smile_onehot_val=val_df["SMILES"].apply(one_hot_smile)
X_seq_onehot_val=val_df["Final_Sequence"].apply(one_hot_seq)
Y_val=val_df["Activation_Status"]

#changing X and Y to tensor
X_smile_onehot_train=list(X_smile_onehot_train)
X_smile_onehot_train=torch.stack(X_smile_onehot_train)
X_seq_onehot_train=list(X_seq_onehot_train)
X_seq_onehot_train=torch.stack(X_seq_onehot_train)

Y_train=list(Y_train)
Y_train=torch.Tensor(Y_train)

X_smile_onehot_val=list(X_smile_onehot_val)
X_smile_onehot_val=torch.stack(X_smile_onehot_val)
X_seq_onehot_val=list(X_seq_onehot_val)
X_seq_onehot_val=torch.stack(X_seq_onehot_val)

Y_val=list(Y_val)
Y_val=torch.Tensor(Y_val)

#defining losses
e_losses = []
v_losses=[]
scores_dic={}
epochwise_train={}
epochwise_test={}
#training model

for e in range(num_epochs):
	print(str(filename) + "_epoch\t"+str(e))
	loss=train_epoch(model,X_smile_onehot_train,X_seq_onehot_train,Y_train,X_smile_onehot_val,X_seq_onehot_val,Y_val, epochwise_train,epochwise_test, e)
	train_loss=loss[0]
	val_loss=loss[1]
	e_losses.append(train_loss.item())
	v_losses.append(val_loss.item())


test_accuracy=[]
test_precision=[]
test_recall=[]
test_kappa=[]
test_bal_acc=[]
for i in epochwise_test:
	test_accuracy.append(epochwise_test[i][0])
	test_precision.append(epochwise_test[i][1])
	test_recall.append(epochwise_test[i][2])
	test_kappa.append(epochwise_test[i][3])
	test_bal_acc.append(epochwise_test[i][4])
# Accuracy per epoch plot: (Train)
train_accuracy=[]
train_precision=[]
train_recall=[]
train_kappa=[]
train_bal_acc=[]
for i in epochwise_train:
	train_accuracy.append(epochwise_train[i][0])
	train_precision.append(epochwise_train[i][1])
	train_recall.append(epochwise_train[i][2])
	train_kappa.append(epochwise_train[i][3])
	train_bal_acc.append(epochwise_train[i][4])

#saving plots
plt.figure()
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(test_accuracy, label="Test Accuracy")
plt.legend()
plt.savefig("{}_AccuracyPerEpoch.pdf".format(filename))
plt.close()
plt.figure()
plt.plot(train_precision, label="Train Precision")
plt.plot(test_precision, label="Test Precision")
plt.legend()
plt.savefig("{}_PrecisionPerEpoch.pdf".format(filename))   
plt.close()
plt.figure()
plt.plot(train_recall, label="Train Recall")
plt.plot(test_recall, label="Test Recall")
plt.legend()
plt.savefig("{}_RecallPerEpoch.pdf".format(filename))
plt.close()
plt.figure()
plt.plot(train_kappa, label="Train kappa")
plt.plot(test_kappa, label="Test kappa")
plt.legend()
plt.savefig("{}_KappaPerEpoch.pdf".format(filename))
plt.close()
plt.figure()
plt.plot(train_bal_acc, label="Train bal_acc")
plt.plot(test_bal_acc, label="Test bal_acc")
plt.legend()
plt.savefig("{}_Balanced_AccuracyPerEpoch.pdf".format(filename))
plt.close()
plt.figure()
plt.plot(v_losses, label='Validation loss')
plt.plot(e_losses, label='Training loss')
plt.legend()
plt.savefig("{}_Loss.pdf".format(filename))
plt.close()

# modelfilename = "{}_model.sav".format(filename)
# pickle.dump(model, open(modelfilename, 'wb'))
torch.save(model.state_dict(), "{}_model.pt".format(filename))

check_accuracy(model,X_smile_onehot_train,X_seq_onehot_train,Y_train, "{}_train".format(filename))
check_accuracy(model,X_smile_onehot_val,X_seq_onehot_val,Y_val, "{}_test".format(filename))

