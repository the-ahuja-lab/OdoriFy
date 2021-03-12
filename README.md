<div align="center"> <h1>OdoriFy </h1> 
<b>A comprehensive AI-driven web-based solution for Human Olfaction</b>
 </div>
 <br>
<div align="center">
<img src="https://imgur.com/2gJZMWo.gif" alt="Odorify" width="500" height="400"></div>
<br>
OdoriFy is an open-source web server with Deep Neural Network-based prediction models coupled with explainable Artificial Intelligence functionalities, developed in an effort to provide the users with a one-stop destination for all their problems pertaining to olfaction. OdoriFy is highly versatile, capable of deorphanizing human olfactory receptors (Odor Finder), odorant prediction (Odorant Predictor), identification of Responsive Olfactory Receptors for a given ligand/s (OR Finder), and Odorant-OR Pair Analysis. With OdoriFy, we aim to provide a simplified and hassle-free environment for users.

**Webserver is freely available at [https://odorify.ahujalab.iiitd.edu.in/](https://odorify.ahujalab.iiitd.edu.in/olfy/)**

Entire webserver code is avaialble at [https://github.com/the-ahuja-lab/Odorify-webserver](https://github.com/the-ahuja-lab/Odorify-webserver)


## Index
* [Prediction Engines](#prediction-engines-)
* [Dependencies](#dependencies-)
* [How to use OdoriFy](#how-to-use-odorify-)



## Prediction Engines: [&uarr;](#index-)

1.  **Odorant Predictor:** OdoriFy allows users to predict or verify whether the user supplemented chemicals qualifies for the odorant properties. It also performs the sub-structure analysis and highlights atoms indispensable for the predicted decision. 
<u>Input:</u> Chemical (SMILES)
    
2.  **OR Finder**: It enables the identification of cognate human olfactory receptors for the user-supplied odorant molecules. Moreover, similar to Odorant Predictor it also highlights odorant sub-structures responsible for the predicted interactions. 
<u>Input:</u> Chemical (SMILES)

3.  **Odor Finder:** OdoriFy allows users to submit the protein sequences of wild-type or mutant human ORs and performs prediction and ranking of their potential cognate odorants.
 <u>Input:</u> Protein sequences (FASTA format)

4.  **Odorant-OR Pair Analysi**s: OdoriFy flexibility also supports the validation of OR-Odor pairs supplement by the user. Moreover, the explainable AI component of OdoriFy returns the sub-structure analysis (odor) and marking of key amino acids (OR) responsible for the predicted decision.
 <u>Input:</u> Chemical (SMILES) and Protein sequences (FASTA Format).

<div align="center">
<img src="https://imgur.com/22h9n9x.png" alt="Architecture" width="650" height="480"></div>

## Dependencies [&uarr;](#prediction-engines-)
1.  Python v.3.4.6 or higher
2.  TensorFlow v1.12
3.  rdkit v.2018.09.2
4. Conda Environment
5. Pytorch
6. Captum


## How to use OdoriFy [&uarr;](#dependencies-)
**Step 1: Build a model**

To build model from own data, Write a config.cfg file in the format specified below:
```
	[Task]
	train_data_file = odorants.csv
	smile_length = 75
	sequence_length = 315
	smile_hidden = 50
	sequence_hidden=50
	smile_o=50
	seq_o=50
	batch_size=32 
	num_epochs=3
	learning_rate= 0.0001
	filename=trial
```

Run `python build-model.py config-build-model.cfg`. This will create a model sav file. 

**Step 2: Find predictions and explainability of predictions**

For finding predictions based on this trained model, Write a config.cfg file similar to this:
```
	[Task]
	train_data_file=data/odorants.csv
	model_file = data/42_model.sav
	apply_data_file = data/sample.csv
	smile_length = 75
	sequence_length = 315
	filename = data/trial
	result_file = data/output.csv
```
Now run prediction engines via:

`python odor-finder.py config-prediction.cfg`	

`python or-finder1.py config-prediction.cfg`

`python or-finder2.py config-prediction.cfg`

 `python odorant-or.py config-prediction.cfg`

