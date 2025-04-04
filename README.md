# AI-powered-Public-Health-Automated-Kiosk-System-
# Title: An AI-powered Public Health Automated Kiosk System for Personalized Care: An Experimental Pilot Study
# ArXive: 


Requirements:

PyTorch >= 0.4
Python >= 3.5

Dataset Link:
The dataset used in this study can be accessed at the following link: https://github.com/sjy1203/GAMENet/tree/master/data  

Data Information in ./data/

records_final.pkl: Input data with four dimensions (patient_idx, visit_idx, medical modal, medical id) where medical modal equals 3 (made of diagnosis, procedure, and drug).
voc_final.pkl: Vocabulary list for transforming medical words to corresponding indices.
ddi_A_final.pkl, ehr_adj_final.pkl: Drug-drug adjacency matrices constructed from EHR and DDI datasets.
drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt: Mapping files for drug code transformation.

Running the HERMES Kiosk Model

To train and run the model: python train.py

AUTHOR(S):

Sonya Falahati,Morteza Alizadeh and Mohammad R. Salmanpour (PHD)

STATEMENT: This files are part of above papers. Package by Mohammad R.Salmanpour,Sonya Falahati, and Morteza Alizadeh. --> Copyright (C) 2025 Mohammad R. Salmanpour, BC Cancer Rsearch Center. This package is distributed in the hope that it will be useful. It is flexible license for research products. For commercial use, please get permison from Dr. Mohammad R. Salmanpour via eamil address m.salmanpoor66@gmail.com or Sonya Falahati via eamil address falahati.sonya@gmail.com.

Any feedback welcome!!! m.salmanpoor66@gmail.com, msalman@bccrc.ca and falahati.sonya@gmail.com 
