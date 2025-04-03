# EEG_DeepLearning_Classifications-
Educational&Research Deep Learning Tool for classifying resting state EEG for pathological investigations

 AD EEG Classification using a modified hybrid 1d CNN and GRU RNN
 original description of Network architecture by Roy, Kornek & Harrer 2018, ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification arXiv, https://arxiv.org/abs/1802.00308 )

 Written by Dr. Antonios Dougalis, Kozani 05th Dec 2024, Greece
modified on 3rd of April 2025, Kozani, Greece

# contact me for any queries on: antoniosdougalis@gmail.com
# THE PARTICIPATNTS FILE IS PROVIDED FOR THE LABELS

# (PLEASE DOWNLOAD THE EEG DATA FROM OpenNeuro: https://openneuro.org/datasets/ds004504 )
# THEN IMPORT THEM APPROPRIATELY AT LINE 176 
# BLOCK #%% read Data and data labels and demographics

# Description
Quantitative EEG analysis and predictions using deep learning tools have attracted a lot of attention due to the powerful repercussion that they can have on diagnosis of neurological and psychiatric disease
Herein I present the analysis of a publicly available, closed eyes, resting state EEG study of 36 individuals diagnosed with Alzheimer's disease (AD group), 23 diagnosed with Frontotemporal Dementia (FTD group) and 29 healthy subjects (CN group). A Nihon Kohden EEG 2100 clinical device was used, with 19 scalp electrodes (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2) according to the 10-20 international system and 2 reference electrodes (A1 and A2) placed on the mastoids for impedance check. Data are publicly available at OpenNeuro: https://openneuro.org/datasets/ds004504 
I used standard band pass filtering, common average referencing and ICA to remove any artefacts, before segmenting the data in 5 sec epochs with 1 second overlaps and removing any bad epochs based on noise levels (A). 
I then use this dataset to perform a three-label classification analysis via deep learning (DL) techniques (B) by building a variant of a hybrid model architecture of sequences of 1D convolutional units (Convolutional Neural Networks [CNNs], Inception modules) followed by series of layers of gated recurrent units (Recurrent Neural Networks [RNNs], GRU modules). Such structures have been utilized before and have been shown to be successful in discerning differences in pathological EEG (see Roy, Kornek & Harrer 2018, ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification arXiv, https://arxiv.org/abs/1802.00308 ) 
I show that after only 50 rounds of training with typical gradient descend learning protocols using an Adam optimizer with labelled epoched data (and a development set) results in mean 88.6 % accuracy of classification prediction when the model is presented with unknown data that it has never seen before (test set C, D). The data show that the model most often confused the FTD condition for AD (18% of epoch case) but scored highly in classifying correctly CN (91%) or AD (94%) individuals. Had that been a two-class problem, the results would have been staggering, but this is not the point here!
