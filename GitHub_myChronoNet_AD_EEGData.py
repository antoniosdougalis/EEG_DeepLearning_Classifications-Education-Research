## AD EEG Classification using a hybrid 1d CNN and GRU RNN
## Written by Dr. Antonios Dougalis, Kozani 05th Dec 2024, Greece
# modified on 3rd of April 2025, Kozani, Greece

#  contact me for any queries on: antoniosdougalis@gmail.com
# (PLEASE DOWNLOAD THE DATA FROM THE OpenNeuro: https://openneuro.org/datasets/ds004504 )
# THEN INSERT THEM AT LINE 176 BLOCK #%% read Data and data labels and demographics

import mne
import os

# from scipy import stats 

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import time
import sys

# for excell file importation
import pandas as pd

#%% Generate some synthetic data to have for inputs

# Generate synthetic data
def generate_synthetic_data(num_samples, input_dim, seq_len, num_classes):
    
    # Random EEG data (num_samples, input_dim, seq_len )
    X = np.random.rand(num_samples, input_dim, seq_len).astype(np.float32)  # 10 time steps
    
    # Random labels
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

    
#%% create a function to read and preprocess the data 

def read_EEG_data(file_path):
    data = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    data.set_eeg_reference(verbose=False) # averaging the channels as reference
    data.filter(l_freq=0.5, h_freq=45, verbose=False) # filter the data
    epochs = mne.make_fixed_length_epochs(data, duration = 5, overlap = 1, verbose=False) # split the data into epochs (units in seconds)
    epochs_array = epochs.get_data() # becomes numpy array
    return epochs_array

#%% Inception Class

class Inception(nn.Module):
    
    def __init__(self, in_channels, out_convChan, printToggle=False):
        super(Inception, self).__init__()
        self.printToggle = printToggle
        self.conv1 = nn.Conv1d(in_channels, out_convChan, 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv1d(in_channels, out_convChan, 2, stride = 2, padding = 0)
        self.conv3 = nn.Conv1d(in_channels, out_convChan, 8, stride = 2, padding = 3)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        if self.printToggle: print(f'shape after 3 conv is {x1.shape}') 
        
        # concatanate data on the channel dimension
        cat = [x1, x2, x3]
        x = torch.cat(cat, dim=1) 
        if self.printToggle: print(f'final Inception output shape after concatenation is {x.shape}') 
        return x

#%% Classify epochs using 1D CNN and a GRU netowrk

# for the conv1d layers
num_timepoints = 2500
in_channels = 19 # real EEG channels
out_convChan = 32 # out feat for each of the three conv units of inception: 
kernel_size = 4
stride = 2
padding = 1

dilation = 1 # dilation for the conv1d
out_feat = 32 # output features also know as out_chan of the last conv1d layer

Lin = num_timepoints
Lout = 0 # initialise

# Calculate Lout for conv1D output per layer (N, Cin, Lout)
for idx in range(3):
   Lout = int( np.floor( ( (Lin + 2*padding - (dilation*(kernel_size-1)) - 1 )/stride)+1  ) )
   # print(idx,Lout)
   Lin = Lout

# parameters for GRU
num_layers = 3
input_size = out_feat # number of input features (output features of the last conv1d layer)
hidden_size = 24  # number of hidden units in GRU
out_classes = 3 # number of classes that need prediction (0 and 1)

GRU1out, GRU2out, GRU3out, GRU4out = 32, 32, 32, 1

#%% myChronoNet (a modification of the original description 
# Roy, Kornek & Harrer 2018, ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification arXiv, https://arxiv.org/abs/1802.00308 )

class myChronoNet(nn.Module):
    
    def __init__(self, in_channels, out_convChan, num_timepoints, out_classes, printToggle=True):
        
        super(myChronoNet, self).__init__()
        self.printToggle = printToggle
        self.inception1 = Inception(in_channels, out_convChan)
        self.inception2 = Inception(3*out_convChan, out_convChan)
        self.inception3 = Inception(3*out_convChan, out_convChan)
        
        self.gru1 = nn.GRU(3*out_convChan, GRU1out, num_layers = 1, batch_first = True)
        self.gru2 = nn.GRU(GRU1out, GRU2out, batch_first = True)
        self.gru3 = nn.GRU(GRU1out + GRU2out, GRU3out, batch_first = True)
        self.gru4 = nn.GRU(GRU1out + GRU2out + GRU3out, GRU4out, batch_first = True)
        
        
        self.affine1 = nn.Linear(Lout, out_classes)  
        

    def forward(self, x):
        x = x.contiguous().view(-1, in_channels, num_timepoints)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        if self.printToggle: print(f'final Inception output shape after concatenation is {x.shape}') 
        
        Lout = x.shape[-1] # get the seq length after dimension reductrion on num_timepoints by the 3 rounds of inception
        
        # prepare for input to GRU: # batch num, Lout, concatenated output of 3rd inception
        x = x.permute(0,2,1).contiguous()
        
        # Passs data through a GRU
        x, _ = self.gru1(x)
        if self.printToggle: print(f'after 1st  GRU I am of shape{x.shape}') 
        
        x_res1 = x # keep the 1st GRU result
        x, _ = self.gru2(x)
        if self.printToggle: print(f'after 2st  GRU I am of shape{x.shape}') 
        
        x_res2 = x # keep the 2nd GRU result
        # concatenate GRU1 amd GRU2 results on the final dim (hidden_size)
        x_cat1 = torch.cat([x_res1, x_res2], dim = 2)
        if self.printToggle: print(f' after 1 & 2 GRU concatenation shape is {x_cat1.shape}') 
        
        # 3rd GRU to the concatenated result
        x, _ = self.gru3(x_cat1) # pass concatenated GRU 1 and 2 result to GRU three
        x = torch.cat([x_res1, x_res2, x], dim = 2) # concatenate all 3 GRU res out on the hidden size dimensaion
        if self.printToggle: print(f' after 3 GRU, All three GRU concatenations and linear shape is {x.shape}') 
        
        x, _ = self.gru4(x)
        
        
        if self.printToggle: print(f' OUTPUT of the 4th GRUs {x.shape}') 
        x = torch.squeeze(x, dim = 2)
        x = self.affine1(x)
        x = x.squeeze()
        if self.printToggle: print(f'final output shape is {x.shape}') 

        return x


#%% use GPU power
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#%% read Data and data labels and demographics

#set directory to find the data (PLEASE DOWNLOAD THE DATA)
os.chdir(r'C:\Users\anton\Documents\Python Scripts\ADNI NeuroImaging')

# load saved data (instead of loading all results from EEG files)
AD_EEGdata = np.load('AD_EEGdata_dict.npz')
image_data = AD_EEGdata['EEGdata']
epochs_PerSubj = AD_EEGdata['epochs_PerSubj']
patient_id = AD_EEGdata['patient_id']

# Define the path to the .tsv file
csv_filePath = r'C:\Users\anton\Documents\Python_Data\EEG derivatives\participants.tsv'

# Read the .tsv file (since it's a tab-separated file, use `sep='\t'`)
df = pd.read_csv(csv_filePath, sep='\t')

# Check the shape of the dataframe (number of rows and columns)
print(df.shape)

# Rename the columns for clarity
df.columns = ['subject_id', 'gender', 'age', 'diagnosis', 'score']

# Display the resulting DataFrame
print(df.keys())

target_map = {'A':2, 'C':0, 'F':1 }
df['diagnosis'] = df['diagnosis'].map(target_map)

uniqueLabels= df['diagnosis'].unique()
num_uniqueLabels= len(uniqueLabels)

# Create the labels based on the epochs per subject
labels = np.zeros( (image_data.shape[0]))
labels.shape

# Create subject tracking labels based on the epochs per subject
group_labels = np.zeros( (image_data.shape[0]))
group_labels.shape

in_channels = image_data.shape[1]
num_timepoints = image_data.shape[-1]
maxFilePairCount = in_channels * in_channels;

# make it cumulative
cum_epochs_PerSubj = np.concatenate( (np.zeros(1), np.cumsum(epochs_PerSubj)) ).astype(int)
max_epochs_PerSubj = np.max(epochs_PerSubj) # for intialising

for idx in range(cum_epochs_PerSubj.shape[0]-1): 
  start = cum_epochs_PerSubj[idx]
  end = cum_epochs_PerSubj[idx+1]
  labels[start:end] = df.loc[idx,'diagnosis'].copy()
  group_labels[start:end] = idx
  

# electrode position x, y coordinates
pos = {
 'Fp1':[-0.0294367,  0.0839171], 'Fp2':[ 0.0298723,  0.0848959], 
 'F7': [-0.0702629,  0.0424743],  'F3': [-0.0502438,  0.0531112],  'Fz': [ 0.0003122,  0.058512 ],  'F4': [ 0.0518362,  0.0543048],  'F8': [ 0.0730431,  0.0444217], 
 'T3': [-0.0841611, -0.0160187],  'C3': [-0.0653581, -0.0116317],  'Cz': [ 0.0004009, -0.009167 ],  'C4': [ 0.0671179, -0.0109003],  'T4': [ 0.0850799, -0.0150203],
 'T5': [-0.0724343, -0.0734527],  'P3': [-0.0530073, -0.0787878],  'Pz': [ 0.0003247, -0.081115 ],  'P4': [ 0.0556667, -0.0785602],  'T6': [ 0.0730557, -0.0730683],
 'O1': [-0.0294134, -0.112449 ],  'O2': [ 0.0298426, -0.112156 ]
}

#extract channel names
ch_names = [i for i in pos]

#%% create and subtract a commonon average reference

# separate dat per patient
R = len(patient_id)
myData = np.full((R, max_epochs_PerSubj, in_channels, num_timepoints ), np.nan)
for pati in range(R):
    myData[pati, :epochs_PerSubj[pati], :, :] = image_data[cum_epochs_PerSubj[pati]:cum_epochs_PerSubj[pati+1], :, :]
    
#calculate CAR on the channel dimension    
EEGcar = np.full((R, max_epochs_PerSubj, in_channels, num_timepoints ), np.nan)
for pati in range(R):
    temp = np.mean(myData[pati, :epochs_PerSubj[pati], :, :], axis = 1) # the channel dimension of the patient
    temp = temp[:, None, :]
    EEGcar[pati, :epochs_PerSubj[pati], :, :] = myData[pati, :epochs_PerSubj[pati], :, :] - temp

# select random channel to plot
chan2plot = np.random.randint( in_channels )
pat2plot = np.random.randint( R )

f, ax = plt.subplots(figsize=(10, 5))
f.suptitle(f'Patient Num {pat2plot} ')

ax.set_title(f'Mean Epoch Voltage TimeSeries for Patient:{ch_names[chan2plot]}')
ax.plot( np.mean( myData[pat2plot, :epochs_PerSubj[pat2plot], chan2plot, :], axis = 0), 'b-', label = 'Voltage data')
ax.plot( np.mean( EEGcar[pat2plot, :epochs_PerSubj[pat2plot], chan2plot, :], axis = 0), 'r-', label = 'CAR Voltage data')
ax.set_xlabel('time')
ax.set_ylabel('Voltage (micV)')

plt.legend()

plt.savefig('AD_Sample_Traces.pdf', dpi=300)
plt.show()

# release memory
del myData

#%% reshape EEGcar back to the original image_data, store it and proceed with Laplacian calcluations

EEGdata = np.zeros( image_data.shape )

for pati in range(R):
    EEGdata[cum_epochs_PerSubj[pati]:cum_epochs_PerSubj[pati+1], :,:] = EEGcar[pati, :epochs_PerSubj[pati], :, :]


# release memory
del EEGcar

# rewrite image_data with the Car corrected voltage for convenience instead of changing all names below
image_data = EEGdata.copy()

#%% label, Z Normalise and shuffle EEG data: ready for use

data_T = torch.from_numpy( image_data )
label_T = torch.from_numpy( labels )
group_label_T = torch.from_numpy( group_labels )

# zscore normalisation
meanT = data_T.mean(dim=2, keepdim=True)  # Mean with shape (7000, 19, 1)
stdT = data_T.std(dim=2, keepdim=True)    # Std with shape (7000, 19, 1)
data_T = (data_T - meanT) / stdT

print(f'the Normalised data array \n data is {data_T.shape} \n and the label array is {label_T.shape}')

#shuffle the data and labels and groups across the first dimension
# Get the first dimension size
num_samples = data_T.shape[0]

# Create a random permutation of indices
indices = torch.randperm(num_samples)

# Shuffle the data using the random indices
# make sure data is type float.32 and labels are long ()
s_data =  data_T[indices,:,:].float()
s_label = label_T[indices].long()
g_label = group_label_T[indices].long()

print(f'the Normalised and Shuffled datta \n data array is {s_data.shape} \n and the label array is {s_label.shape}')

# release memory
# del data_T

#%% Display some random EEG epochs and their labels  

fig, axs = plt.subplots(2,3, figsize =(10,5))
randNum = np.random.randint(0, s_data.shape[0], 6 )

diagnosis_label = ['CN','FTD','AD']

for i, ax in enumerate(axs.flatten()):
  ax.plot( np.arange(0,5,1/500), torch.squeeze(torch.mean( s_data[randNum[i], :, :], axis=0) ) )
  ax.plot([0, 5 ], [0, 0 ], 'r--' )
  subj_idx = np.min( np.where( randNum[i] < cum_epochs_PerSubj[1:]) )# hack for plotting the correct sunbject for the displayed epoch trace
  ax.set_title(f' {patient_id[subj_idx]}: label {diagnosis_label[s_label[ randNum[i] ] ]}')  # Optional: Show filename as title
  ax.legend()
  ax.set(ylim= [-1, 1])
  ax.set_xlabel('time (s)')
  ax.set_ylabel('Z-scored voltage')
  ax.set_yticks(np.arange(-1,1.5,0.5))
  ax.set_xticks(np.arange(0,6,1))

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

print(f' data array size {image_data.shape} ')

#%% Load EEG Data, initialise, split Data and test Model by passing some data for AD Recognition

# resample-downsample data
s_data = s_data[:,:,::2] # keep every other point  
s_data.shape

num_timepoints = s_data.shape[2]
in_channels = s_data.shape[1] # real EEG channels
out_convChan = 32 # out feat for each of the three conv units of inception: 
kernel_size = 4
stride = 2
padding = 1

Lin = num_timepoints
Lout = 0 # initialise

# Calculate Lout for conv1D output per layer (N, Cin, Lout)
for idx in range(3):
   Lout = int( np.floor( ( (Lin + 2*padding - (dilation*(kernel_size-1)) - 1 )/stride)+1  ) )
   # print(idx,Lout)
   Lin = Lout

# for CrossEntrpy MultiClass NOTe it is passing the output through a softmax not a sigmoid, change in EmT
num_classes = num_uniqueLabels    # Number of diagnosis classes
lossfun = nn.CrossEntropyLoss() # expects labels to be longs()

# Get the first dimension size
num_samples = s_data.shape[0]

# data_lim = 1000
# # limit dataset for faster run time
# s_data = s_data[ :data_lim, :, : ] 
# s_label = s_label[ :data_lim ] 

# split data, grab small portion to make the run easily
TrainPortion = 0.8

# X_train, X_test, y_train, y_test = train_test_split(s_data, s_label, train_size = TrainPortion, shuffle=True, stratify=s_label)
# X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, train_size = 0.5, shuffle=True, stratify=y_test)
# print(f' the train: {X_train.shape}: the test: {X_test.shape}: the devset_ {X_dev.shape}')

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    s_data, s_label, g_label, train_size=TrainPortion, shuffle=True, stratify=s_label)
X_dev, X_test, y_dev, y_test, g_dev, g_test = train_test_split(X_test, y_test, g_test, train_size = 0.5, shuffle=True, stratify=y_test)
print(f' the train: {X_train.shape}: the test: {X_test.shape}: the devset_ {X_dev.shape}')

# convert into PyTorch Datasets
train_dataset = TensorDataset(X_train, y_train)
dev_dataset  = TensorDataset(X_dev, y_dev)
test_dataset  = TensorDataset(X_test, y_test)

# translate into dataloader objects
BS = 64
train_dataloader = DataLoader(train_dataset, batch_size = BS, shuffle=True, drop_last=True)
dev_dataloader  = DataLoader(dev_dataset, batch_size = BS, shuffle=True, drop_last=True) # TEST ALSO IN BATCHES NOT AS A SINGLE GO
test_dataloader  = DataLoader(test_dataset, batch_size = BS, shuffle=True, drop_last=True) # TEST ALSO IN BATCHES NOT AS A SINGLE GO

# Initialize the model, and optimizer
model = myChronoNet(in_channels, out_convChan, num_timepoints, out_classes, False)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0)

# just pass some data through to see whether it works: Real DATA
# sniff out the first batch of the train only
dataEEG, EEGlabels = next(iter(train_dataloader))
print(dataEEG.shape)

# run through model and loss function
trial = model(dataEEG)
# trial = torch.argmax(trial, dim = 1)
loss_trial = lossfun(trial, EEGlabels) 

print('mmodel intialised, data ready for use')

# release memory
# del X_train, X_test, y_train, y_test, X_dev, y_dev, dataEEG

EEG_AD_UnseenData_dict = {'X_dev': X_dev, 'y_dev': y_dev, 'X_test': X_test, 'y_test': y_test}
np.savez('EEG_AD_UnseenData_dict', **EEG_AD_UnseenData_dict)

#%% Training loop in epochs
num_epochs = 50

# numpy arrays to store loss and accuracy per epoch
train_losses, train_accuracies, test_losses, test_accuracies, elapsed_time = [ np.zeros((num_epochs)) for _ in range(5) ]
 
for epochi in range(num_epochs):
    
    # starttime
    start_time = time.time() 
       
    # initialise batch holding arrays
    train_batch_loss, train_batch_acc, test_batch_loss, test_batch_acc = [ [ ] for _ in range(4) ] 
    
    # training mode of the model allowed
    model.train()
    
    # load train_ batches: start loop
    for dataEEG, EEGlabels in train_dataloader:
        #print(f'this batch of EEG data has shape of {dataEEG.shape},\n the EEGlabels has shape {EEGlabels.shape}' )
        
        outputs = model(dataEEG)
        loss = lossfun(outputs, EEGlabels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy: CHANGE FOR CROSSENTROPY
        pred_class = torch.argmax(outputs, dim=1) # for Multclass entropy
        pred_acc = 100*torch.mean( (pred_class == EEGlabels).float() )
        
        train_batch_loss.append( loss.item() )
        train_batch_acc.append( pred_acc )
        
        # end of batch loop

    # Calculate average loss and accuracy for the epoch
    train_losses[epochi] = np.mean(train_batch_loss)
    train_accuracies[epochi] = np.mean(train_batch_acc)
    elapsed_time[epochi] = time.time()  - start_time
    
    msg = f' TRAIN: Finished epoch {epochi+1}/{num_epochs}, loss: {train_losses[epochi]:.2f}, accuracies: {train_accuracies[epochi]:.2f} %: total elapsed time {np.sum(elapsed_time):.2f} second so far '
    sys.stdout.write('\r' + msg)
    
    # training mode of the model allowed
    model.eval()
    
    # load test_ batches: start loop
    for dataEEG, EEGlabels in test_dataloader:
        #print(f'this batch of EEG data has shape of {dataEEG.shape},\n the EEGlabels has shape {EEGlabels.shape}' )
        
        with torch.no_grad(): # deactivate auto grad back prop
            outputs = model(dataEEG)
            loss = lossfun(outputs, EEGlabels)

        # Calculate accuracy: CHANGE FOR CROSSENTROPY
        pred_class = torch.argmax(outputs, dim=1) # for Multclass entropy
        pred_acc = 100*torch.mean( (pred_class == EEGlabels).float() )
        
        test_batch_loss.append( loss.item() )
        test_batch_acc.append( pred_acc )
        
        # end of batch loop

    # Calculate average loss and accuracy for the epoch
    test_losses[epochi] = np.mean(test_batch_loss)
    test_accuracies[epochi] = np.mean(test_batch_acc)
    elapsed_time[epochi] = time.time()  - start_time
    
    msg = f'DEVTEST: Finished epoch {epochi+1}/{num_epochs}, loss: {test_losses[epochi]:.2f}, accuracies: {test_accuracies[epochi]:.2f} %: total elapsed time {np.sum(elapsed_time):.2f} second so far '
    sys.stdout.write('\r' + msg)


# Plot loss and accuracy
fig, axs = plt.subplots(1,2, figsize=(12,5))

# Loss plot
axs[0].plot(train_losses, label='train_Loss', color='green')
axs[0].plot(test_losses, label='test_Loss', color='blue')
axs[0].set_title('Loss per Epoch')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid()
axs[0].legend()

# Accuracy plot
axs[1].plot(train_accuracies, label='train_Accuracy', color='green')
axs[1].plot(test_accuracies, label='test_Accuracy', color='blue')
axs[1].set_title('Accuracy per Epoch')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig('Final_myChronoNet_plot.pdf', dpi=300)
print(os.getcwd())

# save the model
torch.save(model.state_dict(),'Final_trained_myChronoNet_EEG_AD_Classifier_Model.pt')
print(f' model saved to disk \n directory {os.getcwd()}')

#%% #REeload the model for inspection
# optionally compare it to a completely random freshly intilaised model

# create a new models of the same class 
trained_model = myChronoNet(in_channels, out_convChan, num_timepoints, out_classes, False)
random_model =  myChronoNet(in_channels, out_convChan, num_timepoints, out_classes, False)

# replace one model's parameters with those of the trained net
trained_model.load_state_dict(torch.load('Final_trained_myChronoNet_EEG_AD_Classifier_Model.pt'))
#random_model.load_state_dict(torch.load('trained_myChronoNet_EEG_AD_Classifier_Model.pt'))

# # send to device
# trained_model.to(device)
# random_model.to(device)

#%% test it on a single random batch of the train set that has not been seen by the model


# trained_model.to(device)
# random_model.to(device)

# Get the total number of batches
num_batches = len(dev_dataloader)

exp_acc, exp_loss, exp_DangerPred = [np.zeros(num_batches) for _ in range(3)]
elapsed_time = np.zeros(num_batches)


trained_model.eval()
start_time = time.time()
    
#Get every indivudal batch 
for experi, (batch_EEGdata, batch_EEGlabels) in enumerate(dev_dataloader):
   
    X = batch_EEGdata
    y = batch_EEGlabels
    # # # # New!
    # X = X.to(device)
    # y = y.to(device)

    # run the data through the loaded trained model
    # generate predictions
    yHatTrained = trained_model(X).detach()
    yHatRandom = random_model(X).detach()

    # generate loss
    trained_loss = lossfun(yHatTrained, y)
    random_loss = lossfun(yHatRandom, y)

    # # # New! bring outputs back
    yHatTrained = yHatTrained.cpu()
    yHatRandom = yHatRandom.cpu()
    y = y.cpu()

    # labels FOR cROISSeNTROPY
    labels_trained = torch.argmax(yHatTrained, axis=1).float() # boolean (minus or plus value) turned 0 to 1 label
    labels_random = torch.argmax(yHatRandom, axis=1).float()

    # comparative to real labels
    comp_trained = (labels_trained==y).float() # this is boolean True/False turned numeric 0 or 1
    comp_random  = (labels_random ==y).float()

    # accuracy based
    trained_acc = 100*torch.mean(comp_trained).item()
    random_acc = 100*torch.mean(comp_random).item()

    misclassified = np.where( (torch.argmax(yHatTrained, axis=1).float() )!= y)[0] # indexes of missclassified
    danger_missclass = torch.sum( torch.logical_and( (labels_trained==0) , (y!=0) ).float() )
    percentDanger = 100*(danger_missclass)/y.shape[0]

    # get every batch experiment loss and accuracy
    exp_acc[experi] = trained_acc
    exp_loss[experi] = trained_loss.item()
    exp_DangerPred[experi] = percentDanger

    elapsed_time[experi] = time.time() - start_time

    msg = f' Random Unseen Batches:{experi+1}/{num_batches}, loss: {exp_loss[experi]:.2f}, accuracies: {exp_acc[experi]:.2f}, danger predictions {exp_DangerPred[experi]:.2f} %, elapsed time {np.sum(elapsed_time):.2f} seconds'
    sys.stdout.write('\r' + msg)


# show that the results overlap
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(range(0, labels_trained.shape[0]), labels_trained, 'bo', mfc = 'w', markersize=10, label='Predicted Labels')
# plt.plot(misclassified, num_classes*np.ones( misclassified.shape[0] ),'ro', label='Misclassified Labels')
plt.plot(misclassified, labels_trained[misclassified],'ro', label='Misclassified Labels')
plt.plot(y,'go',mfc = 'w', label='Real Labels')
plt.legend()
plt.xlabel('Epoch idx')
plt.ylabel('Outcome Class')
# plt.title(f'Model Prediction Comparison: Accuracy: {np.round(trained_acc,2)} %, Dangerous predictions {percentDanger.numpy():.2f} %')
plt.title(f'Model Prediction Comparison: Accuracy: {np.round(trained_acc,2)} % ')
plt.xlim([0,labels_trained.shape[0] ])
plt.xlim([0,labels_trained.shape[0] ])
plt.yticks(np.arange(0, 3, 1), ['CN','FTD','AD'])


# show me the missclassified and their
# Pre =  (yHatTrained> 0.5).float()
Pre =  torch.argmax(yHatTrained, axis=1).float()
Pre = Pre[misclassified]
Real = y[misclassified]

plt.subplot(3,1,2)
plt.plot(Real, 'go-', label = 'real')
plt.plot(Pre, 'bo-', label = 'Pred')
plt.title('Comparison of Misclassified Epochs only')
plt.xlabel('Misclassified Epoch idx')
plt.ylabel('Outcome Class')
plt.yticks(np.arange(0, 3, 1), ['CN','FTD','AD'])
plt.legend()


plt.subplot(3,1,3)
plt.plot(range(0,num_batches,1), exp_acc, 'go-', label = 'Accuracy')
# plt.title(f'Classification Accuracy of Unseen Epochs \n mean acc {np.mean(exp_acc):.2f} % with Dangerous Predictions {np.mean(exp_DangerPred):.2f} %')
plt.title(f'Mean Classification Accuracy of Unseen Epochs: {np.mean(exp_acc):.2f} % ')
plt.xlabel('Number of Random Unseen Experimental Batches presented')
plt.ylabel('% Acccuracy')
plt.legend()
plt.tight_layout()

plt.savefig('ChrononNet_Results_plot.pdf', dpi=300)
print(os.getcwd())

plt.tight_layout()
plt.show()

#%% metrics and confusion matrixes: Calcuate mean maps after 100 experiments
import sklearn.metrics as skm


# repat confusion matrix a number of times
exp_C = np.zeros( (num_batches, len(target_map), len(target_map)) ) 
exp_metrics = np.zeros( (num_batches, len(target_map), len(target_map)) ) 


elapsed_time = np.zeros(num_batches)

#Get every indivudal batch 
for experi, (batch_EEGdata, batch_EEGlabels) in enumerate(dev_dataloader):
   
    X = batch_EEGdata
    y = batch_EEGlabels
    # for cuda
    # X = X.to(device)
    # y = y.to(device)

    # run the data through the loaded trained model
    # generate predictions
   
    yHatTrained = trained_model(X).detach()
    
    # test metrics
    test_metrics  = [0,0,0,0]
    test_predictions = torch.argmax(yHatTrained,axis=1)
    
    test_metrics[0] = skm.accuracy_score (y, test_predictions)
    test_metrics[1] = skm.precision_score(y, test_predictions,average=None)
    test_metrics[2] = skm.recall_score   (y, test_predictions,average=None)
    test_metrics[3] = skm.f1_score       (y, test_predictions,average=None)
    
    # compute the test confusion matrix
    C = skm.confusion_matrix(y, test_predictions, normalize='true')
    exp_C[experi,:,:] = C
    
    msg = f' reached {experi}/{len(dev_dataloader)}, patient please'
    sys.stdout.write('\r' + msg)


# define target_map
target_map = {'CN': 0, 'FTD': 1, 'AD': 2}

# visualize it
fig = plt.figure(figsize=(10,10))
plt.imshow(np.mean(exp_C, axis=0),'jet', vmin = 0, vmax=0.25)

# If you want the sorted dictionary as well:
sorted_keys = sorted(target_map, key=target_map.get)
sorted_dict = {key: target_map[key] for key in sorted_keys}

# make the plot look nicer
plt.xticks(range(len(target_map)),labels = target_map)
plt.yticks(range(len(target_map)),labels = target_map)
plt.title('TEST confusion matrix')
plt.ylabel('True Diagnosis Class')
plt.xlabel('Predicted Diagnosis Class')

for i in range(len(C)):
    for j in range(len(C[i])):
        plt.text(j, i, f'{np.mean(exp_C, axis=0)[i, j]:.2f}', ha='center', va='center', color='black', fontsize=32)

# plt.colorbar()

plt.savefig('ChrononNet_Confusion_Matrix.pdf', dpi=300)
print(os.getcwd())

plt.show()

#%% END OF SCRIPT
