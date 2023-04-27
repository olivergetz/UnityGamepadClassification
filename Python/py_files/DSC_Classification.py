#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import librosa
import sklearn.model_selection
import torch
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
import random
import time
import os
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score


# ## Settings
# 
# Easy access to everything that might need tweaking before training.

# In[118]:


# This seed is used to seed everything seedable.
seed = 4052
# The path where the data was recorded.
data_path = "../CollectedData/Dualsense/"
metrics_path = "./models/systematic_metrics.csv"
best_model_paths = "./models/systematic/"

# The portion of the dataset used for training.
training_size = 0.9
# Since the window is what we are using to make the computation, 
# The buffer in Unity during inference should be the same value.
window_length_feature_extraction = 512
# Since features are computed constantly in Unity, low hop might be best.
hop_length_feature_extraction = 16

# Select dataset
current_dataset = 3
# Name, Sample Rate
dataset_info = [
    ['Guided', 100],
    ['Guided_Exaggerated', 100],
    ['Guided_Highres', 200],
    ['Guided_Long', 100],
    ['Guided_Long_Highres', 300],
    ['No_Guide', 100],
    ['No_Guide_Exaggerated', 300],
    ['Separation_Test', 200]
]

current_metric = 0
metric_names = ['accuracy', 'precision', 'recall', 'f1']

model_name = "dsc_sys-" + dataset_info[current_dataset][0].lower()

# Percentage of data to filter out from start and finish to account for slow start.
filter_portion = 0.02

# These are irrelevant for the current task due to being idle. 
# We remove them to speed up the training process.
cols_to_ignore = ["Timestamp", "D-Pad", "Touch", "L3", "R3", "L2",
                  'Button North', 'Button East']

# Exclude Mean, Var, or RMS for testing, or leave empty to keep all features.
# Remember to set the appropriate bools in Unity.
feats_to_ignore = ["Var"]

# Model Hyperparameters
epochs = 12
hidden_layers = 2
hidden_dims = 64
dropout = 0.7
learning_rate = 5e-7
max_learning_rate = 5e-5
gamma=0.9 # Not used for the final model. Used for experiments with other optimizers.
use_raw = False; # Want to train on the raw data, or the extracted features?
early_stop_patience = 5
early_stop_thresh = 0.0005

# All unique class labels. Used for debugging predictions.
# Order alphabetically.
labels_dict = {
    0 : "high_activity",
    1 : "idle",
    2 : "low_activity",
    3 : "medium_activity"
}


# ## Setup
# 
# Class and function definitions

# In[101]:


# Class Definitions
class DSC_Classifier(nn.Module):
    def __init__(self, n_feats, n_labels, n_hidden_layers, hidden_dims, dropout):
        super().__init__()
        self.input = nn.Linear(n_feats, hidden_dims, bias=True)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims, hidden_dims, bias=True)
            for _ in range(n_hidden_layers)
        ])
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, n_labels)
        )
        
    def forward(self, x):
        x = self.input(x).relu()
        
        # Pass input through all hidden layers
        for layer in self.hidden_layers:
            x = x + layer(x).relu()
            
        logits = self.output(x)
        return logits
        
class EarlyStopper:
    # patience: Max consecutive epochs of no improvement
    # learn_thresh: the amount of improvement needed to count as learning
    def __init__(self, patience : int = 3, learn_thresh : float = 0.001, gl_threshold = 50):
        # GL Stopper
        self.generalization_loss = 0
        self.lowest_loss = 100
        self.gl_threshold = gl_threshold
        
        # Metric stopper
        self.patience = patience
        self.elapsed_epochs = 0
        self.best_score = 0
        self.learn_thresh = learn_thresh
        
    def generalization_stop(self, validation_loss : float):
        # Update lowest validation set error
        if (validation_loss < self.lowest_loss): self.lowest_loss = validation_loss
        # GL = The relative increase in loss over min loss, expressed as %.
        #self.generalization_loss = 100*(validation_loss / self.lowest_loss - 1) 
        self.generalization_loss = 100 * (validation_loss - self.lowest_loss) / self.lowest_loss
        print(self.generalization_loss)
        if (self.generalization_loss > self.gl_threshold):
            return True
        return False
    
    def get_generalization_loss(self):
        return self.generalization_loss
    
    # This is a stopper used to stop when the supplied metric is not improving for n epochs.
    def stop_check(self, metric_current_score : float):
        # Evaluate current score
        if metric_current_score > self.best_score:
            self.best_score = metric_current_score
            self.elapsed_epochs = 0
        elif metric_current_score < (self.best_score - self.learn_thresh):
            self.elapsed_epochs += 1
        # If n epochs have passed, stop.
        if self.elapsed_epochs >= self.patience:
            print("Stopped by EarlyStopper.")
            return True
        return False


# In[102]:


# Function Definitions
def seed_everything(seed_value=4052):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
def feature_extraction(data, sr=100, window_length = 256, hop_length = 64):
    
    extracted_features_frame = pd.DataFrame()
    for col in data.columns:
        # Convert to numpy for easy indexing.
        data_np = data[col].values
        
        '''zcr = librosa.feature.zero_crossing_rate(y=np.array(data[col], dtype=np.float64),
        frame_length=window_length, 
        hop_length=hop_length)'''

        rms = librosa.feature.rms(y=np.array(data[col], dtype=np.float64),
                                  frame_length=window_length,
                                  hop_length=hop_length)
        # Init lists
        variance = []
        mean = []
        click_rate = [] # Click Rate = n_nonZero / butfferSize
        # Sliding window mean and variance
        for i in range(data_np.size // hop_length):
            # Get current window
            current_position = i * hop_length
            window_end_idx = current_position+window_length
            current_window = data_np[current_position:window_end_idx]
            # Calculate features and add them to their respective lists
            window_mean = np.mean(current_window)
            mean.append(window_mean)
            window_variance = np.var(current_window)
            variance.append(window_variance)
            
        # Make sure all other features match the length of the first feature
        features_length = len(mean)
        extracted_features_frame[col + " Mean"] = mean
        extracted_features_frame[col + " Var"] = variance

        # librosa returns array of size (0, n). Remove the first dimension.
        rms = rms.squeeze()
        # Cut or pad features to an appropriate length
        if(rms.size > features_length):
            diff = np.absolute(rms.size-features_length)
            rms = rms[:-diff]
        elif (rms.size < features_length):
            diff = np.absolute(rms.size-features_length)
            rms = np.append(rms, np.zeros(diff))

        extracted_features_frame[col + " RMS"] = rms

    #extracted_features_frame[col + " ZCR"] = zcr.squeeze()

    return extracted_features_frame
    
def train(model, x, y, optimizer, scheduler):
    model.train()
    for feature_vector, label_true in zip(x, y):
        optimizer.zero_grad()
        label_pred = model(feature_vector)
        loss = nn.functional.cross_entropy(label_pred, label_true.long())
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss

@torch.no_grad()
def evaluate(model, x, y, n_classes):
    model.eval()
    labels_true, predictions = [], []
    for feature_vector, label_true in zip(x, y):
        output = model(feature_vector)
        predictions.append(output.argmax().tolist())
        labels_true.append(label_true.tolist())
        loss = nn.functional.cross_entropy(output, label_true.long())

    # Convert to tensors
    preds = torch.tensor(predictions)
    targets = torch.tensor(labels_true)
    
    #accuracy = (torch.tensor(predictions) == torch.tensor(labels_true)).float().mean() * 100.0
    accuracy = MulticlassAccuracy(average='macro', num_classes=n_classes)
    acc = accuracy(preds, targets)
    precision = MulticlassPrecision(average='macro', num_classes=n_classes)
    prec = precision(preds, targets)
    recall = MulticlassRecall(average='macro', num_classes=n_classes)
    rec = recall(preds, targets)
    f1_score = MulticlassF1Score(average='macro', num_classes=n_classes)
    f1 = f1_score(preds, targets)
    return acc, prec, rec, f1, loss

def save_metrics(metrics_path:str,date_and_time,model_name:str,time,
                 data_size:int,train_set:float,seed:int, metrics_dict:dict):
    # Info common to all metrics
    meta_data = {
        'Date Time': date_and_time,
        'Name': model_name,
        'Training Time': total_training_time,
        'Dataset Size': data_size,
        'Train Set': train_set,
        'Seed': seed
    }
    
    # Merge Metadata with Model-specific metrics
    metrics_dict = meta_data | metrics_dict
    
    # Convert to data frame
    metrics_frame = pd.DataFrame(columns=metrics_dict.keys())
    
    # Check if metrics csv exists
    metrics_frame = pd.DataFrame(metrics_dict, index=[0])
    
    if (os.path.exists(metrics_path)):
        df = pd.read_csv(metrics_path, index_col=0)
        df = pd.concat([df, metrics_frame], ignore_index=True)
        df.to_csv(metrics_path)
        
    else:
        metrics_frame.to_csv(metrics_path)
        
def get_best_metric(path, key):
    if (os.path.exists(metrics_path)):
        df = pd.read_csv(metrics_path, index_col=0)
        max_vals = df.max(numeric_only=True)
        return max_vals[key]
    else:
        print(f"Could not find {path}. Returning 0.")
        # Returning 0 so the metric will always be better if the file does not exist.
        return 0


# ## Data Pre-Processing
# 
# This section exctracts features and removes unnecessary/unwanted data.

# In[103]:


seed_everything(seed_value=seed)

data_raw = pd.DataFrame()
labels_raw = np.array([])
data_features = pd.DataFrame()
labels_features = np.array([])
# Extract features for the current dataset, create index labels, and combine .csv files.
print("Labels order:")
for root, dirs, files in os.walk(data_path):
    # Target the data set from 'Settings'
    if (root.endswith(dataset_info[current_dataset][0])):
        current_dataset_name = dataset_info[current_dataset][0]

        # Iterate over files that end with .csv
        files_iter = filter(lambda x: x.endswith('.csv'), files)

        for i, file in enumerate(files_iter):
            #if "Idle" in file: continue # Used for testing.
            labels_dict[i] = file.split('.')[0] # Align index and label.
            # Raw Data
            csv_file_path = data_path + dataset_info[current_dataset][0] + "/" + file
            dataset_frame = pd.read_csv(csv_file_path)
            # Filter out slow start and finish - Removes error values from writing files.
            slice_amt = (int)(dataset_frame.shape[0] * filter_portion)
            dataset_frame = dataset_frame.iloc[slice_amt:-slice_amt]
            if(use_raw):
                # Create aggregated data sets
                labels_raw = np.append(labels_raw, (np.zeros(dataset_frame.shape[0]) + i), axis=0)
                data_raw = pd.concat((data_raw, dataset_frame))
                data_raw.reset_index(drop = True, inplace = True)
            else:
                # Feature Data

                extracted_feats = feature_extraction(
                    dataset_frame,
                    # Set sample rate to the value that was used during recording.
                    sr = dataset_info[current_dataset][1],
                    window_length = window_length_feature_extraction,
                    hop_length = hop_length_feature_extraction
                )
                # Aggregated fatures from all classes
                data_features = pd.concat((data_features, extracted_feats))
                # Ensure df indeces ranges 0...n
                data_features.reset_index(drop = True, inplace = True)
                current_label_feat = np.zeros(extracted_feats.shape[0]) + i
                labels_features = np.append(labels_features, current_label_feat, axis=0)
                # Inspect label alignment:
                print(f"Label {i} = {file}")


# In[104]:


# Remove unwanted features
for col in data_raw.columns:
    for header in cols_to_ignore:
        if header in col:
            data_raw = data_raw.drop(col, axis=1)
 
# Remove columns not relevant for training (i.e. Timestamp)
for col in data_features.columns:
    for header in cols_to_ignore:
        if header in col:
            data_features = data_features.drop(col, axis=1)

# Remove extracted features, if any
for col in data_features.columns:
    for header in feats_to_ignore:
        if header in col:
            data_features = data_features.drop(col, axis=1)

# There is a leading whitespace in each header. This removes it.
for col in data_raw.columns:
    if (col[0] == " "):
        data_raw.rename(columns={col: col.strip()}, inplace=True)
        data_features.rename(columns={col: col.strip()}, inplace=True)

for col in data_features.columns:
    if (col[0] == " "):
        data_features.rename(columns={col: col.strip()}, inplace=True)


# Save features. 
processed_data_path = "./data/" + model_name
data_features.to_csv(processed_data_path + "_data_features.csv")
data_raw.to_csv(processed_data_path + "_data_raw.csv")
pd.Series(labels_features).to_csv(processed_data_path + "_labels_features.csv")
pd.Series(labels_raw).to_csv(processed_data_path + "_labels_raw.csv")


# ## Quick Data Inspection
# 
# Disabled for training.

# In[105]:


"""
print(data_features.columns)
%matplotlib widget
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection ="3d")

ax.scatter3D(data_features['Accelerometer X Mean'], 
    data_features['Accelerometer Y Mean'], 
    data_features['Accelerometer Z Mean'])

ax.scatter3D(data_features['Accelerometer X Var'], 
    data_features['Accelerometer Y Var'], 
    data_features['Accelerometer Z Var'])

ax.set_xlabel('Axis 1')
ax.set_ylabel('Axis 2')
ax.set_zlabel('Axis 3')

plt.show()"""


# ## Data Split
# 
# This part splits the data into training and development sets. Testing will happen directly on new controller input.

# In[106]:


if (use_raw):
    # Convert to tensor
    data_raw_tensor = torch.tensor([data_raw[col].astype('float64') for col in data_raw.columns]).T.double()
    lab_raw_tensor = torch.tensor(labels_raw).double()

    # For training using raw data
    raw_data_train, raw_data_test, raw_lab_train, raw_lab_test = sklearn.model_selection.train_test_split(
        data_raw_tensor,
        lab_raw_tensor,
        train_size=training_size,
        random_state=seed,
        stratify=labels_raw)
else:
    data_features_tensor = torch.tensor([data_features[col].astype('float64') for col in data_features.columns]).T.double()
    lab_features_tensor = torch.tensor(labels_features).double()
    # For training using features
    feat_data_train, feat_data_test, feat_lab_train, feat_lab_test = sklearn.model_selection.train_test_split(
        data_features_tensor,
        lab_features_tensor,
        train_size=training_size,
        random_state=seed,
        stratify=labels_features)


# ## Training

# In[119]:


# Create a model using either the raw data or features
if (use_raw):
   torch.manual_seed(seed)
   model = DSC_Classifier(len(raw_data_train[1]), len(labels_dict), hidden_layers, hidden_dims, dropout) # 16000, 8, 4
else:
   torch.manual_seed(seed)
   model = DSC_Classifier(len(feat_data_train[1]), len(labels_dict), hidden_layers, hidden_dims, dropout) # 128, 8, 4

# This stops training early if a certain criteria is met.
early_stopper = EarlyStopper(patience=early_stop_patience, learn_thresh=early_stop_thresh)
#early_stopper = EarlyStopper(gl_threshold = 50)

#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0) # Causes no learning.
optimizer = torch.optim.Adam(
   model.parameters(), 
   lr=learning_rate
)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
   optimizer, 
   max_lr = max_learning_rate, 
   epochs=epochs, 
   steps_per_epoch=4096
)

# Start Training
start_time_training = time.time()
for epoch in range(epochs):
   if (use_raw):
       # Train the model on raw data
       train(model, raw_data_train.float(), raw_lab_train.float(), optimizer, scheduler)
       # Report validation metrics
       accuracy, precision, recall, f1, train_loss = evaluate(
           model,
           raw_data_test.float(),
           raw_lab_test.float(),
           len(labels_dict)
       )
   else:
       # Train the model on extracted features
       loss = train(model, feat_data_train.float(), feat_lab_train.float(), optimizer, scheduler)
       # Report validation metrics
       accuracy, precision, recall, f1, val_loss = evaluate(
           model,
           feat_data_test.float(),
           feat_lab_test.float(),
           len(labels_dict)
       )

   # Set focus metric to which ever metric we currently find important.
   # Used to avoid having to duplicate code for testing.
   if(metric_names[current_metric] == 'accuracy'): focus_metric = accuracy
   if(metric_names[current_metric] == 'precision'): focus_metric = precision
   if(metric_names[current_metric] == 'recall'): focus_metric = recall
   if(metric_names[current_metric] == 'f1'): focus_metric = f1

   # Update recorded data. 
   model_specific_metrics = {
       "Mean": any(["Mean" in col.split(" ") for col in data_features.columns]),
       "Variance": any(["Var" in col.split(" ") for col in data_features.columns]),
       "RMS": any(["RMS" in col.split(" ") for col in data_features.columns]),
       "Frame Size": window_length_feature_extraction,
       "Hop Length": hop_length_feature_extraction,
       "Accuracy":accuracy.item(),
       "Precision":precision.item(),
       "Recall":recall.item(),
       "F1":f1.item(),
       "Epochs":epoch,
       "Hidden Dimensions":hidden_dims,
       "Hidden Layers":hidden_layers,
       "Dropout":dropout,
       "Learning Rate":learning_rate,
       "Max Learning Rate":max_learning_rate
   }

   print(f'Epoch: {epoch}\tTrain Loss: {loss} \tValidation Loss: {val_loss}  \t{metric_names[current_metric].title()}: {focus_metric:.3f}')
   # Generalization loss is causing bugs.
   #print(f'\t\t  Generalization Loss: {early_stopper.get_generalization_loss():.3f}')

   # If the main metric is not improving, stop learning.
   if(early_stopper.stop_check(focus_metric)):
       model_specific_metrics["Epochs"] = epoch
       break

   #if(early_stopper.generalization_stop(val_loss)):
       #model_specific_metrics["Epochs"] = epoch
       #break

total_training_time = time.time() - start_time_training

# Save the model in ONNX format
out_model_name = model_name
# The dummy input shows the model how the real input should be formatted.
# 1, n_features
dummy_input = torch.randn(1, len(feat_data_train[1]))

# Export model in ONNX format.
torch.onnx.export(model,
   dummy_input,               # Desired inference input shape
   best_model_paths + out_model_name + ".onnx",  # Model export path + name
   export_params=True,        # Store weights with the model
   opset_version=9,           # Unity requires onnx 9
   do_constant_folding=True,  # whether to execute constant folding for optimization
   input_names = ['Input'],   # Used to identify layers during debugging
   output_names = ['Logits']  # Used to identify layers during debugging
)

# Save Metrics
now = datetime.now()

# DateTime formatting
date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")

save_metrics(
   metrics_path,
   date_and_time,
   model_name,
   total_training_time,
   feat_lab_train.shape[0],
   training_size,
   seed,
   model_specific_metrics
)


# In[ ]:




