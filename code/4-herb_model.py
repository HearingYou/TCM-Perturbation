import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.stats import pearsonr

def save_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"finish save {filepath}")

def load_pkl(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"finish load {filepath}")
    return obj



### random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
set_seed(0)
g = torch.Generator()
g.manual_seed(0)




#### load and standardize data
data_df=load_pkl('./processed_data_0107/data_df_geo.pkl')
compound_feature_dict=load_pkl('./processed_data_0107/compound_feature_dict.pkl')
inst_info=load_pkl('./processed_data_0107/inst_info_geo.pkl')
inst_info_trt=inst_info[inst_info['pert_type']=='trt_cp'].reset_index(drop=True)
inst_info_ctl=inst_info[inst_info['pert_type']=='ctl_vehicle'].reset_index(drop=True)
herb_meta=load_pkl('./processed_data_0107/herb_meta_geo.pkl').rename(columns={'dose':'pert_dose','time':'pert_time'})
herb_feature_dict=load_pkl('./processed_data_0107/herb_feature_dict_geo.pkl')
herb_df=load_pkl('./processed_data_0107/herb_df_geo.pkl')
herb_control_meta=load_pkl('./processed_data_0107/herb_control_meta_geo.pkl')
herb_control_df=load_pkl('./processed_data_0107/herb_control_df_geo.pkl')
herb_df=pd.concat([herb_df,herb_control_df],axis=1)
unique_herbs = herb_meta['Herb_cn_name'].nunique()
unique_cell_lines = herb_meta['cellline'].nunique()
unique_combinations = herb_meta[['Herb_cn_name', 'cellline']].drop_duplicates().shape[0]
# gene
gene_scaler = StandardScaler()
data_df_log = np.log10(data_df + 1)
data_df_scaled = gene_scaler.fit_transform(data_df_log.T).T
data_df_scaled = pd.DataFrame(data=data_df_scaled, index=data_df.index, columns=data_df.columns)
# dose
dose_scaler = StandardScaler()
inst_info_trt['pert_dose'] = np.log10(inst_info_trt['pert_dose'])
inst_info_trt['pert_dose'] = dose_scaler.fit_transform(inst_info_trt[['pert_dose']])
# time
time_scaler = StandardScaler()
inst_info_trt['pert_time'] = time_scaler.fit_transform(inst_info_trt[['pert_time']])
# compound
compound_ids = list(compound_feature_dict.keys())
features_array = np.array([compound_feature_dict[cid] for cid in compound_ids])
compound_scaler = StandardScaler()
scaled_features = compound_scaler.fit_transform(features_array)
compound_feature_dict_scaled = {compound_ids[i]: scaled_features[i] for i in range(len(compound_ids))}
# herb
# dose
dose_scaler_herb = StandardScaler()
herb_meta['pert_dose'] = np.log10(herb_meta['pert_dose'])
herb_meta['pert_dose'] = dose_scaler_herb.fit_transform(herb_meta[['pert_dose']])
# time
time_scaler_herb=time_scaler_chem
herb_meta['pert_time'] = time_scaler_herb.transform(herb_meta[['pert_time']])
# gene
gene_scaler_herb = StandardScaler()
herb_df_log = np.log10(herb_df + 1)
herb_df_scaled = gene_scaler_herb.fit_transform(herb_df_log.T).T
herb_df_scaled = pd.DataFrame(data=herb_df_scaled, index=herb_df.index, columns=herb_df.columns)
# herb
compound_scaler_herb = StandardScaler()
herb_ids = list(herb_feature_dict.keys())
herb_features_array = np.array([herb_feature_dict[hid] for hid in herb_ids])
herb_scaled_features = compound_scaler_herb.fit_transform(herb_features_array)
herb_feature_dict_scaled = {herb_ids[i]: herb_scaled_features[i] for i in range(len(herb_ids))}




### model
class SimpleSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):  
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ v
        return out
class CompoundEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x
class ControlEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,hidden_dim3, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))
        x = self.fc4(x)
        return x
class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,hidden_dim3, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.attn = SimpleSelfAttention(hidden_dim1)
        self.norm = nn.LayerNorm(hidden_dim1)   # 归一化，训练更稳定
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = x.unsqueeze(1)
        x = self.attn(x).squeeze(1)
        x = self.norm(x)  
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))
        x = self.fc4(x)
        return x
class CVAE(nn.Module):
    def __init__(self, 
                 target_dim, 
                 compound_dim, 
                 control_dim, 
                 compound_latent_dim, 
                 control_latent_dim):
        super().__init__()
        self.compound_encoder = CompoundEncoder(
            input_dim=compound_dim,
            hidden_dim1=256,
            hidden_dim2=128,
            out_dim=compound_latent_dim
        )
        self.control_encoder = ControlEncoder(
            input_dim=control_dim,
            hidden_dim1=512,
            hidden_dim2=256,
            hidden_dim3=128,
            out_dim=control_latent_dim
        )
        decoder_input_dim = compound_latent_dim + 2 + control_latent_dim
        self.decoder = AttentionDecoder(
            input_dim=decoder_input_dim,
            hidden_dim1=128,
            hidden_dim2=256,
            hidden_dim3=512,
            out_dim=target_dim
        )

    def encode_compound(self, compound):
        return self.compound_encoder(compound)

    def encode_control(self, control):
        return self.control_encoder(control)

    def decode(self, combined):
        return self.decoder(combined)

    def forward(self, compound, dose, time, control_x):
        compound_encoded = self.encode_compound(compound)
        control_encoded = self.encode_control(control_x)
        combined = torch.cat([compound_encoded, dose.unsqueeze(1), time.unsqueeze(1), control_encoded], dim=1)
        output = self.decode(combined)
        return output




### dataset
class CVAEDataset_Herb(Dataset):
    def __init__(self, herb_df_scaled, herb_feature_dict_scaled, herb_meta, herb_control_meta):
        self.herb_df = herb_df_scaled
        self.herb_feature_dict = herb_feature_dict_scaled
        self.herb_meta = herb_meta
        self.herb_control_meta = herb_control_meta

    def __len__(self):
        return len(self.herb_meta)

    def __getitem__(self, idx):
        trt_sample_info = self.herb_meta.iloc[idx]
        trt_inst_id = trt_sample_info['sample']
        control_cellline = trt_sample_info['control_cellline']
        ctl_df=self.herb_control_meta[self.herb_control_meta['control_cellline'] == control_cellline].sample(n=1)
        ctl_inst_id = ctl_df['sample'].iloc[0]
        control_transcriptome = self.herb_df[ctl_inst_id].values
        perturbed_transcriptome = self.herb_df[trt_inst_id].values
        compound_feature = self.herb_feature_dict[trt_sample_info['Herb_latin_name']]
        dose = trt_sample_info['pert_dose']
        time = trt_sample_info['pert_time']
        return {
            'compound_feature': compound_feature,
            'dose': dose,
            'time': time,
            'perturbed_transcriptome': perturbed_transcriptome,
            'control_transcriptome': control_transcriptome
        }
condition_and_cell_groups = herb_meta.groupby(['Herb_latin_name', 'pert_dose', 'pert_time', 'cellline'])
unique_groups = list(condition_and_cell_groups.groups.keys())
np.random.shuffle(unique_groups)
train_conditions_count = int(len(unique_groups) * 0.7)
train_conditions = set(unique_groups[:train_conditions_count])  # 使用集合而不是列表，加速查找
val_conditions = set(unique_groups[train_conditions_count:])    # 使用集合而不是列表，加速查找
condition_to_split = {}
for condition in train_conditions:
    condition_to_split[condition] = 'train'
for condition in val_conditions:
    condition_to_split[condition] = 'val'
herb_meta['condition_tuple'] = list(zip(
    herb_meta['Herb_latin_name'], 
    herb_meta['pert_dose'], 
    herb_meta['pert_time'], 
    herb_meta['cellline']
))
herb_meta['split'] = herb_meta['condition_tuple'].map(condition_to_split)
train_indices = herb_meta[herb_meta['split'] == 'train'].index
val_indices = herb_meta[herb_meta['split'] == 'val'].index
train_herb_info_trt = herb_meta.iloc[train_indices]
val_herb_info_trt = herb_meta.iloc[val_indices]
train_dataset_herb = CVAEDataset_Herb(herb_df_scaled, herb_feature_dict_scaled, train_herb_info_trt, herb_control_meta)
val_dataset_herb = CVAEDataset_Herb(herb_df_scaled, herb_feature_dict_scaled, val_herb_info_trt, herb_control_meta)




### train and validation
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc='Training', dynamic_ncols=True) as pbar:
        for batch in pbar:
            compound = batch['compound_feature'].float().to(device)
            dose = batch['dose'].float().to(device)
            time = batch['time'].float().to(device)
            control_x = batch['control_transcriptome'].float().to(device)
            perturbed_x = batch['perturbed_transcriptome'].float().to(device)
            optimizer.zero_grad()
            output = model(compound, dose, time, control_x)
            loss = torch.nn.functional.mse_loss(output, perturbed_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item()
            })
    avg_loss = total_loss / len(train_loader)
    return avg_loss
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    for batch in val_loader:
        compound = batch['compound_feature'].float().to(device)
        dose = batch['dose'].float().to(device)
        time = batch['time'].float().to(device)
        control_x = batch['control_transcriptome'].float().to(device)
        perturbed_x = batch['perturbed_transcriptome'].float().to(device)
        output = model(compound, dose, time, control_x)
        loss = torch.nn.functional.mse_loss(output, perturbed_x)
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss
batch_size = 64
learning_rate = 1e-3
num_epochs = 500
target_dim = 977 
compound_dim = 300 
control_dim = 977 
compound_latent_dim = 32 
control_latent_dim = 64 
train_loader_herb = DataLoader(train_dataset_herb, batch_size=batch_size, shuffle=True, num_workers=24,pin_memory=True,persistent_workers=True,prefetch_factor=8)
val_loader_herb = DataLoader(val_dataset_herb, batch_size=batch_size, shuffle=False, num_workers=24,pin_memory=True,persistent_workers=True,prefetch_factor=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('./train_data_0107_geo/best_model.pt')
model = CVAE(   
    target_dim=target_dim,
    compound_dim=compound_dim,
    control_dim=control_dim,
    compound_latent_dim=compound_latent_dim,
    control_latent_dim=control_latent_dim,
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
best_val_loss = float('inf')
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=20, min_delta=1e-4) 
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader_herb, optimizer, device)
    val_loss = validate(model, val_loader_herb, device)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}\n")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, './train_data_0107_geo/best_model_herb.pt')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break




### calculate metrics
checkpoint = torch.load('./train_data_0107_geo/best_model_herb.pt')
model = CVAE(   
    target_dim=target_dim,
    compound_dim=compound_dim,
    control_dim=control_dim,
    compound_latent_dim=compound_latent_dim,
    control_latent_dim=control_latent_dim,
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
all_targets = []
all_generated = []
with torch.no_grad():
    for batch in tqdm(val_loader_herb, desc="processing"):
        compound = batch['compound_feature'].float().to(device)
        dose = batch['dose'].float().to(device)
        time = batch['time'].float().to(device)
        control_x = batch['control_transcriptome'].float().to(device)
        perturbed_x = batch['perturbed_transcriptome'].float().to(device)
        output = model(compound, dose, time, control_x)
        all_targets.append(perturbed_x.cpu().numpy())
        all_generated.append(output.cpu().numpy())
all_targets = np.vstack(all_targets)
all_generated = np.vstack(all_generated)
y_val=all_targets
y_val_pred=all_generated
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred.flatten()
pearson_corr, pearson_p = pearsonr(y_val_flat, y_val_pred_flat)
mse = mean_squared_error(y_val_flat, y_val_pred_flat)
r2 = r2_score(y_val_flat, y_val_pred_flat)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Pearson corr: {pearson_corr:.4f}")
print(f"Pearson p: {pearson_p:.4f}")