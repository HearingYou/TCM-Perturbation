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





### select control group
def compute_pca_embedding(data_df_scaled, n_pca=50):
    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(data_df_scaled.T.values)
    inst_ids = data_df_scaled.columns.tolist()
    inst_id_to_pca = {inst_id: data_pca[i] for i, inst_id in enumerate(inst_ids)}
    return inst_id_to_pca
def process_cell_id(cell_id, inst_info_trt, inst_info_ctl, inst_id_to_pca):
    mapping = {}
    trt_ids = inst_info_trt[inst_info_trt['cell_id'] == cell_id]['inst_id'].tolist()
    ctl_ids = inst_info_ctl[inst_info_ctl['cell_id'] == cell_id]['inst_id'].tolist()
    if len(ctl_ids) == 0:
        return mapping
    ctl_embeddings = np.stack([inst_id_to_pca[ctl_id] for ctl_id in ctl_ids])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ctl_embeddings)
    for trt_id in trt_ids:
        trt_embedding = inst_id_to_pca[trt_id].reshape(1, -1)
        distance, index = nbrs.kneighbors(trt_embedding)
        nearest_ctl_id = ctl_ids[index[0][0]]
        mapping[trt_id] = nearest_ctl_id
    return mapping
def build_nearest_control_map(data_df_scaled, inst_info_trt, inst_info_ctl, n_pca=50, n_jobs=30):
    inst_id_to_pca = compute_pca_embedding(data_df_scaled, n_pca=n_pca)
    cell_ids = inst_info_trt['cell_id'].unique().tolist()
    has_control_count = 0
    no_control_count = 0
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_id)(cell_id, inst_info_trt, inst_info_ctl, inst_id_to_pca)
        for cell_id in tqdm(cell_ids, desc="processing")
    )
    treatment_to_control_map = {}
    for cell_id, res in zip(cell_ids, results):
        if len(res) > 0:
            has_control_count += 1
            treatment_to_control_map.update(res)
        else:
            no_control_count += 1
    return treatment_to_control_map
treatment_to_control_map = build_nearest_control_map(data_df_scaled, inst_info_trt, inst_info_ctl, n_pca=50, n_jobs=16)




### create dataset
class CVAEDataset(Dataset):
    def __init__(self, data_df_scaled, compound_feature_dict_scaled, inst_info_trt, inst_info_ctl, treatment_to_control_map):
        self.data_df = data_df_scaled
        self.compound_feature_dict = compound_feature_dict_scaled
        self.inst_info_trt = inst_info_trt
        self.inst_info_ctl = inst_info_ctl
        self.treatment_to_control_map = treatment_to_control_map

    def __len__(self):
        return len(self.inst_info_trt)

    def __getitem__(self, idx):
        trt_sample_info = self.inst_info_trt.iloc[idx]
        trt_inst_id = trt_sample_info['inst_id']
        cell_id = trt_sample_info['cell_id']
        ctl_inst_id = self.treatment_to_control_map[trt_inst_id]
        perturbed_transcriptome = self.data_df[trt_inst_id].values
        control_transcriptome = self.data_df[ctl_inst_id].values
        compound_feature = self.compound_feature_dict[trt_sample_info['pert_id']]
        dose = trt_sample_info['pert_dose']
        time = trt_sample_info['pert_time']
        return {
            'compound_feature': compound_feature,
            'dose': dose,
            'time': time,
            'perturbed_transcriptome': perturbed_transcriptome,
            'control_transcriptome': control_transcriptome
        }
condition_and_cell_groups = inst_info_trt.groupby(['pert_id', 'pert_dose', 'pert_time', 'cell_id'])
unique_groups = list(condition_and_cell_groups.groups.keys())
np.random.shuffle(unique_groups)
train_conditions_count = int(len(unique_groups) * 0.8)
train_conditions = set(unique_groups[:train_conditions_count])  # 使用集合而不是列表，加速查找
val_conditions = set(unique_groups[train_conditions_count:])    # 使用集合而不是列表，加速查找
condition_to_split = {}
for condition in train_conditions:
    condition_to_split[condition] = 'train'
for condition in val_conditions:
    condition_to_split[condition] = 'val'
inst_info_trt['condition_tuple'] = list(zip(
    inst_info_trt['pert_id'], 
    inst_info_trt['pert_dose'], 
    inst_info_trt['pert_time'], 
    inst_info_trt['cell_id']
))
inst_info_trt['split'] = inst_info_trt['condition_tuple'].map(condition_to_split)
train_indices = inst_info_trt[inst_info_trt['split'] == 'train'].index
val_indices = inst_info_trt[inst_info_trt['split'] == 'val'].index
train_inst_info_trt = inst_info_trt.iloc[train_indices]
val_inst_info_trt = inst_info_trt.iloc[val_indices]
train_dataset = CVAEDataset(data_df_scaled, compound_feature_dict_scaled, train_inst_info_trt, inst_info_ctl, treatment_to_control_map)
val_dataset = CVAEDataset(data_df_scaled, compound_feature_dict_scaled, val_inst_info_trt, inst_info_ctl, treatment_to_control_map)




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
batch_size = 2048*8
learning_rate = 1e-3
num_epochs = 500
target_dim = 977  
compound_dim = 300 
control_dim = 977 
compound_latent_dim = 32 
control_latent_dim = 64 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24,pin_memory=True,persistent_workers=True,prefetch_factor=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24,pin_memory=True,persistent_workers=True,prefetch_factor=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CVAE(
    target_dim=target_dim,
    compound_dim=compound_dim,
    control_dim=control_dim,
    compound_latent_dim=compound_latent_dim,
    control_latent_dim=control_latent_dim,
).to(device)
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
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
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
        }, './train_data_0107_geo/best_model.pt')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
save_pkl({'train_losses': train_losses, 'val_losses': val_losses}, './train_data_0107_geo/losses.pkl')




### calculate metrics
checkpoint = torch.load('./train_data_0107_geo/best_model.pt')
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
    for batch in tqdm(val_loader, desc="processing"):
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
mse = mean_squared_error(y_val, y_val_pred, multioutput='uniform_average')
r2 = r2_score(y_val, y_val_pred, multioutput='uniform_average')
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred.flatten()
pearson_corr, pearson_p = pearsonr(y_val_flat, y_val_pred_flat)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Pearson corr: {pearson_corr:.4f}")
print(f"Pearson p: {pearson_p:.4f}")