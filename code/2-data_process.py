import pandas as pd
import numpy as np
import cmapPy as cmap
from cmapPy.pandasGEXpress import parse
from cmapPy.pandasGEXpress import write_gctx
import os
import pickle
import torch
from torchdrug import core, datasets, tasks, models,data
from rdkit import Chem

def save_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"finish save {filepath}")

def load_pkl(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"finish load {filepath}")
    return obj




### process cmap
cell_info1=pd.read_table('./source/GSE70138/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz')
gene_info1=pd.read_table('./source/GSE70138/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz')
pert_info1=pd.read_table('./source/GSE70138/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz')
inst_info1=pd.read_table('./source/GSE70138/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz')
sig_info1=pd.read_table('./source/GSE70138/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz')
cell_info2=pd.read_table('./source/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz')
gene_info2=pd.read_table('./source/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz')
pert_info2=pd.read_table('./source/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz')
inst_info2=pd.read_table('./source/GSE92742/GSE92742_Broad_LINCS_inst_info.txt.gz')
sig_info2=pd.read_table('./source/GSE92742/GSE92742_Broad_LINCS_sig_info.txt.gz')
inst_info1 = inst_info1[inst_info1['pert_type'].isin(['trt_cp','ctl_vehicle'])].reset_index(drop=True)
inst_info2 = inst_info2[inst_info2['pert_type'].isin(['trt_cp','ctl_vehicle'])].reset_index(drop=True)
data_df1=parse.parse('./source/GSE70138/GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx',cid=inst_info1['inst_id']).data_df
data_df2_epsilon = parse.parse('./source/GSE92742/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx', ridx=[0])
valid_inst_ids_epsilon = list(set(inst_info2['inst_id']) & set(data_df2_epsilon.data_df.columns))
data_df2 = parse.parse('./source/GSE92742/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx', cid=valid_inst_ids_epsilon).data_df
inst_info2 = inst_info2[inst_info2['inst_id'].isin(valid_inst_ids_epsilon)].reset_index(drop=True)
data_df = data_df1.join(data_df2,how='inner')
inst_info2 = inst_info2.rename(columns={
    'rna_plate': 'det_plate',
    'rna_well': 'det_well'
})
inst_info1 = inst_info1.drop('pert_mfc_id', axis=1)
inst_info = pd.concat([inst_info1, inst_info2], axis=0, ignore_index=True)
trt_cp_cells = set(inst_info[inst_info['pert_type'] == 'trt_cp']['cell_id'].unique())
ctl_vehicle_cells = set(inst_info[inst_info['pert_type'] == 'ctl_vehicle']['cell_id'].unique())
all_cells_have_control = trt_cp_cells.issubset(ctl_vehicle_cells)
if not all_cells_have_control:
    missing_cells = trt_cp_cells - ctl_vehicle_cells
    print("trt_cp cell without ctl_vehicle:")
    print(missing_cells)
inst_info=inst_info[inst_info['cell_id']!='MCH58'].reset_index(drop=True)
data_df=data_df[inst_info['inst_id']]
gene_symbols_id=pd.read_csv('./source/herb_RNA_new/gene_symbols_id_977.csv')
gene_symbols_id['pr_gene_id'] = gene_symbols_id['pr_gene_id'].astype(str)
data_df = data_df.loc[gene_symbols_id['pr_gene_id']]
data_df.index = gene_symbols_id['pr_gene_symbol']




### calculate compound feature
compound_df=pd.read_csv('./source/first_processed_cmd2.csv')
compound_df=compound_df.dropna(subset='canonical_smiles').drop_duplicates(subset=['pert_id'])
compound_df=compound_df[compound_df['canonical_smiles']!='restricted']
compound_df = compound_df[compound_df['CID'] != 0].reset_index(drop=True)
inst_info = inst_info[inst_info['pert_id'].isin(compound_df['pert_id']) | inst_info['pert_id'].isin(['DMSO', 'PBS', 'H2O'])].reset_index(drop=True)
inst_info = inst_info[~((inst_info['pert_type']=='trt_cp') & (inst_info['pert_dose']==0))].reset_index(drop=True)
data_df = data_df[inst_info['inst_id']]
compound_df.to_csv('./processed_data_0107/compound_df.csv', index=False)
# load model
dataset = datasets.ZINC2m("./molecule-datasets/", node_feature="pretrain", edge_feature="pretrain")
gin_model = models.GIN(input_dim=dataset.node_feature_dim,
                       hidden_dims=[300, 300, 300, 300, 300],
                       edge_input_dim=dataset.edge_feature_dim,
                       batch_norm=True, readout="mean")
model = models.InfoGraph(gin_model, separate_model=False)
model.eval()
task = tasks.Unsupervised(model)
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer, gpus=None, batch_size=1024)
solver.load('ZINC2m_gin_infograph.pth')
# creat test set
mols=[]
for x in compound_df['canonical_smiles'].tolist():
    temp = Chem.MolFromSmiles(x)
    temp = data.Molecule.from_molecule(temp, node_feature="pretrain", edge_feature="pretrain")
    mols.append(temp)
test_set=[]
for x in mols:
    test_set.append({'graph':x})
batch = data.graph_collate(test_set)
# predict
with torch.no_grad():
    pred = task.predict(batch)
# save feature
graph_feature=np.array(pred['graph_feature']).tolist()
compound_df['Compound_feature']=graph_feature
compound_feature_dict = dict(zip(compound_df['pert_id'], compound_df['Compound_feature']))
save_pkl(compound_feature_dict,'./processed_data_0107/compound_feature_dict.pkl')




### process herb data
herb_water_data=pd.read_csv('./source/herb_RNA_new/herb_water_data.csv',index_col=0)
herb_dmso_data=pd.read_csv('./source/herb_RNA_new/herb_dmso_data.csv',index_col=0)
herb_water_info=pd.read_csv('./source/herb_RNA_new/herb_water_info.csv')
herb_dmso_info=pd.read_csv('./source/herb_RNA_new/herb_dmso_info.csv')
data_df_geo=data_df
inst_info_geo=inst_info
herb_df_geo=herb_water_data
herb_meta_geo=herb_water_info
herb_control_df_geo=herb_dmso_data
herb_control_meta_geo=herb_dmso_info

save_pkl(data_df_geo,'./processed_data_0107/data_df_geo.pkl')
save_pkl(inst_info_geo,'./processed_data_0107/inst_info_geo.pkl')
save_pkl(herb_df_geo,'./processed_data_0107/herb_df_geo.pkl')
save_pkl(herb_meta_geo,'./processed_data_0107/herb_meta_geo.pkl')
save_pkl(herb_control_df_geo,'./processed_data_0107/herb_control_df_geo.pkl')
save_pkl(herb_control_meta_geo,'./processed_data_0107/herb_control_meta_geo.pkl')




### calculate herb feature
data_df=load_pkl('./processed_data_0107/data_df_geo.pkl')
inst_info=load_pkl('./processed_data_0107/inst_info_geo.pkl')
herb_df=load_pkl('./processed_data_0107/herb_df_geo.pkl')
herb_meta=load_pkl('./processed_data_0107/herb_meta_geo.pkl')
herb_info1 = pd.read_table("./source/HERB2.0/info/HERB_herb_info_v2.txt").dropna(subset=['Herb_latin_name'])
herb_info1['Herb_latin_name'] = herb_info1['Herb_latin_name'].str.title()
herb_names1=list(set(herb_info1['Herb_latin_name']))
herb_names=list(herb_meta['herb'].unique())
herb_compound_df = pd.read_csv("./source/HERB2.0/herb/herb_ingredient.csv")
herb_compound_df['PubChem_id']=herb_compound_df['PubChem_id'].astype(int)
herb_compound_df=herb_compound_df.rename(columns={'PubChem_id':'cid','Ingredient_Smile':'smiles'})
herb_compound_df=herb_compound_df.dropna(subset=['cid','smiles'])
herb_compound_df=pd.merge(herb_compound_df,herb_info1[['Herb_ID','Herb_latin_name']],on='Herb_ID',how='inner')
for x in herb_names:
    if x not in herb_names1:
        print(x)
herb_meta['Herb_latin_name']=herb_meta['herb']
herb_meta['Herb_cn_name']=herb_meta['herb']
herb_meta.loc[herb_meta['herb']=='Panax Ginseng','Herb_latin_name']='Ginseng Radix Et Rhizoma'
herb_meta.loc[herb_meta['herb']=='Panax Ginseng','Herb_cn_name']='人参'
herb_meta.loc[herb_meta['herb']=='Atractylodis Rhizoma','Herb_cn_name']='苍术'
herb_meta.loc[herb_meta['herb']=='Astragali Radix','Herb_cn_name']='黄芪'
herb_meta.loc[herb_meta['herb']=='Cinnamomi Cortex','Herb_cn_name']='肉桂'
herb_meta.loc[herb_meta['herb']=='Aconiti Lateralis Radix Preparata','Herb_latin_name']='Aconiti Lateralis Radix Praeparata'
herb_meta.loc[herb_meta['herb']=='Aconiti Lateralis Radix Preparata','Herb_cn_name']='附子'
herb_meta.loc[herb_meta['herb']=='Poria Cocos','Herb_latin_name']='Poria'
herb_meta.loc[herb_meta['herb']=='Poria Cocos','Herb_cn_name']='茯苓'
herb_meta.loc[herb_meta['herb']=='Glycyrrhizae Radix','Herb_latin_name']='Glycyrrhizae Radix Et Rhizoma'
herb_meta.loc[herb_meta['herb']=='Glycyrrhizae Radix','Herb_cn_name']='甘草'
herb_meta.loc[herb_meta['herb']=='Angelicae Gigantis Radix','Herb_latin_name']='Angelica Gigas'
herb_meta.loc[herb_meta['herb']=='Angelicae Gigantis Radix','Herb_cn_name']='朝鲜当归'
herb_meta.loc[herb_meta['herb']=='Paeoniae Radix','Herb_latin_name']='Paeoniae Radix Alba'
herb_meta.loc[herb_meta['herb']=='Paeoniae Radix','Herb_cn_name']='白芍'
herb_meta.loc[herb_meta['herb']=='Cnidii Rhizoma','Herb_latin_name']='Chuanxiong Rhizoma'
herb_meta.loc[herb_meta['herb']=='Cnidii Rhizoma','Herb_cn_name']='川芎'
herb_meta.loc[herb_meta['herb']=='Rehmanniae Radix Preparata','Herb_latin_name']='Rehmanniae Radix Praeparata'
herb_meta.loc[herb_meta['herb']=='Rehmanniae Radix Preparata','Herb_cn_name']='熟地黄'
herb_meta=pd.merge(herb_meta,herb_info1[['Herb_latin_name','Herb_cn_name','Herb_ID']],on=['Herb_latin_name','Herb_cn_name'],how='inner')
herb_compound_df=pd.merge(herb_compound_df,herb_meta[['Herb_latin_name','Herb_cn_name','Herb_ID']].drop_duplicates(),on=['Herb_latin_name','Herb_cn_name','Herb_ID'],how='inner')
save_pkl(herb_compound_df,'./processed_data_0107/herb_compound_df_geo.pkl')
save_pkl(herb_meta,'./processed_data_0107/herb_meta_geo.pkl')
# load model
dataset = datasets.ZINC2m("./molecule-datasets/", node_feature="pretrain", edge_feature="pretrain")
gin_model = models.GIN(input_dim=dataset.node_feature_dim,
                       hidden_dims=[300, 300, 300, 300, 300],
                       edge_input_dim=dataset.edge_feature_dim,
                       batch_norm=True, readout="mean")
model = models.InfoGraph(gin_model, separate_model=False)
model.eval()
task = tasks.Unsupervised(model)
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer, gpus=None, batch_size=1024)
solver.load('ZINC2m_gin_infograph.pth')
# create test set
mols=[]
for x in herb_compound_df['smiles'].tolist():
    temp = Chem.MolFromSmiles(x)
    temp = data.Molecule.from_molecule(temp, node_feature="pretrain", edge_feature="pretrain")
    mols.append(temp)
test_set=[]
for x in mols:
    test_set.append({'graph':x})
batch = data.graph_collate(test_set)
# predict
with torch.no_grad():
    pred = task.predict(batch)
# save
graph_feature=np.array(pred['graph_feature']).tolist()
herb_compound_df['Compound_feature']=graph_feature
herb_compound_feature_dict = dict(zip(herb_compound_df['Ingredient_id'], herb_compound_df['Compound_feature']))
save_pkl(herb_compound_feature_dict,'./processed_data_0107/herb_compound_feature_dict_geo.pkl')
herb_feature_dict = {}
for herb_name in herb_compound_df['Herb_latin_name'].unique():
    print(herb_name)
    compound_ids = herb_compound_df[herb_compound_df['Herb_latin_name'] == herb_name]['Ingredient_id'].tolist()
    compound_features = []
    ccc=0
    for compound_id in compound_ids:
        ccc+=1
        compound_features.append(herb_compound_feature_dict[compound_id])
    print(ccc)
    herb_feature = np.mean(compound_features, axis=0)
    # herb_feature = np.max(compound_features, axis=0)
    herb_feature_dict[herb_name] = herb_feature.tolist()
print(f"calculate {len(herb_feature_dict)} herbs")
print(f"each dimension: {len(list(herb_feature_dict.values())[0])}")
save_pkl(herb_feature_dict, './processed_data_0107/herb_feature_dict_geo_average_pooling.pkl')