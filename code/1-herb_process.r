library(GEOquery)
library(clusterProfiler)
library(org.Hs.eg.db)
library(stringr)
library(data.table)
library(openxlsx)
library(biomaRt)
library(tidyverse)




### dowload and preprocess herb rna data
Sys.setenv("VROOM_CONNECTION_SIZE" = 99999999)
geo = c('GSE273185', 'GSE245896', 'GSE245898', 'GSE245903', 'GSE245907', 'GSE245910', 'GSE245911', 'GSE244702', 'GSE244703', 'GSE244704', 'GSE244705', 
  'GSE244706', 'GSE244689', 'GSE244690', 'GSE244691', 'GSE244692', 'GSE244693', 'GSE244682', 'GSE244683', 'GSE244684', 'GSE244685', 'GSE244686')
sample.info.list = list()
# download
for(i in 1:length(geo)){
    gse = geo[i]
    gse <- getGEO(gse, GSEMatrix = TRUE)
    sample_info <- pData(phenoData(gse[[1]]))
    sample.info.list[[i]] = sample_info[,c('title', 'description', 'cell line:ch1', 'treatment:ch1','time:ch1')]
    names(sample.info.list)[i] = gse
}
data = rbindlist(sample.info.list)
# remove prescription
data_fufang=data[grep('tang', data$`treatment:ch1`), ]
data=data[-grep('tang', data$`treatment:ch1`), ]
# DMSO
data_dmso=data[grep('DMSO', data$`treatment:ch1`), ]
data=data[-grep('DMSO', data$`treatment:ch1`), ]
print(nrow(data_dmso))
print(nrow(data))
# remove ethanol
data_ethanol=data[grep('ethanol', data$`treatment:ch1`), ]
data=data[-grep('ethanol', data$`treatment:ch1`), ]
print(nrow(data_ethanol))
print(nrow(data))
# remove wortmannin
data_wortmannin=data[grep('wortmannin', data$`treatment:ch1`), ]
data=data[-grep('wortmannin', data$`treatment:ch1`), ]
print(nrow(data_wortmannin))
print(nrow(data))
# remove water extract
data_water=data[grep('water extract', data$`treatment:ch1`), ]
data=data[-grep('water extract', data$`treatment:ch1`), ]
print(nrow(data_water))
print(nrow(data))

gse.files = list.files('./data')
for(i in 1:length(geo)){
    gse = geo[i]
    ff = gse.files[grep(gse, gse.files)]
    if(length(ff) < 1){
        print(gse)
    }
}
for(i in 1:length(geo)){
    gse = geo[i]
    ff = gse.files[grep(gse, gse.files)]
    fff = unlist(strsplit(ff, split = '_'))
    cell.line = fff[length(fff)]
    cell.line = gsub('\\.txt.gz', '', cell.line)
    dt = read.table(paste0('./data/', ff), header = 1, check.names = F)
    if('Gene' %in% colnames(dt)){
        rownames(dt) = dt$Gene
        dt = dt[, -1]
    }
    if(cell.line == 'ACD'){
        cc = lapply(strsplit(colnames(dt), split = '-'), function(x){x[1]})
        colnames(dt) = paste0(cc, '_', colnames(dt))
    }else{
        colnames(dt) = paste0(cell.line, '_', colnames(dt))
    }
    if(i == 1){
        dtt = dt
    }else{
        dtt = cbind(dtt, dt)
    }
}
data_water$sample = data_water$description
data_water$sample = gsub('sample ', '', data_water$sample)
data_water$sample = paste0(data_water$`cell.line:ch1`, '_',data_water$sample)
data_dmso$sample = data_dmso$description
data_dmso$sample = gsub('sample ', '', data_dmso$sample)
data_dmso$sample = paste0(data_dmso$`cell.line:ch1`, '_',data_dmso$sample)
dt_water=dtt[, data_water$sample]
dt_dmso=dtt[, data_dmso$sample]
# id convert
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
gene_ids <- rownames(dt_dmso)
gene_symbols <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol","entrezgene_id"), 
                      filters = "ensembl_gene_id", 
                      values = gene_ids, 
                      mart = ensembl)
write.csv(gene_symbols, "gene_symbols_id.csv", row.names = FALSE)
gene_symbols_id=read.csv('./gene_symbols_id_977.csv')
dt_water=dt_water[gene_symbols_id$ensembl_gene_id,]
dt_dmso=dt_dmso[gene_symbols_id$ensembl_gene_id,]
rownames(dt_water) <- gene_symbols_id$pr_gene_symbol
rownames(dt_dmso) <- gene_symbols_id$pr_gene_symbol
data_water$`time:ch1` <- str_remove(data_water$`time:ch1`, "hr") %>% as.numeric()
# extract before herb：water extract
data_water$herb <- str_extract(
  data_water$`treatment:ch1`, 
  ".*?(?= water extract)" 
)
# extract dose：number
data_water$dose <- str_extract(
  data_water$`treatment:ch1`, 
  "\\d+" 
) %>% as.numeric() 
# extract  unit：ug/mL
data_water$unit <- str_trim( 
  str_remove(
    data_water$`treatment:ch1`, 
    ".*\\d+"
  )
)
colnames(data_water)=c('title','description','cellline','treatment','time','sample','herb','dose','unit')
colnames(data_dmso)=c('title','description','cellline','treatment','time','sample')
data_water=data_water %>%
  mutate(
    control_cellline = case_when(
      str_detect(sample, "HT29_HT29") ~ "HT29_HT29",
      str_detect(sample, "SW1783_SW1783") ~ "SW1783_SW1783",
      TRUE ~ `cellline`
    )
  )
data_dmso=data_dmso %>%
  mutate(
    control_cellline = case_when(
      str_detect(sample, "HT29_HT29") ~ "HT29_HT29",
      str_detect(sample, "SW1783_SW1783") ~ "SW1783_SW1783",
      TRUE ~ `cellline`
    )
  )

write.csv(data_water,'herb_water_info.csv',row.names = F)
write.csv(data_dmso,'herb_dmso_info.csv',row.names = F)
write.csv(dt_water,'herb_water_data.csv',row.names = T)
write.csv(dt_dmso,'herb_dmso_data.csv',row.names = T)