#Implementation of GraphTL

## Dependencies

- Python 3.6
- PyTorch 1.9.1
- dgl 0.7.1 (Big graph 0.8.1)

## Datasets

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' , 'Pubmed' , 'ogbn-arxiv' and 'PPI'

| Dataset    | # Nodes | # Edges | # Classes       |
|------------| ------- | ------- |-----------------|
| Cora       | 2,708   | 10,556  | 7               |
| Citeseer   | 3,327   | 9,228   | 6               |
| Pubmed     | 19,717  | 88,651  | 3               |
| ogbn-arxiv | 169343  | 1166243 | 40              |
| PPI        | 56944   | 818716  | 121(muti-label) |
## Arguments

| Dataset    | path_length | df  | de  | k    | hid_dim | out_dim | lr     | weight_decay | epoch |
|------------|-------------|-----|-----|------|---------|---------|--------|--------------|-------|
| Cora       | 2           | 0.2 | 0.5 | 50   | 256     | 256     | 0.001  | 1.5e-4       | 350   |
| Citeseer   | 2           | 0.3 | 0.5 | 200  | 512     | 512     | 0.001  | 4.5e-4       | 350   |
| Pubmed     | 3           | 0.4 | 0.8 | 1    | 512     | 512     | 0.0002 | 7e-5         | 2800  |

n_procs need to be tuned to the computer configuration in most of our experiments set to 1 (Pubmed dataset is set to 3). 
For 'ogbn-arxiv' and 'PPI' we provide precise code without modifying any parameters.



