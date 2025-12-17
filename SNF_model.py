import pandas as pd
import snf
import os
from sklearn.model_selection import train_test_split

EXPRESSION_PATH = '../LGG/expression.tsv'
METHYLATION_PATH = '../LGG/methylation.tsv'

output_dir = 'SNF_result'
K = 20
mu = 0.5
metric = 'sqeuclidean'
random_seed = 123  # seed

os.makedirs(output_dir, exist_ok=True)

print('Loading multi-omics datasets...')
expr_df = pd.read_csv(EXPRESSION_PATH, sep='\t', index_col=0)   # samples × genes
meth_df = pd.read_csv(METHYLATION_PATH, sep='\t', index_col=0)  # samples × genes

common_samples = expr_df.index.intersection(meth_df.index)
print(f"Common samples across omics: {len(common_samples)}")

expr_df = expr_df.loc[common_samples]
meth_df = meth_df.loc[common_samples]

print('Splitting into train (80%) and test (20%)...')
train_samples, test_samples = train_test_split(
    common_samples.tolist(),
    test_size=0.2,
    random_state=random_seed
)

# optional
pd.Series(train_samples, name='sampleID').to_csv(f'{output_dir}/train_samples.txt', index=False, header=True)
pd.Series(test_samples, name='sampleID').to_csv(f'{output_dir}/test_samples.txt', index=False, header=True)

print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

expr_train = expr_df.loc[train_samples].values.astype(float)
meth_train = meth_df.loc[train_samples].values.astype(float)

omics_matrices = [expr_train, meth_train]  # Addability

print('Computing affinity matrices...')
affinity_nets = snf.make_affinity(omics_matrices, metric=metric, K=K, mu=mu)

print('Fusing networks via SNF...')
fused_net = snf.snf(affinity_nets, K=K)

print('Saving fused matrix...')
fused_df = pd.DataFrame(fused_net, index=train_samples, columns=train_samples)
fused_df.to_csv(f'{output_dir}/SNF_fused_matrix_train.csv')

# print('Plotting clustermap...')
# np.fill_diagonal(fused_df.values, 0)
# fig = sns.clustermap(fused_df, cmap='vlag', figsize=(10, 10), cbar_kws={'shrink': 0.5})
# fig.savefig(f'{output_dir}/SNF_fused_clustermap_train.png', dpi=300, bbox_inches='tight')

print('* Success! Results saved in the "SNF_result" folder.')
