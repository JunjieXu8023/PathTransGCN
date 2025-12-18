import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from PT_module import PathwayTransformer, PathwayTransformerMAE


def load_pathway_dict(csv_path):
    df = pd.read_csv(csv_path)
    pathway_dict = {}
    for _, row in df.iterrows():
        genes = row['genes'].split('|')
        name = row['name']
        pathway_dict[name] = genes
    return pathway_dict


def filter_pathway_dict(pathway_dict, gene_to_id, omic_types=None):
    filtered_pathway_dict = {}
    for pname, genes in pathway_dict.items():
        valid_genes = [g for g in genes if g in gene_to_id]
        if len(valid_genes) > 0:
            filtered_pathway_dict[pname] = valid_genes
    return filtered_pathway_dict


def load_multiomics_data(omics_files, omics_names):
    """
    Args:
        omics_files (list): ['expr.tsv', 'meth.tsv']
        omics_names (list): ['exp', 'meth']

    Returns:
        X: torch.Tensor of shape (n_samples, n_genes, n_omics)
        gene_to_id: dict mapping gene symbol to index
        num_omics: int
        in_feas_dim: list of feature dims per omics (all = n_genes)
        sample_ids: list of sample IDs (aligned)
    """
    print("Loading and aligning multi-omics data...")

    omics_dfs = []
    for file in omics_files:
        df = pd.read_csv(file, sep='\t', index_col=0)
        df.fillna(0, inplace=True)
        omics_dfs.append(df)

    common_samples = omics_dfs[0].index
    for df in omics_dfs[1:]:
        common_samples = common_samples.intersection(df.index)
    common_samples = sorted(common_samples)
    print(f"Aligned samples: {len(common_samples)}")

    all_genes = set()
    for df in omics_dfs:
        all_genes.update(df.columns)
    all_genes = sorted(all_genes)
    print(f"Total unique genes across omics: {len(all_genes)}")

    gene_to_id = {gene: i for i, gene in enumerate(all_genes)}
    num_genes = len(all_genes)
    num_omics = len(omics_dfs)

    X = torch.zeros((len(common_samples), num_genes, num_omics), dtype=torch.float32)

    for omic_idx, df in enumerate(omics_dfs):
        df_aligned = df.loc[common_samples, :]
        for gene in all_genes:
            if gene in df_aligned.columns:
                gene_idx = gene_to_id[gene]
                X[:, gene_idx, omic_idx] = torch.tensor(
                    df_aligned[gene].values, dtype=torch.float32
                )
            # else: already 0 (by initialization)

    in_feas_dim = [num_genes] * num_omics

    return X, gene_to_id, num_omics, in_feas_dim, common_samples


class ExpressionDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def train(model, dataloader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")


def extract_embeddings(model, dataset, device):
    model.eval()
    all_embeds = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, cls_embed = model(batch)
            all_embeds.append(cls_embed.cpu())
    all_embeds = torch.cat(all_embeds, dim=0).numpy()  # [N_samples, D]
    return all_embeds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 配置组学文件 ===
    omics_files = [
        '../LGG/expression.tsv',
        '../LGG/methylation.tsv'
    ]
    omics_names = ['exp', 'meth']

    # load pathway
    pathway_dict_raw = load_pathway_dict('Pathway/41568_2020_240_MOESM4_ESM.csv')

    X, gene_to_id, num_omics, in_feas_dim, sample_ids = load_multiomics_data(omics_files, omics_names)

    print("NaN:", torch.isnan(X).any().item())
    print("Inf:", torch.isinf(X).any().item())

    # === 过滤通路 ===
    pathway_dict = filter_pathway_dict(pathway_dict_raw, gene_to_id, omic_types=omics_names)
    print(f"[INFO] Original Pathway Number: {len(pathway_dict_raw)}, "
          f"Number of effective pathways after filtering: {len(pathway_dict)}")

    # === 划分训练/验证集（按样本）===
    train_idx, val_idx = train_test_split(
        range(X.shape[0]), test_size=0.2, random_state=42
    )
    train_X = X[train_idx]
    val_X = X[val_idx]

    train_dataset = ExpressionDataset(train_X)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # === 构建模型 ===
    backbone = PathwayTransformer(
        num_omics=num_omics,
        gene_to_id=gene_to_id,
        pathway_dict=pathway_dict,
        in_feas_dim=in_feas_dim,
        embed_dim=128,
        depth=3,
        num_heads=4
    )
    model = PathwayTransformerMAE(backbone, mask_ratio=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # === 训练 ===
    train(model, train_loader, optimizer, device, epochs=300)

    # === 提取全量嵌入 ===
    full_dataset = ExpressionDataset(X)
    embeddings = extract_embeddings(model, full_dataset, device)

    # === 保存嵌入（带 sampleID）===
    os.makedirs('utils', exist_ok=True)
    df_embed = pd.DataFrame(embeddings, index=sample_ids)
    df_embed.insert(0, "Cell_line", sample_ids)
    df_embed.to_csv('utils/tcga_lgg_embed.csv', index=False)
    print("The embedding has been saved to 'utils/tcga_lgg_embed.csv'")


if __name__ == "__main__":

    main()
