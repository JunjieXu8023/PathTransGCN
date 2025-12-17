import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from GCN import GCN
from tools import load_data, accuracy

CONFIG = {
    'featuredata': 'utils/tcga_lgg_embed.csv',
    'adjdata': 'SNF_result/SNF_fused_matrix_full.csv',
    'labeldata': '../LGG_dataset/LGG_histological_subtype.tsv',
    'nclass': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 123,
    'epochs': 10000,
    'learningrate': 0.003,
    'weight_decay': 0.0001,
    'hidden': 64,
    'dropout': 0.7,
    'threshold': 0.005,
    'patience': 3000
}

def load_sample_splits():
    train_samples = pd.read_csv('SNF_result/train_samples.txt')['sampleID'].tolist()
    test_samples = pd.read_csv('SNF_result/test_samples.txt')['sampleID'].tolist()
    print(f"Loaded {len(train_samples)} train and {len(test_samples)} test samples.")
    return set(train_samples), set(test_samples)

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double().sum()
    return correct / len(labels)

def train(epoch, model, optimizer, features, adj, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1:3d} | Loss: {loss_train.item():.4f} | Acc: {acc_train.item():.4f}')
    return loss_train.item()

def test(model, features, adj, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        pred = output[idx_test].cpu().numpy().argmax(axis=1)
        true = labels[idx_test].cpu().numpy()
        f1 = f1_score(true, pred, average='weighted')
        print("Test set results:",
              f"loss= {loss_test.item():.4f}",
              f"accuracy= {acc_test.item():.4f}",
              f"F1= {f1:.4f}")
    return acc_test.item(), f1

def predict_and_save(model, features, adj, all_samples, test_idx, save_path='SNF_result/GCN_predicted_labels.csv'):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred_labels = output.cpu().numpy().argmax(axis=1)
    result_df = pd.DataFrame({
        'Sample': all_samples,
        'predict_label': pred_labels
    })
    result_df = result_df.iloc[test_idx].reset_index(drop=True)
    result_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def main():
    args = CONFIG
    device = torch.device(args['device'])
    setup_seed(args['seed'])

    train_set, test_set = load_sample_splits()

    print("\nLoading adjacency matrix and features...")
    temp_label_file = 'temp_dummy_label.csv'
    fea_temp = pd.read_csv(args['featuredata'], header=0, index_col=None)
    fea_temp.rename(columns={fea_temp.columns[0]: 'Sample'}, inplace=True)
    dummy_label = pd.DataFrame({'Sample': fea_temp['Sample'], 'label': 0})
    dummy_label.to_csv(temp_label_file, index=False)

    adj, data, _ = load_data(
        adj=args['adjdata'],
        fea=args['featuredata'],
        lab=temp_label_file,
        threshold=args['threshold']
    )
    os.remove(temp_label_file)

    problematic_sample = 'TCGA-R8-A6YH-01'
    all_samples_orig = data['Sample'].tolist()
    print(f"Total samples in graph before filtering: {len(all_samples_orig)}")

    if problematic_sample in all_samples_orig:
        keep_mask = data['Sample'] != problematic_sample
        data = data[keep_mask].reset_index(drop=True)
        adj = adj[np.ix_(keep_mask, keep_mask)]
        print(f"Removed sample: {problematic_sample}")
    else:
        print(f"Sample {problematic_sample} not found — no removal performed.")

    all_samples = data['Sample'].tolist()
    n_samples = len(all_samples)
    print(f"Total samples after filtering: {n_samples}")

    print("Loading real labels from TSV...")
    label_df = pd.read_csv(args['labeldata'], sep='\t', header=0)

    if 'samplelD' in label_df.columns:
        label_df.rename(columns={'samplelD': 'Sample'}, inplace=True)
    elif label_df.columns[0] != 'Sample':
        label_df.rename(columns={label_df.columns[0]: 'Sample'}, inplace=True)

    label_col = [col for col in label_df.columns if col != 'Sample'][0]
    print(f"Using label column: '{label_col}'")

    label_df = label_df.set_index('Sample').reindex(all_samples).reset_index()

    if label_df[label_col].isnull().any():
        missing_samples = label_df[label_df[label_col].isnull()]['Sample'].tolist()
        raise ValueError(f"Missing labels for samples: {missing_samples}")

    unique_labels = sorted(label_df[label_col].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label mapping:", label_to_idx)

    labels_np = label_df[label_col].map(label_to_idx).astype(int).values
    args['nclass'] = len(unique_labels)
    print(f"Number of classes: {args['nclass']}")
    print(f"Total samples after removing TCGA-R8-A6YH-01: {n_samples}")

    train_idx = [i for i, s in enumerate(all_samples) if s in train_set]
    test_idx = [i for i, s in enumerate(all_samples) if s in test_set]
    print(f"Graph indices - Train: {len(train_idx)}, Test: {len(test_idx)}")
    print("Masking edges between train and test sets to prevent leakage...")
    is_train = np.array([s in train_set for s in all_samples])
    is_test = np.array([s in test_set for s in all_samples])

    allowed = ~(np.outer(is_train, is_test) | np.outer(is_test, is_train))
    allowed = allowed.astype(np.float32)

    adj = adj * allowed

    adj += np.eye(n_samples) * 1e-8

    cross_edges = adj[np.ix_(train_idx, test_idx)].sum()
    print(f"Remaining train-test edges after masking: {cross_edges:.2f}")

    train_labels = labels_np[train_idx]
    homophily_ratios = []
    for i in train_idx:
        neighbors = np.where(adj[i] > 1e-6)[0]
        if len(neighbors) == 0:
            continue
        neighbor_labels = labels_np[neighbors]
        same_label = (neighbor_labels == labels_np[i]).mean()
        homophily_ratios.append(same_label)

    print(f"Average homophily in train set: {np.mean(homophily_ratios):.3f}")

    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
    features_tensor = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long, device=device)

    assert features_tensor.shape[0] == n_samples
    assert labels_tensor.shape[0] == n_samples
    assert adj_tensor.shape[0] == n_samples

    model = GCN(
        n_in=features_tensor.shape[1],
        n_hid=args['hidden'],
        n_out=args['nclass'],
        dropout=args['dropout']
    ).to(device)

    # GAT_model
    # model = GAT(
    #     nfeat=features_tensor.shape[1],
    #     nhid=args['hidden'],
    #     nclass=args['nclass'],
    #     dropout=args['dropout'],
    #     nheads=4
    # ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args['learningrate'],
        weight_decay=args['weight_decay']
    )

    print("\nStarting training (loss only on train samples)...")
    best_loss = float('inf')
    bad_counter = 0
    best_epoch = 0
    os.makedirs('model/GCN', exist_ok=True)

    train_idx_tensor = torch.tensor(train_idx, device=device)
    best_train_loss = float('inf')
    bad_counter = 0

    for epoch in range(args['epochs']):
        # --- 训练 ---
        model.train()
        optimizer.zero_grad()
        output = model(features_tensor, adj_tensor)
        loss_train = F.cross_entropy(output[train_idx_tensor], labels_tensor[train_idx_tensor])
        loss_train.backward()
        optimizer.step()

        if loss_train.item() < best_train_loss:
            best_train_loss = loss_train.item()
            bad_counter = 0
            torch.save(model.state_dict(), 'model/GCN/best_model.pkl')
            saved = True
        else:
            bad_counter += 1
            if bad_counter >= args['patience']:
                print(f"* Early stopping at epoch {epoch + 1} based on TRAIN loss")
                break

        if epoch % 100 == 0:
            acc_train = (
                        output[train_idx_tensor].argmax(dim=1) == labels_tensor[train_idx_tensor]).float().mean().item()
            print(f"Epoch {epoch:4d} | Train Loss: {loss_train.item():.4f} Acc: {acc_train:.4f}")

    print(f"\n* Loading best model (epoch {best_epoch}) for evaluation...")
    model.load_state_dict(torch.load('model/GCN/best_model.pkl'))

    test_idx_tensor = torch.tensor(test_idx, device=device)
    acc, f1 = test(model, features_tensor, adj_tensor, labels_tensor, test_idx_tensor)

    predict_and_save(model, features_tensor, adj_tensor, all_samples, test_idx)

    print(f"\n* Final Result - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':

    main()

