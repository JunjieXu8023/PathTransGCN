import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, GELU, Softmax
from einops import rearrange
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PathwayTransformer(nn.Module):
    def __init__(self, num_omics, gene_to_id, pathway_dict, in_feas_dim,
                 embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_omics = num_omics
        self.gene_to_id = gene_to_id
        self.pathway_dict = pathway_dict
        self.in_feas = in_feas_dim

        self.pathway_layers = nn.ModuleDict()
        # for key, genes in pathway_dict.items():
        #     input_dim = len(genes) * num_omics
        #     self.pathway_layers[key] = Linear(input_dim, embed_dim)
        for key in self.pathway_dict:
            num_genes_in_pathway = len(self.pathway_dict[key])
            pathway_width = embed_dim
            self.pathway_layers[key] = nn.Linear(num_genes_in_pathway * num_omics, pathway_width)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + len(pathway_dict), embed_dim))
        self.pos_drop = Dropout(p=drop_rate)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, num_genes, num_omics]
        B = x.shape[0]
        pathway_embeddings = []

        for key in self.pathway_dict:
            gene_ids = [self.gene_to_id[x] for x in self.pathway_dict[key] if x in self.gene_to_id]
            if len(gene_ids) == 0:
                continue

            try:
                tmp = self.pathway_layers[key](
                    x[:, gene_ids, :].reshape(-1, len(gene_ids) * self.num_omics))
                pathway_embeddings.append(tmp)
            except Exception as e:
                print(f"[Warning] Pathway {key} processing failure ,skip.Reason: {str(e)}")
                continue

        # 如果没有任何有效通路嵌入，抛出异常以供外层处理
        if len(pathway_embeddings) == 0:
            raise ValueError("The current batch has no valid pathway embeddings. Please check if pathway_dict matches gene_to_id.")

        token_embeddings = torch.stack(pathway_embeddings, dim=1)  # [B, N_pathways, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, token_embeddings), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        x = self.transformer_blocks(x)
        x = self.norm(x)
        return x  # shape: [B, N+1, D]

    def get_cls_token_embedding(self, x):
        x = self.forward(x)
        return x[:, 0, :]  # [B, D]


class PathwayTransformerMAE(nn.Module):
    def __init__(self, backbone: PathwayTransformer, mask_ratio=0.3):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.decoder = nn.Linear(backbone.embed_dim, sum(backbone.in_feas))

    def forward(self, x):
        # x: [B, G, O]
        B, G, O = x.shape
        mask = (torch.rand(B, G, O, device=x.device) < self.mask_ratio)
        x_masked = x.clone()
        x_masked[mask] = 0

        cls_embed = self.backbone.get_cls_token_embedding(x_masked)  # [B, D]
        x_recon = self.decoder(cls_embed)
        x_true = x.view(B, -1)

        loss = F.mse_loss(x_recon[mask.view(B, -1)], x_true[mask.view(B, -1)])
        return loss, cls_embed