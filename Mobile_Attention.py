import torch
import torch.nn as nn


## Core code for Mobile-Attention, Please refer to https://github.com/thuml/Flowformer for specific experiments

class Mobile_Attention(nn.Module):
    # Mobile Attention with head competing mechanism
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6):
        super(Mobile_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Mobile-Attention competing on the head dimension
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / torch.sum(
            (queries + self.eps) * (keys.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        source_outgoing = 1.0 / torch.sum(
            (keys + self.eps) * (queries.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        # (2) conservation refine for source and sink
        conserved_sink = torch.sum((queries + self.eps) * (
                (keys * source_outgoing[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                        1) + self.eps), dim=-1)
        conserved_source = torch.sum((keys + self.eps) * (
                (queries * sink_incoming[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                         1) + self.eps), dim=-1)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # A simple case for correctness check
    q = torch.randn([2, 7, 32])
    k = torch.randn([2, 7, 32])
    v = torch.randn([2, 7, 32])
    mobile_attn = Mobile_Attention(32, 32, 32, 16)
    u = mobile_attn(q, k, v)
    print(u.shape)
