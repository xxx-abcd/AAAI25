import torch
import torch.nn as nn
import torch.nn.functional as F
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, adj_out, adj_in):
        x = F.relu(self.fc1(x))
        x = self.gcn_layer(x, adj_out, adj_in)
        x = F.relu(self.fc2(x))
        return x

    def gcn_layer(self, x, adj_out, adj_in):
        # x_out = torch.spmm(adj_out, x)
        # x_in = torch.spmm(adj_in, x)
        x_out = adj_out @ x
        x_in = adj_in @ x
        x = (x_out + x_in) / 2
        return x

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(GraphTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads)
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        x = x.permute(1, 0, 2)  # Transformer expects (L, B, D)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        linear_output = self.linear(x)
        x = x + self.dropout(linear_output)
        x = self.norm2(x)
        x = x.permute(1, 0, 2)  # Permute back to (B, L, D)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super(EdgeDecoder, self).__init__()
        self.fc = nn.Linear(2 * hidden_channels, 1)

    def forward(self, z):
        batch_size, num_nodes, _ = z.size()
        z_expanded = z.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        edge_features = torch.cat((z_expanded, z_expanded.transpose(1, 2)), dim=-1)
        edge_logits = self.fc(edge_features).squeeze(-1)
        return edge_logits

class GraphEdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers):
        super(GraphEdgePredictor, self).__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels)
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_channels, num_heads) for _ in range(num_layers)]
        )
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x, adj_out, adj_in, node_mask):
        z = self.encoder(x, adj_out, adj_in)
        key_padding_mask = (node_mask == 0)  # 0 表示被 mask 的节点
        for layer in self.transformer_layers:
            z = layer(z, key_padding_mask=key_padding_mask)
        edge_logits = self.decoder(z)
        return edge_logits, z



# import torch
# import torch.nn.functional as F
# import torch.nn as nn
#
# class GNNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(GNNEncoder, self).__init__()
#         self.fc1 = nn.Linear(in_channels, hidden_channels)
#         self.fc2 = nn.Linear(hidden_channels, hidden_channels)
#
#     def forward(self, x, adj_out, adj_in):
#         x = F.relu(self.fc1(x))
#         x = self.gcn_layer(x, adj_out, adj_in)
#         x = F.relu(self.fc2(x))
#         return x
#
#     def gcn_layer(self, x, adj_out, adj_in):
#         x_out = torch.spmm(adj_out, x)
#         x_in = torch.spmm(adj_in, x)
#         x = (x_out + x_in) / 2
#         return x
#
# class GraphTransformerLayer(nn.Module):
#     def __init__(self, hidden_channels, num_heads):
#         super(GraphTransformerLayer, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads)
#         self.linear = nn.Linear(hidden_channels, hidden_channels)
#         self.norm1 = nn.LayerNorm(hidden_channels)
#         self.norm2 = nn.LayerNorm(hidden_channels)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x, mask=None):
#         x = x.permute(1, 0, 2)  # Transformer expects (L, B, D)
#         attn_output, _ = self.attention(x, x, x, attn_mask=mask)
#         x = x + self.dropout(attn_output)
#         x = self.norm1(x)
#         linear_output = self.linear(x)
#         x = x + self.dropout(linear_output)
#         x = self.norm2(x)
#         x = x.permute(1, 0, 2)  # Permute back to (B, L, D)
#         return x
#
# class GraphTransformer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_heads, num_layers):
#         super(GraphTransformer, self).__init__()
#         self.encoder = GNNEncoder(in_channels, hidden_channels)
#         self.transformer_layers = nn.ModuleList(
#             [GraphTransformerLayer(hidden_channels, num_heads) for _ in range(num_layers)]
#         )
#
#     def forward(self, x, adj_out, adj_in):
#         x = self.encoder(x, adj_out, adj_in)
#         for layer in self.transformer_layers:
#             x = layer(x)
#         x = torch.mean(x, dim=1)  # Global mean pooling
#         return x