import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):

    def __init__(self, num_feats):
        super(GAT, self).__init__()
        self.num_feats = num_feats
        self.weight_key = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        nn.init.xavier_uniform_(self.weight_key, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query, gain=1.414)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        :param x: dim: bz x num_node x num_feat
        :return: dim: bz x num_node x num_node
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        key = torch.matmul(x, self.weight_key)
        query = torch.matmul(x, self.weight_query)
        attn_input = key.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, 1) + query.repeat(1, num_nodes,
                                                                                                           1)
        attn_output = attn_input.squeeze(2).view(batch_size, num_nodes, num_nodes)
        attn_output = F.leaky_relu(attn_output, negative_slope=0.2)
        attention = F.softmax(attn_output, dim=2)
        attention = self.dropout(attention)
        attn_feat = torch.matmul(attention, x).permute(0, 2, 1)
        return attn_feat


class GraphAttnEncoder(nn.Module):
    def __init__(self, config):
        super(GraphAttnEncoder, self).__init__()
        self.data_dim = config.data_dim
        self.data_length = config.data_length
        self.input_size = config.input_size
        assert self.data_length % self.input_size == 0
        self.agg_window = self.data_length // self.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.register_buffer("conv_weight", torch.ones((1, 1, self.agg_window)) / self.agg_window)
        self.feat_gat = GAT(self.input_size)
        self.time_gat = GAT(self.data_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.data_dim * 3, self.hidden_size, batch_first=False)
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(x.size(0) * x.size(1), 1, x.size(2))
        x = torch.conv1d(x, self.conv_weight, stride=self.agg_window, padding=0, dilation=1)  # bz x dim x Tw
        x = x.view(batch_size, -1, x.size(2))
        x_time = x.permute(0, 2, 1)  # bz x Tw x dim
        x_feat = x  # bz x dim x Tw
        h_time = self.time_gat(x_time).permute(0, 2, 1)
        h_feat = self.feat_gat(x_feat)
        x = torch.cat([x_time, h_feat, h_time], dim=2).permute(1, 0, 2)
        self.gru.flatten_parameters()
        _, x = self.gru(x)
        x = x.permute(1, 0, 2).squeeze(1)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 4, 100)

    class Config:
        def __init__(self):
            self.data_dim = 4
            self.data_length = 100
            self.input_size = 10
            self.hidden_size = 300
            self.output_size = 128

    config = Config()
    model = GraphAttnEncoder(config)
    print(model(x).shape)
