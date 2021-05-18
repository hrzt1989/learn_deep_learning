import torch
from torch import nn
from torch.nn import functional
class TextCnn(nn.Module):
    def __init__(self, embedding_count, embedding_size, convs, outputs):
        super(TextCnn, self).__init__()
        self.embedding_count = embedding_count
        self.embedding_size = embedding_size
        self.outputs = outputs
        self.embedding = nn.Embedding(num_embeddings=embedding_count, embedding_dim=embedding_size)
        self.embedding_const = nn.Embedding(num_embeddings=embedding_count, embedding_dim=embedding_size)
        conv_list, all_output_channel, = self.init_convs(embedding_size * 2, convs)
        self.conv_list = conv_list
        self.linear_layer = nn.Linear(all_output_channel, outputs)
        self.all_output_channel = all_output_channel

    def init_convs(self, inchannels, convs):
        convs_list = nn.ModuleList()
        all_output_channels = 0
        for kernel_size, outchannel in convs:
            one_conv = nn.Conv1d(in_channels=inchannels, out_channels=outchannel, kernel_size=kernel_size)
            convs_list.append(one_conv)
            all_output_channels += outchannel
        return convs_list, all_output_channels

    def forward(self, input):
        x_embedding = self.embedding(input)
        x_embedding_const = self.embedding_const(input)
        x_feature = torch.cat([x_embedding, x_embedding_const], dim=2).permute(0, 2, 1)
        con_results = []
        for one_conv in self.conv_list:
            one_conv_result = one_conv(x_feature)
            con_results.append(functional.max_pool1d(one_conv_result, kernel_size = one_conv_result.shape[2]))
        linear_feature = functional.dropout(torch.cat(con_results, dim=2).view(-1, self.all_output_channel), 0.7)
        return self.linear_layer(linear_feature)
