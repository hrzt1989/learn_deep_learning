import torch
from torch import nn
from utility.cov_33_block import Cov33Block
class DenseBlock(nn.Module):

    def __init__(self, cov_num, input_channel, growth_rate):
        super(DenseBlock, self).__init__()
        self.output_channel = cov_num * growth_rate + input_channel
        self.block_list = self.gen_net(cov_num, input_channel, growth_rate)

    def gen_net(self, cov_num, input_channel, growth_rate):

        cov_block_list = nn.ModuleList()
        temp_input_channel = input_channel

        for index in range(cov_num):
            cov_block = Cov33Block(temp_input_channel, growth_rate)
            cov_block_list.append(cov_block)
            temp_input_channel += growth_rate
        return cov_block_list

    def forward(self, x):
        result = x
        for one_block in self.block_list:
            y = one_block(result)
            result = torch.cat((result, y), dim = 1)
        return result