import torch
from torch import nn
class WordEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_size):
        super(WordEmbedding, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Embedding(num_embeddings=num_embedding, embedding_dim = embedding_size),
            nn.Embedding(num_embeddings=num_embedding, embedding_dim = embedding_size)
        )

    def forward(self, inputs):
        center, context, = inputs
        centers_embedding = self.net[0](center)
        context_embedding = self.net[1](context)
        return torch.bmm(centers_embedding, context_embedding.permute(0, 2,1))