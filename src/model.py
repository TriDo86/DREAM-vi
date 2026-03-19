import torch
import torch.nn as nn

class DREAMModel(nn.Module):
    def __init__(self, embedding_size, num_languages):
        super(DREAMModel, self).__init__()
        self.language_encoder = nn.Linear(embedding_size, embedding_size)
        self.meaning_encoder = nn.Linear(embedding_size, embedding_size)
        self.language_identifier = nn.Linear(embedding_size, num_languages)

    def forward(self, sentence_embedding):
        language_embedding = self.language_encoder(sentence_embedding)
        meaning_embedding  = self.meaning_encoder(sentence_embedding)
        language_id = self.language_identifier(language_embedding)
        return language_embedding, meaning_embedding, language_id