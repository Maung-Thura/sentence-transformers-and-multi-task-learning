import torch
import torch.nn as nn
from transformers import AutoModel


class SentenceTransformer(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', pooling_strategy='mean'):
        super(SentenceTransformer, self).__init__()
        if pooling_strategy not in ['mean', 'cls']:
            raise ValueError("pooling_strategy must be either 'mean' or 'cls'")

        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        # No extra layers needed for basic embedding generation after pooling

    def forward(self, input_ids, attention_mask):
        # Get token embeddings from the transformer backbone
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (Batch, SeqLen, HiddenDim)

        # --- Apply Pooling Strategy ---
        if self.pooling_strategy == 'mean':
            # Mean Pooling: Average token embeddings, considering attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # Sum embeddings where attention_mask is 1
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            # Sum mask elements to get the count of non-padding tokens
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # Clamp for stability
            sentence_embedding = sum_embeddings / sum_mask
        elif self.pooling_strategy == 'cls':
            # Use the embedding of the [CLS] token (first token)
            sentence_embedding = token_embeddings[:, 0]  # (Batch, HiddenDim)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

        return sentence_embedding
