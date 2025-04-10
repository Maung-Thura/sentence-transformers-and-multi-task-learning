import torch.nn as nn
from transformers import AutoModel


class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name, num_classes, num_sentiment_classes):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Task A: Sentence Classification Head
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Task B: Sentiment Analysis Head
        self.sentiment_classifier = nn.Linear(self.hidden_size, num_sentiment_classes)

        # Add dropout for regularization during training
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token output for classification tasks
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)  # Apply dropout

        # Task-Specific Logits
        class_logits = self.classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        return class_logits, sentiment_logits
