import torch
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self, sentences, class_labels, sentiment_labels, tokenizer, max_len):
        self.sentences = sentences
        self.class_labels = class_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        class_label = self.class_labels[idx]
        sentiment_label = self.sentiment_labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence_text': sentence,  # Keep original text for reference if needed
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'class_labels': torch.tensor(class_label, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_label, dtype=torch.long)
        }
