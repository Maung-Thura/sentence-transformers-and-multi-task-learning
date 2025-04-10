import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim.adamw import AdamW

from MultiTaskDataset import MultiTaskDataset
from MultiTaskSentenceTransformer import MultiTaskSentenceTransformer
from SentenceTransformer import SentenceTransformer

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Configuration ---
# Ensure environment variables are set before running
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", 'sentence-transformers/all-MiniLM-L6-v2')
MTL_MODEL_NAME = os.getenv("MTL_MODEL_NAME", 'bert-base-uncased')
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
EPOCHS = int(os.getenv("EPOCHS", 4))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
MAX_LEN = int(os.getenv("MAX_LEN", 128))


# Task 1, Implement Sentence Transformer
def test_sentence_transformer(sentences: list[str]):
    # Instantiate the tokenizer (outside the model)
    tokenizer = AutoTokenizer.from_pretrained(ST_MODEL_NAME)

    # Instantiate the Sentence Transformer model
    model = SentenceTransformer(model_name=ST_MODEL_NAME, pooling_strategy='mean').to(device)
    model.eval()  # as there is no training

    print("\n--- Encoding Sentences ---")

    # Tokenize the sentences
    # Ensure padding and truncation are handled, and return PyTorch tensors
    inputs = tokenizer(
        sentences,
        padding=True,  # Pad sequences to the longest sequence in the batch
        truncation=True,  # Truncate sequences longer than model max length
        return_tensors="pt"  # Return PyTorch tensors
    ).to(device)  # Move tokenized inputs to the chosen device

    # Perform inference to get embeddings
    with torch.no_grad():  # Disable gradient calculations for inference
        embeddings = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    print(f"\nObtained Embeddings Shape: {embeddings.shape}")

    print("\nSample Embeddings (first few dimensions of the first 2 sentences):")
    for i in range(min(2, len(sentences))):  # Show first 2 sentences
        print(f"Sentence: '{sentences[i]}'")
        # Convert to numpy for easier printing, show first 10 dimensions
        print(f"  Embedding[:10]: {embeddings[i, :10].cpu().numpy()}")
        # print("\nFull embedding for the first sentence:\n", embeddings[0].cpu().numpy())
        print("-" * 20)


# Task 2, Implement Multi-Task Learning Expansion
def test_sentence_classification_and_named_entity_recognition(train_sentences: list[str],
                                                              train_class_labels: list[str],
                                                              train_sentiment_labels: list[str],
                                                              test_sentences: list[str],
                                                              test_class_labels: list[str],
                                                              test_sentiment_labels: list[str]
                                                              ):
    # --- Label Mapping (String to Integer ID) ---
    # Base mappings on training data labels to ensure consistency
    class_labels_list = sorted(list(set(train_class_labels)))
    sentiment_labels_list = sorted(list(set(train_sentiment_labels)))

    class_label_to_id = {label: i for i, label in enumerate(class_labels_list)}
    sentiment_label_to_id = {label: i for i, label in enumerate(sentiment_labels_list)}

    id_to_class_label = {i: label for label, i in class_label_to_id.items()}
    id_to_sentiment_label = {i: label for label, i in sentiment_label_to_id.items()}

    num_classes = len(class_labels_list)
    num_sentiment_classes = len(sentiment_labels_list)

    print(f"\nClass Labels: {class_label_to_id}")
    print(f"Sentiment Labels: {sentiment_label_to_id}")

    # --- Convert string labels to integer IDs ---
    # Training data labels
    train_class_ids = [class_label_to_id[label] for label in train_class_labels]
    train_sentiment_ids = [sentiment_label_to_id[label] for label in train_sentiment_labels]

    # --- Prepare Datasets and DataLoaders ---
    tokenizer = AutoTokenizer.from_pretrained(MTL_MODEL_NAME)

    train_dataset = MultiTaskDataset(
        sentences=train_sentences,
        class_labels=train_class_ids,
        sentiment_labels=train_sentiment_ids,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # Shuffle training data
    )

    # --- Initialize Model, Optimizer, Loss ---
    model = MultiTaskSentenceTransformer(
        model_name=MTL_MODEL_NAME,
        num_classes=num_classes,
        num_sentiment_classes=num_sentiment_classes
    ).to(device)

    # Loss Functions
    class_loss_fn = nn.CrossEntropyLoss().to(device)
    sentiment_loss_fn = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    # Scheduler
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value
        num_training_steps=total_steps
    )

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            class_labels = batch['class_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            class_logits, sentiment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate losses
            loss_class = class_loss_fn(class_logits, class_labels)
            loss_sentiment = sentiment_loss_fn(sentiment_logits, sentiment_labels)

            # Combine losses
            total_loss = loss_class + loss_sentiment
            total_train_loss += total_loss.item()

            # Backward pass
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            # Scheduler step
            scheduler.step()

            progress_bar.set_postfix({'batch_loss': total_loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Training Loss: {avg_train_loss:.4f}")

    print("--- Training Finished ---")

    # --- Evaluation on Test Data ---
    print("\n--- Evaluating on Test Data ---")
    # Tokenize Test Data
    # Handle potential errors if test labels are not in training label set during mapping
    try:
        true_test_class_ids = [class_label_to_id[label] for label in test_class_labels]
        true_test_sentiment_ids = [sentiment_label_to_id[label] for label in test_sentiment_labels]
    except KeyError as e:
        print(f"Error: Label '{e}' in test set not found in training set label mapping.")
        print("Cannot calculate accuracy.")
        return  # Exit function if labels don't match

    test_encodings = tokenizer(
        test_sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,  # Use the same MAX_LEN as training
        return_tensors="pt"
    )
    # Create a simple dataloader for test set for batch processing if needed,
    # or process all at once if memory allows (as done here)
    test_input_ids = test_encodings['input_ids'].to(device)
    test_attention_mask = test_encodings['attention_mask'].to(device)

    # Perform Inference
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        class_logits, sentiment_logits = model(input_ids=test_input_ids, attention_mask=test_attention_mask)

    # Get Predictions (Indices)
    class_preds_indices = torch.argmax(class_logits, dim=1)
    sentiment_preds_indices = torch.argmax(sentiment_logits, dim=1)

    # Move predictions to CPU and convert to NumPy for comparison
    predicted_class_ids_np = class_preds_indices.cpu().numpy()
    predicted_sentiment_ids_np = sentiment_preds_indices.cpu().numpy()

    # Convert true test labels to NumPy array
    true_class_ids_np = np.array(true_test_class_ids)
    true_sentiment_ids_np = np.array(true_test_sentiment_ids)

    # --- Calculate Accuracy ---
    class_correct_predictions = np.sum(predicted_class_ids_np == true_class_ids_np)
    sentiment_correct_predictions = np.sum(predicted_sentiment_ids_np == true_sentiment_ids_np)

    total_test_samples = len(test_sentences)

    class_accuracy = class_correct_predictions / total_test_samples
    sentiment_accuracy = sentiment_correct_predictions / total_test_samples

    print(f"Sentence Classification Accuracy: {class_accuracy:.4f} ({class_correct_predictions}/{total_test_samples})")
    print(
        f"Sentiment Analysis Accuracy: {sentiment_accuracy:.4f} ({sentiment_correct_predictions}/{total_test_samples})")
    # --- End Accuracy Calculation ---

    # Map prediction indices to labels for display
    class_predictions = [id_to_class_label.get(i, "Unknown") for i in predicted_class_ids_np]
    sentiment_predictions = [id_to_sentiment_label.get(i, "Unknown") for i in
                             predicted_sentiment_ids_np]

    # Display Results
    print("\n--- Predictions on Test Sentences ---")
    for i, sentence in enumerate(test_sentences):
        true_class_label = test_class_labels[i]
        true_sentiment_label = test_sentiment_labels[i]
        pred_class_label = class_predictions[i]
        pred_sentiment_label = sentiment_predictions[i]

        print(f"Sentence: {sentence}")
        print(
            f"  Class -> True: {true_class_label}, Predicted: {pred_class_label} {'(Correct)' if true_class_label == pred_class_label else '(Incorrect)'}")
        print(
            f"  Sentiment -> True: {true_sentiment_label}, Predicted: {pred_sentiment_label} {'(Correct)' if true_sentiment_label == pred_sentiment_label else '(Incorrect)'}")
        print("-" * 20)


if __name__ == '__main__':
    _train_sentences = [
        "The successful new quantum computing discovery could revolutionize industries.",  # Tech/Sci, Positive
        "Unfortunately, the constant failures caused significantly hampered the students expectation to meet grading requirements.",
        # General, Negative
        "Researchers cannot pinpoint the cause of the phenomenon.",
        # Science, Negative - Mismatch with label below? Correcting label
        "My new phone's battery life is amazing!",  # Technology, Positive
        "Recent market change is a bad news for traders.",  # General, Negative
        "Some staff expressed their dissatisfaction regarding the unexpected changes in office policies.",
        # General, Negative
        "Scientists are exploring new methods for carbon capture.",  # Science, Positive
        "A critical bug that causes considerable dissatisfaction to the users.",  # Technology, Negative
    ]

    # Task A: Sentence Classification (Topic)
    _train_class_labels = [
        "Technology", "General", "Science", "Technology",
        "General", "General", "Science", "Technology"
    ]
    # Task B: Sentiment Analysis
    _train_sentiment_labels = [
        "Positive", "Negative", "Negative", "Positive",
        "Negative", "Negative", "Positive", "Negative"
    ]

    _test_sentences_1 = [
        "The weather today is sunny and warm.",
        "Machine learning models require significant data.",
        "This really is a bad news!",
        "Embeddings capture semantic meaning.",
        "It is unfortunate that earthquake cannot be forecasted."
    ]

    _test_sentences_1_class_labels = [
        "General", "Technology", "General", "Technology", "Science"
    ]

    _test_sentences_1_sentiment_labels = [
        "Positive", "Positive", "Negative", "Positive", "Negative"
    ]

    _test_sentences_2 = [
        "Unfortunately, the persistent network outages have significantly hampered our team's ability to meet crucial deadlines this week.",
        "The recent community volunteer event was incredibly successful, bringing neighbors together for a productive afternoon.",
        "While the initial software setup was complex, the long-term benefits and improved efficiency are becoming apparent.",
        "The updated report outlines the quarterly financial performance, detailing both revenue streams and operational expenditures.",
        "Many customers expressed considerable dissatisfaction regarding the unexpected changes to the subscription service terms."
    ]

    _test_sentences_2_class_labels = [
        "Technology", "General", "Technology", "General", "General"
    ]

    _test_sentences_2_sentiment_labels = [
        "Negative", "Positive", "Positive", "Positive", "Negative"
        # Assuming neutral reports treated as Positive for 2-class sentiment
    ]

    print("=" * 40)
    print("RUNNING TEST SET 1")
    print("=" * 40)
    test_sentence_transformer(_test_sentences_1)
    test_sentence_classification_and_named_entity_recognition(_train_sentences, _train_class_labels,
                                                              _train_sentiment_labels, _test_sentences_1,
                                                              _test_sentences_1_class_labels,
                                                              _test_sentences_1_sentiment_labels)

    print("\n" + "=" * 40)
    print("RUNNING TEST SET 2")
    print("=" * 40)
    test_sentence_transformer(_test_sentences_2)
    test_sentence_classification_and_named_entity_recognition(_train_sentences, _train_class_labels,
                                                              _train_sentiment_labels, _test_sentences_2,
                                                              _test_sentences_2_class_labels,
                                                              _test_sentences_2_sentiment_labels)
