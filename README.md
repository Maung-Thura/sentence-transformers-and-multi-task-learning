# Sentence Transformer and Multi-Task Learning Exercise

## Overview

This repository contains the implementation for the ML Apprentice Take Home Exercise[cite: 1]. The objective of this exercise is to implement, train, and discuss neural network architectures, specifically focusing on sentence transformers and their expansion into a multi-task learning (MTL) framework[cite: 1]. This README outlines the completed tasks, architectural decisions, setup instructions, and usage.

## Tasks Completed

### Task 1: Sentence Transformer Implementation

* **Objective:** Implement a model to encode input sentences into fixed-length embeddings using a transformer backbone[cite: 3, 4].
* **Implementation:**
    * A `SentenceTransformer` class was created using PyTorch and the Hugging Face `transformers` library.
    * It utilizes a pre-trained transformer model (`sentence-transformers/all-MiniLM-L6-v2` chosen for its efficiency and performance on sentence similarity tasks).
    * It aggregates token embeddings into a fixed-size sentence embedding using a selectable pooling strategy (`mean` or `cls`)[cite: 6]. Mean pooling was demonstrated as the default.
* **Testing:** The implementation was tested with sample sentences, showcasing the shape and sample values of the obtained embeddings[cite: 5].
* **Architectural Choices (Outside Backbone):**
    * **Pooling Strategy:** Implemented both 'mean' and 'cls' pooling to convert variable-length token sequences to fixed-length sentence vectors. Mean pooling averages non-padding token embeddings, considering the whole sentence. CLS pooling uses the output of the `[CLS]` token[cite: 6]. This choice is crucial for obtaining a single sentence representation.
    * **No Post-Pooling Layers:** No additional dense, activation, or dropout layers were added after pooling, as the goal was direct embedding generation[cite: 6].
    * **No Built-in Normalization:** L2 normalization was not applied within the model's forward pass, leaving it as an optional downstream step.

### Task 2: Multi-Task Learning Expansion

* **Objective:** Expand the sentence transformer to handle multiple tasks simultaneously[cite: 7].
    * Task A: Sentence Classification (Topic Classification - e.g., Technology, Science, General)[cite: 8].
    * Task B: Sentiment Analysis (e.g., Positive, Negative)[cite: 9].
* **Implementation:**
    * A `MultiTaskSentenceTransformer` class was created.
    * It uses a shared transformer backbone (`bert-base-uncased` was chosen as a standard base for fine-tuning classification tasks).
    * Separate linear "heads" (`nn.Linear`) were added for each task, mapping the shared representation to task-specific logits[cite: 9].
    * CLS pooling was used in this implementation to generate the shared representation fed into the heads.
    * A dropout layer was added after pooling for regularization during training.
* **Changes from SentenceTransformer:**
    * Addition of task-specific linear heads.
    * Modification of the `forward` pass to return multiple outputs (logits per task).
    * Standardized pooling strategy (CLS pooling used here).
    * Inclusion of a dropout layer for training regularization.
    * Use of a standard fine-tuning backbone (`bert-base-uncased`) suitable for classification tasks.

### Task 3: Training Considerations (Discussion)

* **Analysis:** Discussed the implications of freezing different parts of the network during training in an MTL context[cite: 10]:
    * **Entire Network Frozen:** Not suitable for training the network itself; only for fixed feature extraction[cite: 10].
    * **Backbone Frozen:** Trains only task heads. Faster, less memory, preserves general knowledge well, good for small datasets[cite: 11]. Limits adaptation.
    * **One Head Frozen:** Trains backbone and unfrozen head(s). Preserves performance on the frozen task but backbone adapts only to the unfrozen task(s)[cite: 11].
* **Transfer Learning Approach:** Discussed strategy[cite: 12]:
    * **Model Choice:** Selecting based on task relevance, domain, and resources (e.g., BERT for general classification fine-tuning)[cite: 12].
    * **Freezing Strategy:** Decisions based on dataset size, task similarity, and resources (e.g., freeze backbone for small data, fine-tune all for large data, gradual unfreezing as a compromise)[cite: 13].
    * **Rationale:** Balancing adaptation to new tasks with preservation of pre-trained knowledge[cite: 13].

### Task 4: Training Loop Implementation (Bonus)

* **Objective:** Implement the training loop for the MTL model[cite: 14].
* **Implementation:**
    * Included data preparation (label mapping, `Dataset`, `DataLoader`).
    * Initialized model, loss functions (`nn.CrossEntropyLoss` per task), optimizer (`AdamW`), and learning rate scheduler.
    * Implemented a standard PyTorch training loop iterating through epochs and batches.
    * **MTL Specifics:**
        * **Forward Pass:** Retrieves separate logits for each task.
        * **Combined Loss:** Calculates loss for each task individually and combines them (simple summation used in the example: `total_loss = loss_class + loss_sentiment`). Assumes equal task importance and comparable loss scales for this simple approach[cite: 14].
        * **Backpropagation:** Backpropagates the single, combined loss value, allowing gradients from all tasks to update the shared backbone parameters.
        * **Metrics:** Calculated accuracy independently for each task during evaluation[cite: 14].
* **Assumptions/Decisions:** Assumed aligned data (labels for all tasks per sentence), equal task weighting in loss summation, separate accuracy as sufficient evaluation metrics for demonstration[cite: 14].

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure a `requirements.txt` file listing necessary libraries like `torch`, `transformers`, `numpy`, `tqdm` is present in the repository)*[cite: 18].

## Usage

The main script (e.g., `main.py` or within a Jupyter Notebook) demonstrates the usage:

1.  **Task 1 (`test_sentence_transformer`):** Shows how to instantiate the `SentenceTransformer`, tokenize sentences, and generate fixed-length embeddings.
2.  **Task 2 & 4 (`test_sentence_classification_and_named_entity_recognition`):** Demonstrates setting up the `MultiTaskSentenceTransformer`, preparing data, running the training loop, performing inference on test data, and calculating accuracy for each task.

Modify the sample sentences and labels within the main script execution block (`if __name__ == '__main__':`) to experiment with different inputs.

## Key Decisions & Rationale Summary

* **Framework:** PyTorch and Hugging Face `transformers` were chosen for flexibility and ease of use with pre-trained models.
* **Sentence Transformer Pooling:** Mean pooling was chosen as a default for `SentenceTransformer` as it generally performs well for sentence similarity by incorporating information from all tokens[cite: 6]. CLS pooling was also implemented as an alternative.
* **MTL Backbone:** `bert-base-uncased` was chosen for the MTL task as it's a standard, robust backbone for fine-tuning various classification tasks, aligning well with the common practice of using the CLS token's output for such tasks[cite: 12].
* **MTL Architecture:** Hard parameter sharing (shared backbone, separate heads) was used for simplicity and efficiency[cite: 9]. CLS pooling was used to feed the heads.
* **MTL Loss:** Simple loss summation was implemented for demonstration purposes, assuming equal task importance[cite: 14].
* **Freezing Strategy:** Discussed trade-offs, recommending freezing the backbone for small datasets/limited resources and considering full fine-tuning or gradual unfreezing for larger datasets[cite: 13].

*(Self-reflection: This exercise provided valuable practice in implementing transformer architectures, understanding pooling strategies, expanding to multi-task learning with shared representations, and considering the practical aspects of fine-tuning strategies based on data and resource constraints.)*