# 🔤 Next Word Predictor

> A deep learning-based Next Word Prediction system built using **LSTM** and **GRU** architectures with Natural Language Processing (NLP) techniques.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture Models](#architecture-models)
  - [LSTM (Long Short-Term Memory)](#lstm-long-short-term-memory)
  - [GRU (Gated Recurrent Unit)](#gru-gated-recurrent-unit)
  - [LSTM vs GRU Comparison](#lstm-vs-gru-comparison)
- [NLP Pipeline](#nlp-pipeline)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Results](#results)

---

## Overview

The **Next Word Predictor** is a sequence modeling project that predicts the most likely next word(s) given a sequence of input text. It leverages the power of Recurrent Neural Networks (RNNs) — specifically **LSTM** and **GRU** — to learn language patterns and contextual relationships from training data.

The model processes text as sequences of tokens, learns the statistical patterns of the language, and generates predictions for the next word in a sentence.

---

## Architecture Models

### LSTM (Long Short-Term Memory)

**LSTM** is a special type of Recurrent Neural Network (RNN) designed to learn **long-range dependencies** in sequential data. It solves the vanishing gradient problem that standard RNNs suffer from.

#### How LSTM Works

An LSTM cell consists of **three gates** and a **cell state**:

```
Input Sequence → [Forget Gate] → [Input Gate] → [Cell State Update] → [Output Gate] → Hidden State → Next Word
```

| Gate | Function |
|------|----------|
| **Forget Gate** | Decides what information to discard from the previous cell state using a sigmoid activation. `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` |
| **Input Gate** | Decides which new values to update in the cell state. `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` |
| **Cell State** | Carries long-term memory across time steps, updated by forget and input gates. |
| **Output Gate** | Determines the next hidden state (short-term memory) to pass to the next step. `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` |

#### LSTM Architecture in This Project

```
Input Text
    ↓
Embedding Layer  (maps words → dense vectors)
    ↓
LSTM Layer 1     (learns sequential patterns)
    ↓
Dropout Layer    (prevents overfitting)
    ↓
LSTM Layer 2     (deeper context understanding)
    ↓
Dense Layer      (fully connected)
    ↓
Softmax Output   (probability over vocabulary)
    ↓
Predicted Word
```

#### Why LSTM for Text?

- Captures **long-range word dependencies** (e.g., subject-verb agreement across long sentences)
- Remembers relevant context over many time steps
- Well-suited for **language modeling**, text generation, and prediction tasks

---

### GRU (Gated Recurrent Unit)

**GRU** is a simplified version of LSTM introduced by Cho et al. (2014). It achieves similar performance with **fewer parameters** and **faster training**, making it an efficient alternative for sequence modeling.

#### How GRU Works

A GRU cell uses **two gates** instead of three:

```
Input Sequence → [Reset Gate] → [Update Gate] → Hidden State → Next Word
```

| Gate | Function |
|------|----------|
| **Reset Gate** | Controls how much of the previous hidden state to forget when computing the candidate hidden state. `r_t = σ(W_r · [h_{t-1}, x_t])` |
| **Update Gate** | Decides how much of the past hidden state to keep vs. replace with the new candidate state. `z_t = σ(W_z · [h_{t-1}, x_t])` |
| **Candidate Hidden State** | A new hidden state proposal based on reset gate output. |

The final hidden state is a linear interpolation between the previous state and the candidate:
```
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

#### GRU Architecture in This Project

```
Input Text
    ↓
Embedding Layer   (maps words → dense vectors)
    ↓
GRU Layer 1       (sequential pattern learning)
    ↓
Dropout Layer     (regularization)
    ↓
GRU Layer 2       (deeper context)
    ↓
Dense Layer       (fully connected)
    ↓
Softmax Output    (probability distribution over vocabulary)
    ↓
Predicted Word
```

#### Why GRU for Text?

- **Faster to train** than LSTM due to fewer parameters
- Performs comparably on shorter sequences
- Better choice when **computational resources are limited**
- Less prone to overfitting on smaller datasets

---

### LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|---|---|---|
| Gates | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Long-range Dependencies | Better | Good |
| Memory Cell | Separate cell state | No separate cell state |
| Best For | Long sequences, complex data | Shorter sequences, faster training |
| Overfitting Risk | Higher (more params) | Lower |

---

## NLP Pipeline

The project uses standard NLP preprocessing to prepare raw text for model training:

```
Raw Text
    ↓
Text Cleaning          (lowercasing, punctuation removal)
    ↓
Tokenization           (splitting text into individual words/tokens)
    ↓
Vocabulary Building    (creating word-to-index mappings)
    ↓
Sequence Generation    (creating input-output word pairs with sliding window)
    ↓
Padding                (ensuring uniform sequence lengths)
    ↓
Embedding Layer        (dense vector representation of words)
    ↓
Model Training
```

### Key NLP Concepts Used

**Tokenization** — Breaking text into individual words or subword units that the model can process numerically.

**Vocabulary Mapping** — Each unique word is assigned an integer index. The model works with these indices rather than raw text.

**N-gram Sequence Generation** — Sliding windows over the training text create input-output pairs. For example, given "The cat sat on", the model learns to predict "on" from ["The", "cat", "sat"].

**Word Embeddings** — Words are mapped to dense, low-dimensional vectors that capture semantic relationships. Words with similar meanings appear closer together in embedding space.

**Softmax Output** — The final layer outputs a probability distribution over the entire vocabulary. The word with the highest probability is selected as the prediction.

---

## Use Cases

This Next Word Predictor has real-world applications across multiple domains:

### 1. 📱 Mobile Keyboard Autocomplete
Predicts the next word as the user types, speeding up text input on smartphones and tablets (similar to SwiftKey or Gboard).

### 2. ✍️ Smart Writing Assistants
Assists authors, bloggers, and content creators by suggesting contextually relevant next words, reducing writer's block and improving writing speed.

### 3. 📧 Email / Message Autocomplete
Powers smart reply and next-word suggestions in email clients (like Gmail's Smart Compose) and messaging applications.

### 4. 🔍 Search Engine Query Suggestions
Suggests query completions as a user types in a search bar, improving search experience and discoverability.

### 5. 🏥 Medical / Legal Document Assistance
Domain-specific models trained on medical or legal corpora can assist professionals in drafting standardized documents faster.

### 6. 🌐 Language Learning Applications
Helps language learners by providing contextually appropriate word suggestions, reinforcing vocabulary and grammar patterns.

### 7. ♿ Accessibility Tools
Assists users with disabilities (e.g., motor impairments) by reducing the number of keystrokes needed to compose messages.

### 8. 💬 Chatbot Response Generation
Powers conversational AI systems by predicting likely next tokens in dialogue generation pipelines.

---

## Project Structure

```
Next_Word_Predictor/
│
├── next_word_pred/
│   ├── next_word_predictor.ipynb   # Main Jupyter notebook
│   ├── model_lstm.h5               # Saved LSTM model
│   ├── model_gru.h5                # Saved GRU model
│   └── tokenizer.pkl               # Saved tokenizer object
│
└── README.md
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework for LSTM & GRU models |
| **NumPy** | Numerical computation |
| **Pandas** | Data handling and preprocessing |
| **NLTK / RegEx** | NLP preprocessing (tokenization, cleaning) |
| **Matplotlib** | Training visualization (loss/accuracy curves) |
| **Jupyter Notebook** | Interactive development and experimentation |

---

## How It Works

1. **Data Ingestion** — Raw text corpus is loaded and cleaned.
2. **Tokenization** — Text is tokenized and converted to integer sequences using Keras `Tokenizer`.
3. **Sequence Preparation** — N-gram sequences are generated. Each sequence's last word is the target label.
4. **Padding** — All sequences are padded to the same length using `pad_sequences`.
5. **Model Training** — Both LSTM and GRU models are trained on the prepared sequences.
6. **Prediction** — Given an input seed text, the model predicts the top-N most probable next words using the trained model.

```python
# Example usage
seed_text = "The quick brown"
next_word = predict_next_word(model, tokenizer, seed_text)
print(f"Next word: {next_word}")
# Output: Next word → "fox"
```

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas nltk matplotlib jupyter
```

### Run the Notebook

```bash
git clone https://github.com/ruderapratap/Next_Word_Predictor.git
cd Next_Word_Predictor
jupyter notebook next_word_pred/next_word_predictor.ipynb
```

---

## Results

Both models were trained and evaluated on the same corpus. Key observations:

- **LSTM** achieved slightly better accuracy on longer input sequences due to its stronger long-term memory.
- **GRU** trained faster and performed comparably on shorter sequences with less memory overhead.
- Word embeddings learned meaningful semantic relationships from the training data.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Rudera Pratap**  
[![GitHub](https://img.shields.io/badge/GitHub-ruderapratap-black?logo=github)](https://github.com/ruderapratap)

---

> ⭐ If you found this project useful, please consider giving it a star!
