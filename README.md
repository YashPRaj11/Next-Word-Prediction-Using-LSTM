# Next Word Prediction using LSTM & RNN

A deep-learning based **next-word prediction system** trained on Shakespeare's *Hamlet*. The project uses TensorFlow/Keras to tokenize the text, create n-gram training sequences, train recurrent neural networks, and predict the next word from a user-provided sequence.

A **Streamlit web application** is included for interactive next-word prediction.

## Features

- Shakespeare's *Hamlet* used as the training corpus
- Automatic word-level tokenization
- N-gram sequence generation
- Pre-padding of variable-length sequences
- One-hot encoded target labels
- Train/test split using scikit-learn
- Recurrent neural-network architectures using:
  - LSTM
  - GRU
- Dropout regularization
- Early stopping with restoration of the best validation weights
- Saved Keras model and tokenizer
- Interactive Streamlit interface for next-word prediction

## Project Structure

```text
LSTM RNN/
│
├── app.py
├── experiemnts.ipynb
├── hamlet.txt
├── next_word_lstm.h5
├── next_word_lstm_model_with_early_stopping.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

> Note: The notebook is named `experiemnts.ipynb` in the original project.

## How It Works

The overall pipeline is:

```text
Hamlet Text
     ↓
Text Preprocessing
     ↓
Word Tokenization
     ↓
N-gram Sequence Generation
     ↓
Padding
     ↓
Input / Target Separation
     ↓
Train / Test Split
     ↓
LSTM / GRU Model
     ↓
Early Stopping
     ↓
Saved Model + Tokenizer
     ↓
Streamlit Application
     ↓
Next Word Prediction
```

### 1. Dataset

The project uses the **Shakespeare Hamlet** text available through the NLTK Gutenberg corpus.

The text is downloaded and saved locally as:

```text
hamlet.txt
```

### 2. Tokenization

Keras' `Tokenizer` converts words into integer IDs.

For example:

```text
"to be or not to be"
```

is converted into a sequence of integer token IDs.

The tokenizer is saved as:

```text
tokenizer.pickle
```

so that the same vocabulary can be used during inference.

### 3. N-gram Sequence Generation

For each line, progressively longer sequences are generated.

For example, a sentence such as:

```text
to be or not
```

can produce sequences conceptually similar to:

```text
to be
to be or
to be or not
```

The final token in each sequence becomes the prediction target, while the preceding tokens become the model input.

### 4. Padding

Because the generated sequences have different lengths, they are padded to a common sequence length using:

```python
pad_sequences(..., padding="pre")
```

The input is separated from the final word:

```python
x = input_sequences[:, :-1]
y = input_sequences[:, -1]
```

The target is then converted to one-hot encoded vectors.

## Model Architecture

### LSTM Model

The notebook defines an LSTM-based sequence model with:

```text
Embedding
   ↓
LSTM(150, return_sequences=True)
   ↓
Dropout(0.2)
   ↓
LSTM(100)
   ↓
Dense(total_words, softmax)
```

The model is compiled using:

```python
loss="categorical_crossentropy"
optimizer="adam"
metrics=["accuracy"]
```

### GRU Model

The notebook also defines a GRU-based recurrent model:

```text
Embedding
   ↓
GRU(150, return_sequences=True)
   ↓
Dropout(0.2)
   ↓
GRU(100)
   ↓
Dense(total_words, softmax)
```

Both architectures use a 100-dimensional embedding representation.

## Training

The dataset is divided into training and testing sets using an 80/20 split:

```python
train_test_split(x, y, test_size=0.2)
```

Training is configured for up to 50 epochs:

```python
model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)
```

### Early Stopping

The project uses:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

This stops training when validation loss does not improve for three consecutive epochs and restores the best-performing weights.

## Next-Word Prediction

During inference, the input sentence is:

1. Converted into token IDs.
2. Trimmed to the expected sequence length if necessary.
3. Pre-padded.
4. Passed through the trained model.
5. Converted into a probability distribution over the vocabulary.
6. The word with the highest predicted probability is selected.

The core prediction step is:

```python
predicted_word_index = np.argmax(predicted, axis=1)
```

The corresponding word is then recovered from the tokenizer vocabulary.

## Streamlit Application

The project includes `app.py`, which provides a simple web interface.

Run it with:

```bash
streamlit run app.py
```

The application provides an input box with an example prompt:

```text
To be or not to
```

After clicking **Predict Next Word**, the application displays the predicted word.

## Installation

### 1. Clone or download the project

Place all project files in the same directory.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project currently specifies:

```text
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras
```

If you run the notebook from scratch, you will also need NLTK and its Gutenberg corpus:

```bash
pip install nltk
```

Then download the corpus:

```python
import nltk
nltk.download("gutenberg")
```

## Running the Project

### Option 1: Use the Existing Trained Model

If `next_word_lstm.h5` and `tokenizer.pickle` are present:

```bash
streamlit run app.py
```

Then enter a text sequence and click **Predict Next Word**.

### Option 2: Train the Model Again

Open:

```text
experiemnts.ipynb
```

and execute the cells in order.

The notebook:

- prepares the Hamlet dataset
- tokenizes the text
- creates n-gram sequences
- builds the recurrent model
- trains it with early stopping
- tests next-word prediction
- saves the model
- saves the tokenizer

## Saved Files

### `next_word_lstm.h5`

Saved Keras model used by the Streamlit application.

### `next_word_lstm_model_with_early_stopping.h5`

Additional saved model artifact included with the project.

### `tokenizer.pickle`

Serialized Keras tokenizer containing the vocabulary and word-to-index mapping.

### `hamlet.txt`

Local copy of the Shakespeare *Hamlet* training corpus.

## Example

Input:

```text
To be or not to
```

The model predicts a single next word based on the learned language patterns.

Other example prompts:

```text
To be or not to
```

```text
Barn. Last night of all
```

Because the model is trained only on *Hamlet*, predictions are expected to reflect the vocabulary and writing style present in that corpus.

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **LSTM**
- **GRU**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **NLTK**
- **Streamlit**
- **Matplotlib**
- **TensorBoard**

## Important Implementation Note

The notebook defines both an LSTM model and a GRU model using the same `model` variable. The later GRU definition replaces the earlier LSTM object before the training cell is executed.

Therefore, when reproducing the notebook from top to bottom, the model that reaches the training and saving cells is the **GRU-based model**, despite the saved filename:

```text
next_word_lstm.h5
```

If the intention is specifically to train and save the LSTM model, train/save the LSTM model before creating the GRU model, or use separate variable names such as:

```python
lstm_model = Sequential(...)
gru_model = Sequential(...)
```

## Future Improvements

- Add top-k next-word predictions instead of returning only one word
- Display prediction probabilities
- Compare LSTM and GRU performance quantitatively
- Add temperature-based sampling for more diverse text generation
- Use a larger Shakespeare corpus
- Add model quantization for a smaller deployment footprint
- Export the trained model to TensorFlow Lite
- Improve the Streamlit interface
- Add text-generation mode for predicting multiple words
- Track and visualize training/validation loss and accuracy

## License

This project is intended for educational and research purposes. Check the applicable licensing and usage terms of the underlying dataset and software dependencies before redistributing the project.

---

**Built with TensorFlow, Keras, and Streamlit for sequence modeling and next-word prediction.**
