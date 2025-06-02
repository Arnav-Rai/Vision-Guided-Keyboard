import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import json
import os

# --- Configuration ---
FILE_PATH = 'common_words.txt'  # Path to your word list
SEQUENCE_LENGTH = 20  # Length of input character sequences
EMBEDDING_DIM = 64    # Dimension of character embeddings
LSTM_UNITS = 128      # Number of units in the LSTM layer
EPOCHS = 50           # Number of training epochs (adjust based on dataset size and desired accuracy)
BATCH_SIZE = 128      # Number of samples per gradient update
MODEL_SAVE_PATH = 'my_char_lstm_model.h5'
CHAR_MAP_SAVE_PATH = 'char_mappings.json'

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")

try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found. Please make sure it's in the same directory or provide the correct path.")
    # Using a small fallback list if the file is not found, for demonstration
    words = ["hello", "world", "adaptive", "keyboard", "python", "tensorflow", "learning", "model", "suggestion"]
    print(f"Using fallback word list: {words}")


# Clean words: lowercase and filter out empty strings
words = [word.lower().strip() for word in words if word.strip()]

# Create a single string of all text, separated by a unique character if needed,
# or just process word by word. For character-level, concatenating is fine.
# Adding a space between words can help the model learn word boundaries.
text = " ".join(words) # Add spaces to help learn word separation

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size (number of unique characters): {vocab_size}")
print(f"Characters in vocabulary: {''.join(chars)}")

# Create character-to-index and index-to-character mappings
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# --- 2. Generate Training Sequences ---
print("Generating training sequences...")
input_sequences = []
target_chars = []

# Iterate through the text to create sequences
# We use a step of 1 to generate many overlapping sequences
for i in range(0, len(text) - SEQUENCE_LENGTH, 1):
    # Input sequence is characters from i to i + SEQUENCE_LENGTH - 1
    seq_in = text[i:i + SEQUENCE_LENGTH]
    # Target character is the character at i + SEQUENCE_LENGTH
    seq_out = text[i + SEQUENCE_LENGTH]
    
    input_sequences.append([char_to_index[char] for char in seq_in])
    target_chars.append(char_to_index[seq_out])

num_sequences = len(input_sequences)
print(f"Total number of sequences generated: {num_sequences}")

if num_sequences == 0:
    print("Error: No sequences were generated. This might happen if your text is shorter than SEQUENCE_LENGTH.")
    print("Please check your common_words.txt or reduce SEQUENCE_LENGTH.")
    exit()

# --- 3. Prepare Data for the Model ---
print("Preparing data for the model (vectorization)...")

# Convert input sequences to numpy array
X = np.array(input_sequences)

# One-hot encode the target characters
# y will have shape (num_sequences, vocab_size)
y = to_categorical(target_chars, num_classes=vocab_size)

print(f"Shape of input data (X): {X.shape}")    # Should be (num_sequences, SEQUENCE_LENGTH)
print(f"Shape of target data (y): {y.shape}")  # Should be (num_sequences, vocab_size)


# --- 4. Define the LSTM Model Architecture ---
print("Defining the LSTM model architecture...")
model = Sequential([
    # Embedding layer: Turns character IDs into dense vectors.
    # input_length is SEQUENCE_LENGTH because we are feeding the full sequence.
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    
    # LSTM layer: Learns sequential patterns.
    LSTM(LSTM_UNITS),
    
    # Dense output layer: Predicts the probability of each character in the vocab being the next one.
    Dense(vocab_size, activation='softmax')
])

# --- 5. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 6. Train the Model ---
print(f"Training the model for {EPOCHS} epochs...")
# Ensure you have enough data for the batch size
if num_sequences < BATCH_SIZE:
    print(f"Warning: Number of sequences ({num_sequences}) is less than batch size ({BATCH_SIZE}). Adjusting batch size.")
    BATCH_SIZE = max(1, num_sequences // 2 if num_sequences > 1 else 1)


history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# --- 7. Save the Trained Model and Mappings ---
print("Saving the trained model and character mappings...")

# Save the Keras model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Save character mappings
# In train_char_model.py
mappings = {
    'char_to_index': char_to_index,
    'index_to_char': index_to_char,
    'sequence_length': SEQUENCE_LENGTH, # This is the input_length for Embedding
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,     # Add this
    'lstm_units': LSTM_UNITS          # Add this
}
with open(CHAR_MAP_SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(mappings, f)
print(f"Character mappings saved to {CHAR_MAP_SAVE_PATH}")

print("\n--- Training Complete ---")
print("Next steps:")
print(f"1. You should now have '{MODEL_SAVE_PATH}' and '{CHAR_MAP_SAVE_PATH}'.")
print("2. Convert the Keras model to TensorFlow.js format using the command:")
print(f"   tensorflowjs_converter --input_format keras ./{MODEL_SAVE_PATH} ./tfjs_model_output_directory")
print("3. Integrate the converted TF.js model and the char_mappings.json into your JavaScript application.")

# --- Optional: Example of how to generate text with the trained model (for testing) ---
def generate_text(seed_text, num_chars_to_generate=50):
    print(f"\n--- Generating text from seed: '{seed_text}' ---")
    generated_output = seed_text.lower()
    current_input = seed_text.lower()

    for _ in range(num_chars_to_generate):
        # Prepare the current input sequence
        sequence_ids = [char_to_index.get(char, 0) for char in current_input[-SEQUENCE_LENGTH:]] # Use 0 for unknown chars
        padded_sequence = pad_sequences([sequence_ids], maxlen=SEQUENCE_LENGTH, padding='pre')
        
        # Predict the next character
        predicted_probabilities = model.predict(padded_sequence, verbose=0)[0]
        
        # Sample an index from the probabilities (could also use np.argmax for greedy)
        # Using np.random.choice allows for some randomness based on probabilities
        predicted_index = np.random.choice(len(predicted_probabilities), p=predicted_probabilities)
        # Alternatively, for greedy prediction:
        # predicted_index = np.argmax(predicted_probabilities)

        predicted_char = index_to_char.get(predicted_index, '?') # Use '?' for unknown index
        
        generated_output += predicted_char
        current_input += predicted_char # Append predicted char for next input
        
    print(generated_output)
    print("--- End of generation ---")

# Example usage of text generation (uncomment to test after training)
# if num_sequences > 0: # Only if model was trained
#     test_seed = "keyb" 
#     if all(c in char_to_index for c in test_seed): # Ensure seed chars are in vocab
#         generate_text(test_seed, 50)
#     else:
#         print(f"Cannot generate text from seed '{test_seed}' as it contains characters not in the vocabulary.")
#     generate_text(list(char_to_index.keys())[0] * SEQUENCE_LENGTH, 50) # Generate from a sequence of the first char

