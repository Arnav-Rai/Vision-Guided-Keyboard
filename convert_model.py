import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
import json

# --- Configuration ---
KERAS_MODEL_PATH = 'my_char_lstm_model.h5'  # Input Keras model weights file
CHAR_MAP_PATH = 'char_mappings.json'       # Path to character mappings and model config
TFJS_MODEL_OUTPUT_DIR = 'tfjs_model_output_directory' # Output directory for TF.js model

def rebuild_and_convert_keras_to_tfjs(keras_weights_path, char_map_path, output_dir):
    """
    Rebuilds the Keras model architecture, loads weights, and converts to TensorFlow.js.
    """
    if not os.path.exists(keras_weights_path):
        print(f"Error: Keras model weights file not found at '{keras_weights_path}'.")
        print("Please ensure you have successfully run the training script first.")
        return

    if not os.path.exists(char_map_path):
        print(f"Error: Character mapping file not found at '{char_map_path}'.")
        print("This file is required to rebuild the model architecture.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    print(f"Loading character mappings and model config from '{char_map_path}'...")
    try:
        with open(char_map_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        vocab_size = mappings['vocab_size']
        sequence_length = mappings['sequence_length']
        # Assuming these were standard in your training script, if not, add them to char_mappings.json
        embedding_dim = mappings.get('embedding_dim', 64) # Default if not in JSON
        lstm_units = mappings.get('lstm_units', 128)      # Default if not in JSON

        print("Character mappings and model config loaded successfully.")
        print(f"  Vocab Size: {vocab_size}, Sequence Length: {sequence_length}")
        print(f"  Embedding Dim: {embedding_dim}, LSTM Units: {lstm_units}")

    except Exception as e:
        print(f"\n--- Error loading character mappings ---")
        print(f"An error occurred: {e}")
        return

    print("Rebuilding Keras model architecture...")
    try:
        # Define the model architecture EXACTLY as in your training script
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
            LSTM(lstm_units),
            Dense(vocab_size, activation='softmax')
        ])
        print("Model architecture rebuilt.")
        
        print(f"Loading weights from '{keras_weights_path}' into the rebuilt model...")
        model.load_weights(keras_weights_path)
        print("Weights loaded successfully into the rebuilt model.")

    except Exception as e:
        print("\n--- Model Rebuilding or Weight Loading Failed ---")
        print(f"An error occurred: {e}")
        return

    print(f"Starting conversion of the rebuilt Keras model to TensorFlow.js format...")
    print(f"Output will be saved in '{output_dir}'")

    try:
        tfjs.converters.save_keras_model(model, output_dir)
        print("\n--- Conversion Complete ---")
        print(f"TensorFlow.js model files (model.json and .bin files) should now be in the '{output_dir}' directory.")
    except Exception as e:
        print("\n--- Conversion Failed ---")
        print(f"An error occurred during conversion: {e}")

if __name__ == '__main__':
    current_dir = os.getcwd()
    keras_weights_full_path = os.path.join(current_dir, KERAS_MODEL_PATH)
    char_map_full_path = os.path.join(current_dir, CHAR_MAP_PATH)
    tfjs_output_full_path = os.path.join(current_dir, TFJS_MODEL_OUTPUT_DIR)
    
    rebuild_and_convert_keras_to_tfjs(keras_weights_full_path, char_map_full_path, tfjs_output_full_path)
