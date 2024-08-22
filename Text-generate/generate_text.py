import tensorflow as tf
from Text_generator import TextGenerator

def main():
    # Constants
    SEQ_LENGTH = 40
    TEXT_FILE_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    MODEL_FILE = 'textGenerator.keras'

    # Load text
    filepath = tf.keras.utils.get_file('ReSharper.txt', TEXT_FILE_URL)
    text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
    text = text[30000:80000]  # Truncate the text for the example

    # Initialize TextGenerator
    text_gen = TextGenerator(text, MODEL_FILE, seq_length=SEQ_LENGTH)

    # Generate text with different temperatures
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f'------------{temp}------------')
        print(text_gen.generate_text(length=300, temperature=temp))

if __name__ == '__main__':
    main()
