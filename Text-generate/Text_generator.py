import numpy as np
import random
import tensorflow as tf

class TextGenerator:
    def __init__(self, text, model_filepath, seq_length=40):
        self.text = text
        self.seq_length = seq_length
        self.char_to_index, self.index_to_char = self.create_mappings(text)
        self.model = tf.keras.models.load_model(model_filepath)

    @staticmethod
    def create_mappings(text):
        characters = sorted(set(text))
        char_to_index = {c: i for i, c in enumerate(characters)}
        index_to_char = {i: c for i, c in enumerate(characters)}
        return char_to_index, index_to_char

    @staticmethod
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)  # Normalize to get a probability distribution
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(self, length, temperature=1.0):
        start_index = random.randint(0, len(self.text) - self.seq_length - 1)
        generated = ''
        sentence = self.text[start_index: start_index + self.seq_length]
        generated += sentence
        
        for i in range(length):
            x = np.zeros((1, self.seq_length, len(self.char_to_index)))
            for t, character in enumerate(sentence):
                x[0, t, self.char_to_index[character]] = 1
                
            predictions = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(predictions, temperature)
            next_character = self.index_to_char[next_index]
            
            generated += next_character
            sentence = sentence[1:] + next_character
        
        return generated
