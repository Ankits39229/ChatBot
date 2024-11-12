import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
from colorama import Fore, Style
import random
import pickle
import os

colorama.init()

# Define base path for your project
BASE_PATH = r"C:\Users\Bihar\Desktop\chatBot"

# Load intents data
with open(os.path.join(BASE_PATH, "intents.json")) as file:
    data = json.load(file)

def save_interaction(user_input, bot_response, intent):
    history_path = os.path.join(BASE_PATH, "interaction_history.json")
    new_entry = {
        "intent": intent,
        "user_input": user_input,
        "bot_response": bot_response
    }
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        history.append(new_entry)
    else:
        history = [new_entry]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

def retrain_model():
    # Load interaction history for new data
    history_path = os.path.join(BASE_PATH, "interaction_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            interaction_history = json.load(f)
            data["intents"].extend(interaction_history)
            
        # Code to retrain model with updated data
        # ...
        
        # After retraining, save the updated model
        # model.save(os.path.join(BASE_PATH, r"models\chat_model.keras"))
        
        # Clear interaction history after retraining
        os.remove(history_path)

def chat():
    model = keras.models.load_model(os.path.join(BASE_PATH, r"models\chat_model.keras"))

    with open(os.path.join(BASE_PATH, r"models\tokenizer.pickle"), 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(os.path.join(BASE_PATH, r"models\label_encoder.pickle"), 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

        for i in data['intents']:
            if i['intent'] == tag:
                response = np.random.choice(i['responses'])
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, response)
                save_interaction(inp, response, tag)  # Save each interaction for future learning

    retrain_model()  # Retrain model after each session (optional)

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
