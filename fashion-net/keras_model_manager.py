import json
import csv
from keras.models import model_from_json, load_model, model_from_yaml

class Save(object):
    """Helper class for saving Keras models"""

    @staticmethod
    def save_model_to_h5(model, save_directory_path, model_name):
        print("[INFO] serializing network...")
        model.save('{0}/{1}_model.h5'.format(save_directory_path, model_name))
        print("[INFO] complete")

    @staticmethod
    def save_model_to_json(model, save_directory_path, model_name):
        print("[INFO] serializing model to JSON...")
        # serialize model to JSON
        model_json = model.to_json()
        with open('{0}/{1}_model.json'.format(save_directory_path, model_name), "w") as json_file:
            json_file.write(model_json)
        print("[INFO] complete")

    @staticmethod
    def save_model_to_yaml(model, save_directory_path, model_name):
        print("[INFO] serializing model to YAML...")
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open('{0}/{1}_labels.yml'.format(save_directory_path, model_name), 'w') as yaml_file:
            yaml_file.write(model_yaml)
        print("[INFO] complete")

    @staticmethod
    def save_history(history, save_directory_path, model_name):
        print("[INFO] serializing history to JSON...")
        # Get the dictionary containing each metric and the loss for each epoch
        # Save it under the form of a json file
        json.dump(history.history, open('{0}/{1}_history.json'.format(save_directory_path, model_name), 'w'))
        print("[INFO] complete")

    @staticmethod
    def save_classes_to_csv(classes, save_directory_path, model_name):
        print("[INFO] serializing label classes to CSV...")
        with open('{0}/{1}_labels.json'.format(save_directory_path, model_name), 'w', newline='') as file:
            writer = csv.writer(file)
            for classify in classes:
                writer.writerow(classify)
        print("[INFO] complete")

    @staticmethod
    def save_weights(model, save_directory_path, model_name):
        print("[INFO] serializing weights...")
        model.save_weights('{0}/{1}_weights.h5'.format(save_directory_path, model_name))
        print("[INFO] complete")

class Load(object):
    """Helper class for load Keras models"""

    @staticmethod
    def load_model_from_h5(model_path):
        return load_model(model_path)

    @staticmethod
    def load_model_from_json(model_path, weights_path, optimizer):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        return model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    @staticmethod
    def load_model_from_yaml(model_path, weights_path, optimizer):
        # load YAML and create model
        yaml_file = open('model.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        model.load_weights(weights_path)
        return model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    @staticmethod
    def load_classes_from_csv(class_pass):
        with open(class_pass, 'r') as f:
            return list(csv.reader(f))
