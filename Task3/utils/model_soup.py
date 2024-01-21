import tensorflow as tf
from tensorflow.keras.models import load_model

# List of model file paths
model_paths = ['model1.h5', 'model2.h5', 'model3.h5']

# Load models
models = [load_model(path) for path in model_paths]

# Initialize an empty model with the same architecture as the loaded models
averaged_model = load_model('model1.h5')  # Load a model just to get the architecture
averaged_model.set_weights(models[0].get_weights())  # Set initial weights

# Perform weight averaging
for i in range(1, len(models)):
    model_weights = models[i].get_weights()
    averaged_weights = averaged_model.get_weights()

    # Calculate the average of the weights
    averaged_weights = [w1 + w2 for w1, w2 in zip(averaged_weights, model_weights)]

# Finalize the averaging by dividing by the number of models
averaged_weights = [w / len(models) for w in averaged_weights]

# Set the averaged weights to the new model
averaged_model.set_weights(averaged_weights)

# Save the averaged model
averaged_model.save('averaged_model.h5')