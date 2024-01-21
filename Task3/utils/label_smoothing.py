import tensorflow as tf

# Function to apply label smoothing to a batch of labels
def label_smooth(y_true, epsilon=0.1):
    K = tf.keras.backend
    num_classes = 8
    # Create smooth labels
    smooth_labels = (1.0 - epsilon) * y_true + epsilon / num_classes
    
    return smooth_labels

