import os 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# custom function to load data from directory
def load_data_from_directory(path):
    image_paths = []
    labels = []
    for class_directory in os.listdir(path):
        image_file_names = os.listdir(os.path.join(path, class_directory))
        for image_file_name in image_file_names:
            image_paths.append(os.path.join(path, class_directory, image_file_name))
            labels.append(class_directory)
    return image_paths, labels


def split_into_K_folds(x, y, k=5):
    k = 5  # number of folds
    kf = KFold(n_splits=k, shuffle=True)

    folds = []
    for train_index, test_index in kf.split(x):
        train_data = [x[i] for i in train_index]
        train_labels = [y[i] for i in train_index]
        test_data = [x[i] for i in test_index]
        test_labels = [y[i] for i in test_index]
        folds.append((train_data, train_labels, test_data, test_labels))
    return folds

def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
 
    return accuracy, recall, precision, f1