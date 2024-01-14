import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from itertools import zip_longest
import ast
import pandas as pd
import numpy as np
import os
import seaborn as sn
from sklearn.metrics import confusion_matrix
from itertools import zip_longest
import ast


import matplotlib.pyplot as plt

def plot_mean_values(column_name):
    logs = pd.read_csv('logs.csv', index_col=0)
    
    unique_values = logs[column_name].unique()
    
    mean_values = []
    
    for value in unique_values:
        df = logs[logs[column_name] == value]
        val_accuracies = df['ValAccuracy'].values
        val_accuracies = [ast.literal_eval(x) for x in val_accuracies]
        
        mean_value = []
        for values in zip_longest(*val_accuracies, fillvalue=np.nan):
            mean_value.append(np.nanmean(values))
        
        mean_values.append(mean_value)
    
    for i, value in enumerate(unique_values):
        plt.plot(mean_values[i], label=f"Model with {column_name} = {value}")
    
    plt.legend()
    plt.savefig(f"{column_name}.png")
    plt.show()

# Example usage:
plot_mean_values('RegCoeff')
