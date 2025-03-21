import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import SGD, Adam ,RMSprop
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchviz import make_dot

spam = pd.read_csv('./hyperparametertuning/clinical_data/diabetes.csv')

X = spam.iloc[:,0: 8].values
y = spam.Outcome.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

scales = MinMaxScaler(feature_range=(0, 1)) 
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.fit_transform(X_test)




nb_classes = 2
output_dim = nb_classes

def mlp_objective_function_Diabete(param, N_repeat = 5): # 1 batch_size, 2 learning_rate, 3 learning_rate_decay
    parameter_range = [[32, 128], [1e-5, 1e-2], [1e-6, 1e-3],[4, 16]]

    batch_size_param_ = param[0]
    batch_size_param = batch_size_param_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
    batch_size_param = int(np.round(batch_size_param))
    
    learning_rate_ = param[1]
    learning_rate = learning_rate_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    
    learning_rate_decay_ = param[2]
    learning_rate_decay = learning_rate_decay_ * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]

    hidden_dim_param = output_dim * (param[3] * (parameter_range[3][1] - parameter_range[3][0]) + parameter_range[3][0])
    hidden_dim_param = int(np.round(hidden_dim_param))
    
    batch_size_param = batch_size_param
    learning_rate = learning_rate
    learning_rate_decay = learning_rate_decay
    hid_dim = hidden_dim_param
    
    error_all = np.zeros(N_repeat)
    for repeat in range(N_repeat):
        
        class myMLP(nn.Module):
            def __init__(self):
                super(myMLP, self).__init__()
    
                self.hidden1 = nn.Sequential(
                    nn.Linear(in_features=8, 
                              out_features=hid_dim, 
                              bias=True 
                              ),
                    nn.ReLU()
                )
                self.classify = nn.Sequential(
                    nn.Linear(hid_dim, 2), 
                    nn.Sigmoid()
                )  

            def forward(self, x):
                fc1 = self.hidden1(x)
                output = self.classify(fc1)
                return fc1,  output

        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        X_test_t = torch.from_numpy(X_test.astype(np.float32))
        y_test_t = torch.from_numpy(y_test.astype(np.int64))

        train_data_nots = Data.TensorDataset(X_train_t, y_train_t)

        train_nots_loader = Data.DataLoader(
            dataset = train_data_nots, 
            batch_size = batch_size_param, 
            shuffle = True, 
            num_workers = 1,
        )

        testnet = myMLP()
        optimizer = RMSprop(testnet.parameters(), lr=learning_rate , weight_decay = learning_rate_decay)
        loss_func = nn.CrossEntropyLoss()

        _, output = testnet(X_test_t)
        _, pre_lab = torch.max(output, 1)
        test_accuracy = accuracy_score(y_test_t, pre_lab)

        error_all[repeat] = 1 - test_accuracy

    return np.average(error_all)    

if __name__ == '__main__':
    
    import time
    start = time.time()

    input_query = np.random.random(4)
    output_observation = mlp_objective_function_Diabete(input_query)
