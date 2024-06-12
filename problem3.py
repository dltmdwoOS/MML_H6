import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt


def PCA(X , num_components):
    '''
        X: D x N matrix (to be consistant with the description in the textbook)
    '''
    #Step-1
    X_mean    = np.mean(X, axis=1, keepdims=True)  #treat X as a DxN matrix (treat each column as a sample and each row as a feature)

    ### Fill in ################################
    #Step-2
    X_shifted = X - X_mean
    cov_mat  = np.matmul(X_shifted, np.transpose(X_shifted)) / X.shape[1]
    ############################################    
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    
    ### sort from highest to lowest eigenvalues
    eigen_vectors = eigen_vectors[:, np.argsort(-eigen_values)] 
    eigen_values  = eigen_values[np.argsort(-eigen_values)]
    
    ### Fill in ################################
    B         = eigen_vectors[:, :num_components]
    Z         = np.matmul(np.transpose(B), X_shifted)
    ############################################    
    
    return Z


if __name__ == "__main__":

    #Get the IRIS dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

    #prepare the data
    x = np.asarray(data.iloc[:,0:4].T)  #change dimension from NxD -> DxN

    #prepare the target
    target = data.iloc[:,4]
       

    #Applying it to PCA function
    z = PCA(x , 2)

    #Creating a Pandas DataFrame of latent codes
    latent_code = pd.DataFrame(z.T , columns = ['PC1','PC2'])

    #Concat it with target variable to create a complete Dataset
    latent_code = pd.concat([latent_code , pd.DataFrame(target)] , axis = 1)


    ### Visualize data in PCA dimensions
    plt.figure(figsize = (6,6))
    sb.scatterplot(data = latent_code , x = 'PC1',y = 'PC2' , hue = 'target' , s = 60 , palette= 'icefire')

    plt.grid()
    plt.show()
    plt.close()