import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

#load image
def load_image(filepath='./Background_CAU.jpg'):
    image_raw = imread(filepath)

    # Displaying an image in grayscale.
    image_sum = image_raw.sum(axis=2)
    X         = image_sum/image_sum.max() #treat X as a D x N matrix (treat each column as a sample and each row as a feature)
    X_mean    = np.mean(X, axis=1, keepdims=True) 

    plt.figure(figsize=[12,8])
    plt.imshow(X, cmap=plt.cm.gray)
    plt.title('Original Image', fontsize=15)

    plt.show()
    plt.close()

    return X, X_mean

# compute covariance matrix
def compute_cov_mat(X, X_mean):
    ### Fill in ################################
    X_shifted = X - X_mean
    cov_mat  = np.matmul(X_shifted, np.transpose(X_shifted)) / X.shape[1]
    ############################################    
    return cov_mat


if __name__ == "__main__":

    X, X_mean  = load_image('./Background_CAU.jpg')
    cov_mat    = compute_cov_mat(X, X_mean)
    
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    ### sort from highest to lowest eigenvalues
    eigen_vectors = eigen_vectors[:, np.argsort(-eigen_values)]
    eigen_values  = eigen_values[np.argsort(-eigen_values)]
    
    ### Compute the projectoin matrix B_10, B_50, B_100, B_500, B_1000
    ### Fill in ################################
    B_10   = eigen_vectors[:, :10] 
    B_50   = eigen_vectors[:, :50]
    B_100  = eigen_vectors[:, :100]
    B_500  = eigen_vectors[:, :500]
    B_1000 = eigen_vectors[:, :1000]
    ############################################    

    ### Reconstructe image: make X_tilde using B_10, B_50, B_100, B_500, B_1000
    ### Don't forget PCA is performed on the mean-shifted image. So you have to add the mean after reconstruction!
    ### Fill in ################################
    X_shifted = X - X_mean
    X_tilde_10    = np.matmul(np.matmul(B_10, np.transpose(B_10)), X_shifted) + X_mean
    X_tilde_50    = np.matmul(np.matmul(B_50, np.transpose(B_50)), X_shifted) + X_mean
    X_tilde_100   = np.matmul(np.matmul(B_100, np.transpose(B_100)), X_shifted) + X_mean
    X_tilde_500   = np.matmul(np.matmul(B_500, np.transpose(B_500)), X_shifted) + X_mean
    X_tilde_1000  = np.matmul(np.matmul(B_1000, np.transpose(B_1000)), X_shifted) + X_mean
    ############################################   

    for K in [10, 50, 100, 500, 1000]:
        plt.figure(figsize=[12,8])
        if K == 10:
            plt.imshow(X_tilde_10, cmap=plt.cm.gray)
        elif K == 50:
            plt.imshow(X_tilde_50, cmap=plt.cm.gray)
        elif K == 100:
            plt.imshow(X_tilde_100, cmap=plt.cm.gray)
        elif K == 500:
            plt.imshow(X_tilde_500, cmap=plt.cm.gray)
        elif K == 1000:
            plt.imshow(X_tilde_1000, cmap=plt.cm.gray)
            
        plt.title('Reconstructed Image (K={})'.format(K), fontsize=15)

        plt.show()
        plt.close()