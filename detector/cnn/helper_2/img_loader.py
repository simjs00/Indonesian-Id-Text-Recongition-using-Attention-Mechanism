import os
from scipy import misc
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

def get_image_and_labels(dataset, portion_train, portion_val):
    imgtrn = []
    lbltrn = []
    imgval = []
    lblval = []

    for i in range(len(dataset)):
        img_paths_flat = dataset[i].image_paths

        lbls_flat = [i] * len(dataset[i].image_paths)
        
        imgtrn += img_paths_flat[:int(len(img_paths_flat) * portion_train)]

        lbltrn += lbls_flat[:int(len(img_paths_flat) * portion_train)]

        imgval += img_paths_flat[-int(len(img_paths_flat) * portion_val):]

        lblval += lbls_flat[-int(len(img_paths_flat) * portion_val):]


    return imgtrn, lbltrn, imgval, lblval


def get_class(paths) :
        classes = []
        for path in paths.split(':'):
            path_exp = os.path.expanduser(path)
            classes = os.listdir(path_exp)

        classes.sort()
        return classes
def prewhiten1(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def prewhiten(X):

    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    #normalization 

    mean = np.mean(X)
    std=np.std(X)
    with np.errstate(invalid='ignore'):
        X = np.divide(np.subtract(X, mean), std)
    if(np.isnan(X).any()==True) :
        return 0
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 0.1
    # ZCA Whitening matrix: U * Lambda * U'

    P = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    M = np.mean(X)
    xZCAMatrix = np.dot(P, np.subtract(X, M)) 
    return xZCAMatrix

def read_data1(image_list, image_size):
    image_batch = []
    for path in image_list:
        im = cv2.imread(path)
        im = cv2.resize(im, (image_size, image_size))
        #im = prewhiten(im)
        image_batch.append(im)
    return image_batch
def removal_text(gray, image_size) :
    

    #gray = cv2.equalizeHist(img)
    #gray = cv2.equalizeHist(gray)
    #gray = cv2.bilateralFilter(gray,10,97,75)
    
    #gray =cv2.normalize(gray,  gray, 0, 255, cv2.NORM_MINMAX)
    
    #kernel = np.ones((1,1), np.uint8)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    #ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    # for i in range(len((gray)) ):
    #     for j in range(len(thresh[i])) :
    #         if(thresh[i][j]==255) :
    #             gray[i][j]=255

    #thresh = cv2.Canny(img,100,200)
    # edges1 = cv2.Canny(gray,100,200)
    # cv2.imshow('edges',edges)
    # cv2.imsobel=cv2.normalize(sobel,sobel,0,255,cv2.NORM_MINMAX)show('edges1',edges1)
    #gray=cv2.GaussianBlur(gray, (7, 7), 0)
    # gray = clahe.apply(gray)
    # gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )
    #gray = cv2.Canny(gray,127,255)
    gray = prewhiten(gray)

    #gray =cv2.normalize(gray,  gray, 0, 255, cv2.NORM_MINMAX)
   
    #gray = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    if(np.isscalar(gray) ==True ) :
        return 0    

    return gray
def read_data(image_list, image_size):
    count=0
    image_batch = []
    for path in image_list:
        #print(path)
        im = cv2.imread(path)

        if(im is not None) :
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im,(image_size,image_size))
            im = removal_text(im,image_size)
            if(np.isscalar(im)==True ) :
                im = np.zeros((image_size, image_size))
                os.remove(path)
                print("remove file")
                #im =np.zeros((32,32))
                #count+=1

        #print(path)
        #im = misc.imresize(im, (image_size, image_size))
        # im = cv2.resize(im, (image_size, image_size))
        # im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        #im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        #im = np.reshape(im, (np.product(im.shape),))
        # im = prewhiten(im)

        # im = np.reshape(im, (image_size,image_size,1))
        else :
            print("remove dir")
            os.rmdir(path) 
            im = np.zeros((image_size, image_size))
            #count+=1

        im = np.reshape(im, (image_size,image_size,1))
        image_batch.append(im)
    
        #plt.imshow(prewhiten(im))
        #plt.show()
    return image_batch


def read_data_test(im,image_size):
        image_batch = []

        im = cv2.resize(im,(image_size,image_size))

        im = removal_text(im,image_size) 

        #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 
        from time import gmtime, strftime
        from datetime import datetime
        dt = datetime.now()
        dt=dt.microsecond
        path = "./bnw/"+str(strftime("%Y-%m-%d_%H:%M:%S:", gmtime()))+str(dt)+".png"
        # cv2.imshow('gray',im)
        # cv2.waitKey(100000)
        

        #im = prewhiten(im)
        cv2.imwrite(path,im)   
        im = np.reshape(im, (image_size,image_size,1))
        image_batch.append(im)
        return image_batch


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', '+str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []

    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)

        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            chardir = os.path.join(path_exp, class_name)
            if os.path.isdir(chardir):
                image_paths =[]
                if(class_name=='+') :
                    images = os.listdir(chardir)
                    for j,img in enumerate(images) :
                        img=os.path.join(chardir, img)
                        img_2 = os.listdir( img)
                        for k,pth in enumerate(img_2) :
                                image_paths.append(os.path.join(img, pth)) 
                else :
                        images = os.listdir(chardir)
                        #image_paths = [os.path.join(chardir, img) for img in images]
                        for j,img in enumerate(images) :
                            image_paths.append(os.path.join(chardir, img)) 


                dataset.append(ImageClass(class_name, image_paths))

    return dataset,classes

