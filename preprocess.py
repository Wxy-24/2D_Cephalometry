import cv2
import numpy as np
import skimage.io as io
import os
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from math import*
from scipy.stats import multivariate_normal

#define path
train_path="RawImage/TrainingData"
label_path1="400_senior"
label_path2="400_junior"

def preprocess():

    # rename files
    for file in os.listdir(train_path):
        name = file.split('.')[0]
        os.rename(os.path.join(train_path, file), os.path.join(train_path,str(int(name)))+".bmp")
    for file in os.listdir(label_path1):
        name = file.split('.')[0]
        os.rename(os.path.join(label_path1, file), os.path.join(label_path1,str(int(name)))+".txt")
    for file in os.listdir(label_path2):
        name = file.split('.')[0]
        os.rename(os.path.join(label_path2, file), os.path.join(label_path2,str(int(name)))+".txt")

    # crop images
    for i in range(1,151):
        img = io.imread(os.path.join(train_path, "%d.bmp" % i),)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=gray[100:2324,0:1920]
        io.imsave(os.path.join(train_path, "%d_crop.bmp" % i), gray)
    for i in range(151,301):
        img = io.imread(os.path.join("RawImage/Test1Data", "%d.bmp" % i),)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=gray[100:2324,0:1920]
        io.imsave(os.path.join(train_path, "%d_crop.bmp" % i), gray)

#generate heatmap
def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))

def gaussian(array_like_hm, mean, sigma):
    """modified version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0] ** 2
    y_term = array_like_hm[:, 1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)

def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    k1 = multivariate_normal(mean=m1, cov=s1)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width))
    return img

def test(width, height, x, y, array_like_hm):
    dmax = 100
    edge_value = 0.01
    sigma = cal_sigma(dmax, edge_value)
    return draw_heatmap(width, height, x, y, sigma, array_like_hm)

def create_dataset(num_samples,test_ratio,validate_ratio):

    #define heatmap size
    xres = 480
    yres = 544
    x = np.arange(xres, dtype=np.float)
    y = np.arange(yres, dtype=np.float)
    xx, yy = np.meshgrid(x, y)
    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    train = []
    heatmap = []
    label = []
    for i in range(1, num_samples+1):
        img = io.imread(os.path.join(train_path, "%d_crop.bmp" % i), )
        img = img[0:2176, :]
        img = zoom(img, zoom=0.25, order=1)
        train.append(img)

    for j in range(1, num_samples+1):
        f = open(os.path.join(label_path1, "%d.txt" % j))
        fp = open(os.path.join(label_path2, "%d.txt" % j))
        line = f.readlines()
        linep = fp.readlines()
        #extract coordinate of anterior nasal spine
        py = (int(line[17].split(',')[1]) + int(linep[17].split(',')[1]) - 200) // 8
        px = (int(line[17].split(',')[0]) + int(linep[17].split(',')[0])) // 8
        coordinate = np.array([px, py])
        imga = test(xres, yres, px, py, xxyy.copy())
        heatmap.append(imga)
        label.append(coordinate)
    X_train = np.array(train)
    Y_train = np.array(heatmap)
    z_train = np.array(label)

    #train test split
    X_train = np.expand_dims(X_train, axis=3)
    Y_train = np.expand_dims(Y_train, axis=3)
    x_training, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=test_ratio, random_state=666)
    x_training, x_test, z_train, z_test = train_test_split(X_train, z_train, test_size=test_ratio, random_state=666)
    x_train, x_validate, y_train, y_validate = train_test_split(x_training, y_train, test_size=validate_ratio, random_state=666)
    x_train, x_validate, z_train, z_validate = train_test_split(x_training, z_train, test_size=validate_ratio, random_state=666)
    return x_train, x_validate, x_test, y_train, y_validate, y_test, z_train, z_validate, z_test



















