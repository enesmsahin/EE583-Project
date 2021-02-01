from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from skimage.feature import hog
from skimage.transform import rescale
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

import joblib
from skimage.io import imread
from skimage.transform import resize
import pickle

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1}) images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"

    dataFile = Path(pklname)
    if dataFile.exists():
        print("Data file exists. Loading it from the disk!")
        data = joblib.load(pklname)
    else:
        print("Data file does not exist. Reading it and saving it to the disk!")
        # read all images in PATH, resize and write to DESTINATION_PATH
        for subdir in os.listdir(src):
            if subdir in include:
                print(subdir)
                current_path = os.path.join(src, subdir)
    
                for file in os.listdir(current_path):
                    if file[-3:] in {'jpg', 'png'}:
                        im = imread(os.path.join(current_path, file))
                        im = resize(im, (width, height)) #[:,:,::-1]
                        data['label'].append(subdir)
                        data['filename'].append(file)
                        data['data'].append(im)
    
            joblib.dump(data, pklname)
    return data

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()

width = 120
isHog = True

load_dir = "./svm_trained_models/hog_120/"

if isHog:
    sgd_clf = joblib.load(load_dir + "svm_model.joblib")
    grayify = joblib.load(load_dir + "grayify.joblib")
    hogify = joblib.load(load_dir + "hogify.joblib")
    scalify = joblib.load(load_dir + "scalify.joblib")
else:
    sgd_clf = joblib.load(load_dir + "svm_model_noHog.joblib")
    scalify = joblib.load(load_dir + "scalify_noHog.joblib")
print("SVM Model loaded from the disk!\n")

print("Test Data is being loaded!")
data_path = './Dataset/Test2/'
base_name = './Dataset/Test2/'
include = {'Izmir', 'Metu_blue', 'Metu_red'}
data = resize_all(src=data_path, pklname=base_name, width=width, include=include)

from collections import Counter
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
X_val = np.array(data['data'])
Y_val = np.array(data['label'])
if isHog:
    X_val = grayify.transform(X_val)
    X_val = hogify.transform(X_val)
    X_val = scalify.transform(X_val)
else:
    X_val = scalify.transform(X_val.reshape(len(data['data']), -1)).reshape(X_val.shape)
print("Test Data is loaded!\n")

print("Prediction started!")

if isHog:
    y_pred_sgd = sgd_clf.predict(X_val)
else:
    y_pred_sgd = sgd_clf.predict(X_val.reshape(X_val.shape[0], -1))

print("Prediction finished!\n")

print("Classification report for - \n{}:\n{}\n".format(
    sgd_clf, metrics.classification_report(Y_val, y_pred_sgd)))


print('')
print('Accuracy: {0}%'.format(100*np.sum(y_pred_sgd == Y_val)/len(Y_val)))

for i, y_pred in enumerate(y_pred_sgd):
    if y_pred != Y_val[i]:
        print(data["filename"][i])
        print("Prediction: " + y_pred + "\t GT: " + Y_val[i])
