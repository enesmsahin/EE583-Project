# METU EE583 PATTERN RECOGNITION 2020-2021 FALL TERM PROJECT
## Study and Comparison of EfficientNet and SVM Classifier on An Image Classification Task

For presentation of the project, see: https://www.youtube.com/watch?v=zMB8mXVaVFs

Dataset is available on: https://drive.google.com/file/d/1groGlqFeX_YcK9Ai0j7NQ-M2ZcYsO0H9/view?usp=sharing

All codes are tested on Windows 10, CUDA 10.2, PYTORCH 1.7.1.

See ```requirements.txt``` for requirements.

### Explanation of .py files ### 
```train.py```: Training script for EfficientNet-B0
```test.py```: Test script for EfficientNet-B0
```train_svm.py```: Training script for SVM
```test_svm.py```: Testing script for SVM
```visualize_hog.py```: Script for visualizing HOG features
```data_augmentation.py```: Script for data augmentation

Set ```isHog``` variable to false if you would like to train SVM with raw pixel values.
In order to execute scripts, download dataset and extract it to the same directory as scripts:  
--Dataset  
|----orig_train_dev  
|----orig_val  
|----Test2  
|----Train-dev  
|----Validation  
--train.py  
--test.py  
--train_svm.py  
--test_svm.py  
...

Reference for SVM-HOG implementation: https://kapernikov.com/tutorial-image-classification-with-scikit-learn/