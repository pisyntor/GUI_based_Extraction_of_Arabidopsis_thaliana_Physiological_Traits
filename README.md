# Computer-based Recognition of Arabidopsis thaliana Ecotypes

This repository provides the full source code developed to classify various A. thaliana ecotypes using an RGB image analysis pipeline. This pipeline operates with a large volume of data collected by an X-Y indoor HTPP system.

## Description

The RGB image analysis pipeline consists of several elements (see Figure 1):
A – Collection of RGB image data in an indoor HTPP environment, B – Preparation and organization of the data using auto-pot and auto-cropping procedures, C – Assessment of complexity (variability) within generated datasets, D – Data re-definition sub-component composed of GUI-based image processing and three levels of transformation, E – Partitioning data into training, validation, and test sets, F – Data augmentation using random affine transformations, G – Auto-optimization sub-system that performs single-image and sequence-of-images classification, H – Storage of the best-performing deep learning models, I –Inferencing on test data as well as J – externally sourced data,  K – Visualization of critical regions using heat and saliency maps.

![image](images/ex_GUI_flow.png)

**Figure 1**. The complete workflow of the developed RGB image analysis pipeline for the recognition of various Arabidopsis ecotypes.

## Requirements
- [python>=3.7](https://www.python.org/downloads/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [pillow](https://pypi.org/project/pillow/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [kyvi](https://pypi.org/project/Kivy/)

## Structure
The fundamental filesystem structure resembles the tree shown below. Essentially, we have two main folders: ```code```.
```
code
├───analyzer.py
│
├───displaybox.py
│
├───inputbox.py
│
├───listbox.py
│
├───main.py
│
├───manager.py
│
└───settingbox.py
```

## LICENSE
This repo is distributed under [LICENSE](LICENSE).
