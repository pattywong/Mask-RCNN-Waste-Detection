# Waste detection with instance segmentation using Mask R-CNN
The implementation of waste detection with instance segmentation for a robotic sorting system using 
[Mask R-CNN](https://arxiv.org/abs/1703.06870) is extended from [Matterport - Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN). This model is based on Feature Pyramid Network and a ResNet101 backbone, that generates bounding boxes, segmentation masks, and object classes for each instance of a waste object in the image. 

[pictures]

The repository includes:
- Source code of Mask R-CNN (Customized)
- Jupyter notebooks for data and model visualization
- Training and inference code
- Waste dataset with annotation files
- Model object
- Model weights

## Requirements
- Python 3.6.1
- OpenCV 4.2.0
- Tensorflow 1.14
- Keras 2.1.6
- Other packages listed in `requirements.txt`

## Cloning an environment from an environment.yml file
```
conda-env create --name mrcnn --file \path\to\mrcnn.yml
```

## Detection
```
conda-env create --name mrcnn --file \path\to\mrcnn.yml
```

## Application of AI in a robotic sorting system


