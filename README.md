# Waste detection with instance segmentation using Mask R-CNN
The implementation of waste detection with instance segmentation for a robotic sorting system using 
[Mask R-CNN](https://arxiv.org/abs/1703.06870) is extended from [Matterport - Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN). This model is based on Feature Pyramid Network and a ResNet101 backbone, that generates bounding boxes, segmentation masks, and object classes for each instance of a waste object in the image. 

![result_img](/assets/result_img.png)


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
conda-env create -n mrcnn -f mrcnn.yml
```
## Dataset and Annotations
An open-source annotation software , [Make Sense](https://www.makesense.ai/), from [Piotr - makesense.ai](https://github.com/SkalskiP/make-sense) is used to generate .JSON annotation files for Dataset images.
![raw_img](/assets/raw_img.jpg)
![labeled_img](/assets/labeled_img.png)

## Training model 
In `waste_main.py`, the code provides options of training on the network where parameters are modified in `/mrcnn/waste_config.py`.
```
# Train a new model starting from pre-trained COCO weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last
```

## Detection
```
conda-env create --name mrcnn --file \path\to\mrcnn.yml
```

## Application of AI in a robotic sorting system
![Capture_Bin](/assets/Capture_Bin.png)

