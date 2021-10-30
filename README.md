# Waste Detection with Instance Segmentation using Mask R-CNN
The implementation of waste detection with instance segmentation for a robotic sorting system using 
[Mask R-CNN](https://arxiv.org/abs/1703.06870) is extended from [Matterport - Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN). This model is based on Feature Pyramid Network and a ResNet101 backbone, that generates bounding boxes, segmentation masks, and object classes for each instance of a waste object in the image. 

![result_img](/assets/result_img.png)
![result_img2](/assets/result_img2.png)

The repository includes:
- Source code of Mask R-CNN (Customized)
- Jupyter notebooks for data and model visualization
- Training and inference code
- Sample image with annotation file

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
An open-source annotation software , [Make Sense](https://www.makesense.ai/), from [Piotr - makesense.ai](https://github.com/SkalskiP/make-sense) is used to generate .JSON annotation files for Dataset images. Click to see [Samples of my raw dataset images](https://drive.google.com/drive/folders/1-xJjNR9B8QJW3Mw6G_OLPfguWCJtHTAk?usp=sharing).

![raw_img](/assets/raw_img.jpg)
![labeled_img](/assets/labeled_img.png)

```inspect_waste_data.ipynb``` provides data visualization.

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
![training_loss](/assets/training_loss.png)

## Detection
```inspect_waste_model.ipynb``` provides step by step detection with visualization.
## Application of AI in a robotic sorting system
The model enables a robotic arm to be able to pick a waste object and put it in the right bin by localizing the positions to get their real-world poses of each waste object and classifying a type of waste whether it is a bottle, a snack bag, or a can with its alignments. The model achieved 0.91 mAP on test images of unseen waste objects. [Link](https://drive.google.com/file/d/10tBnl9jbO7x6e_eqKj4Gk4imgKbYP2JA/view?usp=sharing) to see how it works.

![robotics_sorting_system](/assets/robotic_sorting_system.png)
![bins](/assets/bins.jpeg)

## Citation
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```


