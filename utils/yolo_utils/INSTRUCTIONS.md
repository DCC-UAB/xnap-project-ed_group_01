# YOLO Data Configuration

This file contains the necessary instructions for setting up the data configuration for YOLO object detection.

## Directory Structure

To use the YOLO model, please ensure your data directory follows the following structure: 
```
data/
└── images/
|   └── train/
|   └── test/
└── labels/
    └── train/
    └── test/
```


The `images` directory should contain the training and testing images, while the `labels` directory should contain the corresponding label files for each image.

## Configuring `data.yaml`

In order to train and test the YOLO model, you need to edit the `data.yaml` file. Open the `data.yaml` file and update the following information:

1. Set the `data_directory` field to the path of your data directory.
2. Specify the `train` and `test` subdirectories where your training and testing images and labels are located.
3. List all the classes that you want the YOLO model to detect under the `classes` field. Each class should be on a new line.

Here is an example of how the `data.yaml` file should look like:

```yaml
data_directory: /path/to/your/data/directory
train: labels/train/
test: labels/test/
# Classes
nc: 1
names: ['character']

