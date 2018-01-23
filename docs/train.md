
# Procedure for installing TensorFlow object detection API

```sh
$ # First install TensorFlow and its dependencies
$ sudo pip install pillow
$ sudo pip install lxml
$ sudo pip install jupyter
$ sudo pip install matplotlib
$ # For CPU
$ sudo pip install tensorflow
$ # Install object detection API
$ mkdir tensorflow
$ cd tensorflow/
$ git clone https://github.com/madi/models.git
$ cd models/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:$HOME/tensorflow/models/research:$HOME/tensorflow/models/research/slim
$ sudo python setup.py install
$ # Test the installation
$ python object_detection/builders/model_builder_test.py
```


# Procedure for training a new neural network with TensorFlow

### STEP 1: Labelling the images using labelImg

Label the target classes in the images using 
[labelImg](https://github.com/tzutalin/labelImg).

```sh
$ sudo pip install labelImg
```

### STEP 2: Split images in 2 folders, train (90%) and test (10%)
$HOME/TensorFlow_utils/trees_recognition/images/train
$HOME/TensorFlow_utils/trees_recognition/images/test

### STEP 3: Convert xml to csv using the utility xml_to_csv.py
Original source: [xml_to_csv.py](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py).
Run the modified version in:

```sh
$ cd $HOME/TensorFlow_utils/trees_recognition/
$ python xml_to_csv.py
```

### STEP 4: Create TFRecord using the utility generate_tfrecord.py
Original source:: [generate_tfrecord.py](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)
Run the modified version in:

```sh
$ cd $HOME/TensorFlow_utils/trees_recognition/
$ # Create train data:
$ python generate_tfrecord.py \
  --csv_input=$HOME/TensorFlow_utils/trees_recognition/data/train_labels.csv  \
  --output_path=$HOME/TensorFlow_utils/trees_recognition/data/train.record \
  --images_path=$HOME/TensorFlow_utils/trees_recognition/images/train
$ 
$  # Create test data:
$ python generate_tfrecord.py --csv_input=$HOME/TensorFlow_utils/trees_recognition/data/test_labels.csv  \
  --output_path=$HOME/TensorFlow_utils/trees_recognition/data/test.record \
  --images_path=$HOME/TensorFlow_utils/trees_recognition/images/test
```

### STEP 5: Create configuration file and choose the model

If you want, you can create your own model and you will need to create 
your config file following the instructions in [Configuring the Object 
Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).
Alternatively, if you want to use a pre-existing model, you can pick an 
existing config file from [samples configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
For a comparison among available models, see [Tensorflow detection model 
zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

We choose 
[faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz).

In the config file faster_rcnn_resnet101_coco.config, we need to change:

* Number of classes
* batch_size: if you get memory error you have to lower this number (not recommended!)
* All paths indicated by "PATH_TO_BE_CONFIGURED"
* Name of the .record file
* label_map_path = $HOME/TensorFlow_utils/trees_recognition/training/trees_detection.pbtxt

Create $HOME/TensorFlow_utils/trees_recognition/training/trees_detection.pbtxt 
indicating the classes:

```sh
item {
  id: 1
  name: 'sick'
}
item {
  id: 2
  name: 'dead'
}
```

### STEP 6: Create training folder

Create the following folder:

$HOME/TensorFlow_utils/trees_recognition/training

and put trees_detection.pbtxt and config file faster_rcnn_resnet101_coco.config
inside it.


### STEP 7: Launch the training

```sh
$ cd $HOME/tensorflow/models/research/object_detection
$ python train.py --logtostderr \
--train_dir=$HOME/TensorFlow_utils/trees_recognition/training \
--pipeline_config_path=$HOME/TensorFlow_utils/trees_recognition/training/faster_rcnn_resnet101_coco.config
```

### STEP 8: Monitoring the training

Launch TensorBoard

```sh
$ cd $HOME/tensorflow/models/research/
$ tensorboard --logdir=$HOME/TensorFlow_utils/trees_recognition/training/

```

This command will create a file event* in the training folder, that is 
used by TensorBoard.
Open TensorBoard in the browser.

### STEP 9: Stop the training

Watch the TotalLoss function in TensorBoard and stop the training when it
converges towards 0.

### STEP 10: Export inference graph and use it for prediction

Use the utility 
[export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) 
in the object_detection folder.

```sh
$ cd $HOME/tensorflow/models/research/object_detection
$ python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $HOME/TensorFlow_utils/trees_recognition/training/faster_rcnn_resnet101_coco.config \
    --trained_checkpoint_prefix $HOME/TensorFlow_utils/trees_recognition/training/model.ckpt-42845 \
    --output_directory $HOME/TensorFlow_utils/trees_recognition/training/tree_detection_graph
```

where you should put the actual last recorded step in place of the number. 

### STEP 11

Run prediction

```sh
$ cd $HOME/TensorFlow_utils/
$ python run_pred_bulk.py \
--imagesPath=$HOME/TensorFlow_utils/trees_recognition/images/pred
```

### STEP 12

If prediction is not satisfactory, we can resume the training. In config 
file, change:

```sh
fine_tune_checkpoint: "$HOME/TensorFlow_utils/trees_recognition/training/model.ckpt-9261"
```
where 9261 is the last checkpoint.

Then launch the training again:

```sh
$ cd $HOME/tensorflow/models/research/object_detection
$ python train.py --logtostderr \
--train_dir=$HOME/TensorFlow_utils/trees_recognition/training \
--pipeline_config_path=$HOME/TensorFlow_utils/trees_recognition/training/faster_rcnn_resnet101_coco.config
```

### STEP 13

Convert boxes into shapefiles using 
[convert_coords_boxes.py](https://github.com/madi/geo_tensorflow/blob/master/convert_coords_boxes.py)

```sh
$ cd $HOME/TensorFlow_utils/
$ python convert_coords_boxes.py
```
