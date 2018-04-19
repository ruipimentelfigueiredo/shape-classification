# shape classification using caffe

## Training 
1. Go inside ```$SHAPE_DETECTION_DIR``` folder

2. Download images dataset (train/test), and pre-trained caffe models: bash download.sh (change the folder paths inside)

3. Pre-process dataset:

  - 3.1 Separate images into train and test, augment the train dataset (vertical flipping): 
  
```
python python/modify_dataset.py dataset/
```
  
  - 3.2 run create_lmdb.py to transform images to lmdb files (PUT FOLDER HERE AS A VAR)
  ```
  python python/create_lmdb.py dataset dataset/lmdb
  ```

  - 3.3 compute mean.binaryproto: 
  ```
  $CAFFE_DIR/build/tools/compute_image_mean -backend=lmdb $DATASET_DIR/lmdb/train_lmdb $DATASET_DIR/mean.binaryproto
  ```
4. Train: 
  - 4.1 Open train_val.prototxt. 
  
  Locate and change lines:
  from
  
  ```source: "examples/imagenet/ilsvrc12_train_lmdb"``` 
  
  to
  
  ```source: "$SHAPE_DETECTION_DIR/dataset/lmdb/train_lmdb"```

  from
  
  ```source: "examples/imagenet/ilsvrc12_val_lmdb"``` 
  
  to
  
  ```source: "$SHAPE_DETECTION_DIR/dataset/lmdb/validation_lmdb"```
  
  
  - 4.2 Train:
```
$CAFFE_DIR/build/tools/caffe train --solver=$SHAPE_DETECTION_DIR/base_networks/squeezenet/solver.prototxt --weights $SHAPE_DETECTION_DIR/base_networks/squeezenet/squeezenet_v1.1.caffemodel 2>&1 | tee $DATASET_DIR/model_1_train.log
```
## Testing
the .log file contains train and test error 


