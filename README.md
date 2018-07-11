# shape classification using caffe

## Training 
1. Go inside ```$SHAPE_DETECTION_DIR``` folder

2. Download images dataset (train/test), and pre-trained caffe models: ```bash download.sh``` (change the folder paths inside)

3. Pre-process dataset:

  - 3.1 Separate images into train and test, augment the train dataset (vertical flipping): 
  
```
python src/python/modify_dataset.py
```
  
  - 3.2 run create_lmdb.py to transform images to lmdb files
  ```
  python src/python/create_lmdb.py 
  ```

  - 3.3 compute mean.binaryproto: 
  ```
  $CAFFE_DIR/build/tools/compute_image_mean -backend=lmdb $DATASET_DIR/lmdb/train_lmdb $DATASET_DIR/mean.binaryproto
  ```
4. Train: 

  - 4.1 Switch to the root of this repository. 
  
  - 4.2 Train:
  
   replace ```$DATASET_FOLDER``` with absolute path inside ```base_networks/squeezenet/train_val.prototxt``` then:
  
  ```
  $CAFFE_DIR/build/tools/caffe train --solver base_networks/squeezenet/solver.prototxt --weights base_networks/squeezenet/squeezenet_v1.1.caffemodel 2>&1 | tee model.log
  ```
## Testing
the .log file contains train and test errors. The following line, plots the learning curve:

```
python src/python/plot_learning_curve.py --caffe-path $CAFFE_DIR
```
The baseline predictions for comparision are provided in ```inference_data/baseline``` directory. The following line will save the **precision recall** curve in the ```inference_data``` folder.

```
python src/python/precision_recall.py
```

### Reference

In case you use our library in your research, please cite our work

```
@inproceedings{figueiredo2017shape,
  title={Shape-based attention for identification and localization of cylindrical objects},
  author={Figueiredo, Rui and Dehban, Atabak and Bernardino, Alexandre and Santos-Victor, Jos{\'e} and Ara{\'u}jo, Helder},
  booktitle={IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL-EpiRob)},
  volume={18},
  pages={21},
  year={2017}
}

```
[paper]: http://vislab.isr.ist.utl.pt/wp-content/uploads/2017/09/rfigueiredo-icdlepirob2017.pdf
