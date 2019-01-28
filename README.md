# shape classification using caffe

## Training 
1. Clone and go to the root folder of this repository.

2. Download images dataset (train/test), and pre-trained caffe models: ```bash scripts/download_dataset.sh``` and ```bash scripts/download_models.sh```

3. Pre-process dataset:

  - 3.1 Separate images into train and validation, and augment the train dataset (vertical flipping): 
  
  ```
  python src/python/modify_dataset.py --data-path dataset/train/images_clusters --output output -ft train -st validation -sp 0.1 --augment  True -m train 
  ```
  - 3.2 run create_lmdb.py to transform images to lmdb files

  ```
  python src/python/create_lmdb.py --data-path output --lmdb-path output/lmdb 
  ```

  - 3.3 compute mean.binaryproto:

  ```
  $CAFFE_DIR/build/tools/compute_image_mean -backend=lmdb output/lmdb/train_lmdb output/mean.binaryproto
  ```
  
4. Train: 

  - 4.1. Switch to the root of this repository. 
  
  - 4.2. Train:
  
   replace ```$SHAPE_CLASSIFICATION_FOLDER``` with repository absolute path inside ```base_networks/squeezenet/train_val.prototxt``` then, train the model:
  
  ```
    $CAFFE_DIR/build/tools/caffe train --solver base_networks/squeezenet/solver.prototxt --weights base_networks/squeezenet/squeezenet_v1.1.caffemodel 2>&1 | tee output/model.log
  ```

  - 4.3. The .log file contains train and test errors. The following line, plots the learning curve:

  ```
  python src/python/plot_learning_curve.py --caffe-path $CAFFE_DIR --log-path output/
  ```

## Testing

  - 1. Copy test images inside output folder
  ```
  python src/python/modify_dataset.py --data-path dataset/test/images_clusters --output output -ft test -m test
  ```

  The baseline predictions for comparision are provided in ```dataset/test/pointclouds_clusters/{OBJ_TYPE}/{SEQUENCE}/results``` directories. The following line will save the **precision recall** curve in the ```inference_data``` folder.

  - 2. Plot precision recall curves
  ```
  python src/python/precision_recall.py --fitting-cylinders-path dataset/test/pointclouds_clusters/cylinder/ --fitting-others-paths dataset/test/pointclouds_clusters/sphere/,dataset/test/pointclouds_clusters/box/ --weights-path train_iter_201.caffemodel --data-path output/test/ --mean-binaryproto output/mean.binaryproto --output-file output/p-r.pdf
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
