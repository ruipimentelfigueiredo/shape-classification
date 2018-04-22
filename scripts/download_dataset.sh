#!/bin/bash

##### Functions

function get_dataset
{
	mkdir dataset/
        cd dataset/
	curl -L -o dataset.zip 'https://drive.google.com/uc?export=download&id=1cRD5PpQ5oDLIQ-4iFaBT3JRERtPiTn8m'
        cd ..
	unzip dataset/dataset.zip
	rm -f dataset/dataset.zip
}



# Get network models
get_dataset

