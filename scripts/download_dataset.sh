#!/bin/bash

##### Functions
function get_dataset
{
	mkdir dataset/
        cd dataset/
	#curl -L -o dataset.zip 'https://drive.google.com/uc?export=download&id=1cRD5PpQ5oDLIQ-4iFaBT3JRERtPiTn8m'
	wget http://soma.isr.tecnico.ulisboa.pt/vislab_data/facyl/facyl.zip
	unzip facyl.zip
	rm -f facyl.zip
}

# Get network models
get_dataset

