/*
 *  Copyright (C) 2018 Rui Pimentel de Figueiredo
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *      http://www.apache.org/licenses/LICENSE-2.0
 *      
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*!    
    \author Rui Figueiredo : ruipimentelfigueiredo
*/

#ifndef SHAPECLASSIFIER_H
#define SHAPECLASSIFIER_H

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <limits>
#include <boost/algorithm/string.hpp>

#include "network_classes.hpp"
//#include "laplacian_foveation.hpp"


using namespace caffe;
using namespace std;

using std::string;


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;



class CylinderClassifier
{

	std::string absolute_path_folder;
	std::string model_file;
	std::string weight_file;
	std::string mean_file;
	std::string device;
	unsigned int device_id;
	boost::shared_ptr<Network> network; 
public:
	CylinderClassifier(const std::string & absolute_path_folder_,
			   const std::string & model_file_,
			   const std::string & weight_file_,
			   const std::string & mean_file_,
		           const std::string & device_,
			   const unsigned int & device_id_=0);

	float classify(const cv::Mat& img);
	int classifyBest(const cv::Mat& img);
};


#endif // SHAPECLASSIFIER_H


