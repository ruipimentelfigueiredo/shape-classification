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
#ifndef NETWORKCLASSES_H
#define NETWORKCLASSES_H

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <memory>
#include <math.h>
#include <limits>
#include <sstream>


#define CV_64FC3 CV_MAKETYPE(CV_64F,3)


using namespace caffe;
using namespace std;
using std::string;


/************************************************************************/
// Function PairCompare
// Compare 2 pairs
/************************************************************************/
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}


/************************************************************************/
// Function CalcRGBmax
// Get the highest value of R,G,B for each pixel
/************************************************************************/

/*cv::Mat CalcRGBmax(cv::Mat i_RGB) {

    std::vector<cv::Mat> planes(3);

    cv::split(i_RGB, planes);

    cv::Mat maxRGB = max(planes[2], cv::max(planes[1], planes[0]));

    return maxRGB;
}*/



/************************************************************************/
// Function Argmax
// Return the indices of the top N values of vector v
/************************************************************************/
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);

    return result;
}



/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class ClassData{
public:
    ClassData(int N_): N(N_)    {  // construtor
      label.resize(N);
      score.resize(N);
      index.resize(N);
    }
    int N;
    std::vector<string> label;
    std::vector<float> score;
    std::vector<int> index;

    friend ostream &operator<<( ostream &output,const ClassData &D ) {
        for(int i=0; i<D.N;++i) {
            output << " Index: " << D.index[i] << "\n"
                   << " Label: " << D.label[i] << "\n"
                   << " Confidence: " << D.score[i] << "\n" << endl;
        }
        return output;
    }
};

class ClassificationData 
{

	public:
		unsigned int id;
		float confidence;

	ClassificationData(const unsigned int id_, const float confidence_) : id(id_), confidence(confidence_)
	{};

    friend ostream &operator<<( ostream &output,const ClassificationData &class_data ) {

            output << " index: " << class_data.id << "\n"
                   << " confidence: " << class_data.confidence << "\n" << endl;

        return output;
    }
};

class Network{
public:
    Network(const string& model_file,
            const string& weight_file,
            const string& mean_file); //construtor

    // Return prediction
    std::vector<ClassificationData> Classify(const cv::Mat& img);

    float* Limit_values(float* bottom_data); // NEW
    float find_max(cv::Mat gradient_values);
    int ClassifyBest(const cv::Mat& img);
private:
    void SetMean(const string& mean_file);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    std::vector<float> Predict(const cv::Mat& img);

    int num_channels;
    std::shared_ptr<Net<float> > net;


    cv::Mat mean_;
    std::vector<string> labels;
    cv::Size input_geometry;		// size of network - width and height
};

#endif // NETWORKCLASSES_H
