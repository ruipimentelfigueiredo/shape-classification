#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary file of model contains trained weights. "
                      "It could be a file with extensions .caffemodel (Caffe), "
                      ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet) }"
    "{ config c    | | Path to a text file of model contains network configuration. "
                      "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet) }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       | -1 | Preprocess input image by resizing to a specific width. }"
    "{ height      | -1 | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ backend     |  0 | Choose one of computation backends: "
                         "0: default C++ backend, "
                         "1: Halide language (http://halide-lang.org/), "
                         "2: Intel's Deep Learning Inference Engine (https://software.seek.intel.com/deep-learning-deployment)}"
    "{ target      |  0 | Choose one of target computation devices: "
                         "0: CPU target (by default),"
                         "1: OpenCL }";

using namespace cv;
using namespace dnn;

float confThreshold;
std::vector<std::string> classes;


void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}


class Detection {
	public:
		Detection(const double & confidence_, const int & class_id_, const std::string & class_name_, const cv::Rect & bounding_box_) : 
			confidence(confidence_), class_id(class_id_), class_name(class_name_), bounding_box(bounding_box_) 
		{};
		double confidence;
		int class_id;
		std::string class_name;
		cv::Rect bounding_box;
};

class ObjectDetection {
    Net net;
    bool swapRB;
    int inpWidth;
    int inpHeight;


    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    void postprocess(Mat& frame, const Mat& out, Net& net);

public:
    ObjectDetection(const std::string & model, const std::string & config, const int & backend, const int & target, const bool & swapRB_, const int inpWidth_, const int inpHeight_);
    ~ObjectDetection() {};
    cv::Mat detect(cv::Mat & frame, const Scalar & mean, const double & scale);
};
