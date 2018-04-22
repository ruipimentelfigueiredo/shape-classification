#include "object_detection.h"

void ObjectDetection::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
	CV_Assert(classId < (int)classes.size());
	label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
	      Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}


void ObjectDetection::postprocess(Mat& frame, const Mat& out, Net& net)
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	float* data = (float*)out.data;
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
	    // Network produces output blob with a shape 1x1xNx7 where N is a number of
	    // detections and an every detection is a vector of values
	    // [batchId, classId, confidence, left, top, right, bottom]
	    for (size_t i = 0; i < out.total(); i += 7)
	    {
		float confidence = data[i + 2];
		if (confidence > confThreshold)
		{
		    int left = (int)data[i + 3];
		    int top = (int)data[i + 4];
		    int right = (int)data[i + 5];
		    int bottom = (int)data[i + 6];
		    int classId = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
		    drawPred(classId, confidence, left, top, right, bottom, frame);
		}
	    }
	}
	else if (outLayerType == "DetectionOutput")
	{
	    // Network produces output blob with a shape 1x1xNx7 where N is a number of
	    // detections and an every detection is a vector of values
	    // [batchId, classId, confidence, left, top, right, bottom]
	    for (size_t i = 0; i < out.total(); i += 7)
	    {
		float confidence = data[i + 2];
		if (confidence > confThreshold)
		{
		    int left = (int)(data[i + 3] * frame.cols);
		    int top = (int)(data[i + 4] * frame.rows);
		    int right = (int)(data[i + 5] * frame.cols);
		    int bottom = (int)(data[i + 6] * frame.rows);
		    int classId = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
		    drawPred(classId, confidence, left, top, right, bottom, frame);
		}
	    }
	}
	else if (outLayerType == "Region")
	{
	    // Network produces output blob with a shape NxC where N is a number of
	    // detected objects and C is a number of classes + 4 where the first 4
	    // numbers are [center_x, center_y, width, height]
	    for (int i = 0; i < out.rows; ++i, data += out.cols)
	    {
		Mat confidences = out.row(i).colRange(5, out.cols);
		Point classIdPoint;
		double confidence;
		minMaxLoc(confidences, 0, &confidence, 0, &classIdPoint);
		if (confidence > confThreshold)
		{
		    int classId = classIdPoint.x;
		    int centerX = (int)(data[0] * frame.cols);
		    int centerY = (int)(data[1] * frame.rows);
		    int width = (int)(data[2] * frame.cols);
		    int height = (int)(data[3] * frame.rows);
		    int left = centerX - width / 2;
		    int top = centerY - height / 2;
		    drawPred(classId, (float)confidence, left, top, left + width, top + height, frame);
		}
	    }
	}
	else
	    CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
}


ObjectDetection::ObjectDetection(const std::string & model, const std::string & config, const int & backend, const int & target, const bool & swapRB_, const int inpWidth_, const int inpHeight_) :

	swapRB(swapRB_),
	inpWidth(inpWidth_),
	inpHeight(inpHeight_)
{
	std::cout << model << std::endl;
	net=dnn::readNetFromCaffe(model, config);
	// Load a model.
	net.setPreferableBackend(backend);
	net.setPreferableTarget(target);

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	int initialConf = (int)(confThreshold * 100);
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, this);
}

cv::Mat ObjectDetection::detect(cv::Mat & frame, const Scalar & mean, const double & scale)
{
	// Create a 4D blob from a frame.
	Size inpSize(inpWidth > 0 ? inpWidth : frame.cols, inpHeight > 0 ? inpHeight : frame.rows);
	std::cout << inpSize << std::endl;
	std::cout << mean << std::endl;
	cv::Mat blob=blobFromImage(frame);//, scale, inpSize, mean, swapRB);
	//blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
	//                      const Scalar& mean = Scalar(), bool swapRB=true, bool crop=true);
	// Run a model.

	net.setInput(blob);

	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
	    resize(frame, frame, inpSize);
	    Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
	    net.setInput(imInfo, "im_info");
	}

	Mat out = net.forward();

	postprocess(frame, out, net);

	// Put efficiency information.
	std::vector<double> layersTimes;
	//double freq = getTickFrequency() / 1000;
	//double t = net.getPerfProfile(layersTimes) / freq;
	//std::string label = format("Inference time: %.2f ms", t);
	//putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	static const std::string kWinName = "Deep learning object detection in OpenCV";

	imshow(kWinName, frame);
	//return out;
	return out;
}


/*int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    confThreshold = parser.get<float>("thr");
    float scale = parser.get<float>("scale");
    Scalar mean = Scalar(0,0,0); //parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    CV_Assert(parser.has("model"));

    ObjectDetection object_detector(parser.get<std::string>("model"), parser.get<std::string>("config"),parser.get<int>("backend"), parser.get<int>("target"),  swapRB, inpWidth, inpHeight);


    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);

    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

	cv::Mat detections=object_detector.detect(frame, mean, scale);


    }
    return 0;
}*/




