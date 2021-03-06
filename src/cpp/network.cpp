#include "network_classes.hpp"

/************************************************************************/
// Function Network
// Load network, mean file and labels
/************************************************************************/
Network::Network(const string& model_file,
                 const string& weight_file,
                 const string& mean_file)
{

    // Load Network and set phase (TRAIN / TEST)
    net.reset(new Net<float>(model_file, TEST));

    // Load pre-trained net
    net->CopyTrainedLayersFrom(weight_file);

    // Set input layer and check number of channels
    Blob<float>* input_layer = net->input_blobs()[0];
    num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
            << "Input layer should have 1 or 3 channels";

    input_geometry = cv::Size(input_layer->width(), input_layer->height());

    // Load mean file
    SetMean(mean_file);

    //Blob<float>* output_layer = net->output_blobs()[0];
    /*CHECK_EQ(labels.size(), output_layer->channels())
                << "Number of labels is different from the output layer dimension.";*/
}

/************************************************************************/
// Function SetMean
// Create a mean image
/************************************************************************/
void Network::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    // Convert from BlobProto to Blob<float>
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);			  // make copy
    CHECK_EQ(mean_blob.channels(), num_channels)
            << "Number of channels of mean file doesn't match input layer";


    // The format of the mean file is planar 32-bit float BGR or grayscale
    std::vector<cv::Mat> channels;
   // Mat* channels2;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels; ++i) {
        // Extract an individual channel
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);

        channels.push_back(channel);
        //channels2->push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    // Merge the separate channels into a single image
    cv::Mat mean;
    cv::merge(channels, mean);


    // Compute the global mean pixel value and create a mean image filled with this value
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry, mean.type(), channel_mean);
}

/************************************************************************/
// Function Classify
// Return the top N predictions
/************************************************************************/
std::vector<ClassificationData> Network::Classify(const cv::Mat& img) {
    std::vector<float> output = Predict(img);  // output is a float vector
   
    std::vector<ClassificationData> classification_data;
    for(unsigned int i=0; i<output.size(); ++i)
    {

        classification_data.push_back(ClassificationData(i,output[i]));
	//std::cout << classification_data << std::endl;
    }

    return classification_data;
}

/************************************************************************/
// Function Classify
// Return the top N predictions
/************************************************************************/
int Network::ClassifyBest(const cv::Mat& img) {
    std::vector<float> output = Predict(img);  // output is a float vector
    for(unsigned int i=0; i<output.size();++i)
    {
	std::cout << output[i] << std::endl;
    }
    std::cout << std::endl;
    //ClassData mydata(N); // objecto
    std::vector<int> maxN = Argmax(output, 1); // tem o top
    int idx = maxN[0];
    return idx; 
}

/************************************************************************/
// Function Predict
// wrap input layers and make preprocessing
/************************************************************************/
std::vector<float> Network::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net->input_blobs()[0];

    input_layer->Reshape(1, num_channels, input_geometry.height, input_geometry.width);

    // Forward dimension change to all layers
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    // Convert the input image to the input image format of the network
    Preprocess(img, &input_channels);

    net->Forward();

    // Copy the output layer to a std::vector
    Blob<float>* output_layer = net->output_blobs()[0];
    //boost::shared_ptr<caffe::Blob<float> > output_layer = net->blob_by_name("fc8");
    const float* begin = output_layer->cpu_data();      // output of forward pass
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}


/************************************************************************/
// Function WrapInputLayer
// Wrap the input layer of the network in separate cv::Mat objects (one per channel)
// The last preprocessing operation will write the separate channels directly to the input layer.
/************************************************************************/
void Network::WrapInputLayer(std::vector<cv::Mat>* input_channels){
    Blob<float>* input_layer = net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}


/************************************************************************/
// Function Preprocess
// Subtract mean, swap channels, resize input
/************************************************************************/
void Network::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels){

	cv::Mat img_=img.clone(); 
    // Convert the input image to the input image format of the network
    // swap channels from RGB to BGR
    cv::Mat sample;
    //std::cout << "num_channels:"<< num_channels <<  "  img.channels():" << img.channels() << std::endl;
    if (img_.channels() == 3 && num_channels == 1)
        cv::cvtColor(img_, sample, cv::COLOR_BGR2GRAY);
    else if (img_.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img_.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img_.channels() == 1 && num_channels == 3)
        cv::cvtColor(img_, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img_;

   // EQUALIZE
   {
	std::vector<cv::Mat> channels;
        cv::split(sample,channels);
        cv::Mat B,G,R;

        cv::equalizeHist( channels[0], B );
        cv::equalizeHist( channels[1], G );
        cv::equalizeHist( channels[2], R );
        std::vector<cv::Mat> combined;

        combined.push_back(R);
        combined.push_back(G);
        combined.push_back(B);
        cv::merge(combined,sample);


   }


    // Resize if geometry of image != input geometry of the network
    cv::Mat sample_resized;
    if (sample.size() != input_geometry)
        cv::resize(sample, sample_resized, input_geometry, cv::INTER_CUBIC);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels == 3)		// RGB
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    // Subtract the dataset-mean value in each channel
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
       input layer of the network because it is wrapped by the cv::Mat
       objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

/************************************************************************/
// Function Limit Values                                                 
// Find min and max value of vector                                      
// Return bottom_data normalized					 
/************************************************************************/
float* Network::Limit_values(float* bottom_data){
    float smallest = bottom_data[0];
    float largest = bottom_data[0];
    for (unsigned int i=1; i<sizeof(bottom_data); i++) {
        if (bottom_data[i] < smallest)
            smallest = bottom_data[i];
        if (bottom_data[i] > largest)
            largest= bottom_data[i];
    }

    std::vector<float> result;
    result.push_back(smallest);
    result.push_back(largest);

    // Normalize
    for (unsigned int i=0; i< sizeof(bottom_data); ++i){
        bottom_data[i] = bottom_data[i]-result[0];
        bottom_data[i] = bottom_data[i]/result[1];
        //cout << bottom_data[i] << "\n" << endl;
    }

    return bottom_data;
}





