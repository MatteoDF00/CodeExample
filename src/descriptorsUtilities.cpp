#include "descriptorsUtilities.h"

/**     
 * SIFT descriptor and plot.
 * This function is used to compute sift features and the relative
 * descriptors. Those keypoints are then plotted in a new image.
 *
 * @param image: input image.
 * @param keypoints: vector of features extracted by SIFT
 * @return win_name: name of the window with keypoints.
 */
cv::Mat SIFTkpPlot(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, const std::string winName) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Mat descriptor;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptor);
    //-- Draw keypoints
    cv::Mat img_kp;
    cv::drawKeypoints(image, keypoints, img_kp);
    cv::namedWindow(winName); cv::imshow(winName, img_kp);
    return descriptor;      //return the descriptor, not the image
}
/**     
 * SIFT descriptor 
 * This function is used only to compute sift features and the relative
 * descriptors.
 *
 * @param image: input image.
 * @param keypoints: vector of features extracted by SIFT
 */
cv::Mat SIFTdescriptor(cv::Mat image, std::vector<cv::KeyPoint>& keypoints) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Mat descriptor;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptor);
    return descriptor;      //return the descriptor, not the image
}

/**     
 * UTILITY function 
 * This function is used to return the standar notation for image type.
 *
 * @param type: type in int returned by mat.size()
 */
std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}
/**     
 * UTILITY function
 * This function is used to pre process the image. It increase the size of it
 * with CUBIC interpolation or reduce its size. 
 * The function preserves the shape of the image so the final size is relative to
 * the smallest size of the image.
 *
 * @param img: input image.
 * @param final_size: the desired size of the image.
 */
cv::Mat img_preprocessing(cv::Mat img, const int final_size) {
    cv::Mat dst;

    int h = img.rows, w = img.cols;

    float scale = (float)final_size / std::min(h, w);
    if (scale > 1)
        cv::resize(img, dst, cv::Size(0, 0), scale, scale, cv::INTER_CUBIC);
    else
        cv::resize(img, dst, cv::Size(), scale, scale, cv::INTER_AREA);
    return dst;
}

/**     
 * LBP 
 * This function is used to compute the extended LOCAL BINARY PATTERN image
 * 
 *
 * @param src_: input image.
 * @param radius: the larger the radius, the smoother the LBP image
 * @param neighbors: the higher number of sampling points, the more patterns it is possible to encode.
 */
cv::Mat elbp(cv::Mat src_, int radius, int neighbors) {
    cv::Mat src, dst;
    cv::cvtColor(src_, src, cv::COLOR_BGR2GRAY);
    
    dst = cv::Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
    
    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius) * sin(2.0 * CV_PI * n / static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * cos(2.0 * CV_PI * n / static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;

        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {

                float t = w1 * src.at<char>(i + fy, j + fx) + w2 * src.at<char>(i + fy, j + cx) + w3 * src.at<char>(i + cy, j + fx) + w4 * src.at<char>(i + cy, j + cx);
                
                dst.at<int>(i - radius, j - radius) += ((t > src.at<char>(i, j)) || (std::abs(t - src.at<char>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

/**     
 * LBP descriptor
 * This function is used to compute the extended LOCAL BINARY PATTERN descriptor
 * from the ELBP image.
 *
 * @param src_: input image.
 * @param radius: the larger the radius, the smoother the LBP image
 * @param neighbors: the higher number of sampling points, the more patterns it is possible to encode.
 */
cv::Mat lbp_descriptor(const cv::Mat img, const int radius, const int neigh, int gray_range, bool single_hist) {
    //std::cout << "MSER dio ca" << std::endl;
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    ms->setMinArea(30);
    ms->setMaxArea(2500);
    std::vector< std::vector< cv::Point> > regions;
    std::vector<cv::Rect> mser_bbox;
    ms->detectRegions(img, regions, mser_bbox);

    //std::cout << "detected regions porco di quel di" << std::endl;
    /*cv::Mat tmp = img.clone();
    for (int i = 0; i < regions.size(); i++)
        if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
            rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
    cv::imshow("mser", tmp);*/

    //std::cout << "plotted regions... madonnola" << std::endl;

    float range_gray[] = { 0, gray_range };

    const float* hist_range = { range_gray };

    bool uniform = true, accumulate = true;
    if (!single_hist)
        accumulate = false;

    cv::Mat hist, multi_hist;

    //std::cout << "calcolor histogrANO" << std::endl;
    //cv::Mat MSER_regions;
    bool flag_first_region = true;
    for (int i = 0; i < regions.size(); i++) {
        if (mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10)
            continue;
        cv::Mat sub_img = (img.clone())(mser_bbox[i]);
        //std::cout << "subimg diocantante" << std::endl;
        cv::Mat sub_elbp = elbp(sub_img, radius, neigh);
        //std::cout << "subimg(" << i << ") size: " << sub_img.size() << std::endl;
        //MSER_regions.push_back(sub_elbp);
        //cv::calcHist(&sub_elbp, 1, 0, cv::Mat(), hist, 1, 255, &hist_range, true, true);
        //std::cout << " zio maiale sto histogramma del porco d \n";
        if (single_hist)
            cv::calcHist(&sub_elbp, 1, 0, cv::Mat(), hist, 1, &gray_range, &hist_range, uniform, accumulate);
        else {
            cv::calcHist(&sub_elbp, 1, 0, cv::Mat(), hist, 1, &gray_range, &hist_range, uniform, accumulate);
            hist = hist.reshape(1, 1);
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
            //std::cout << " lbp hist size: " << hist.size() << std::endl;
            if (flag_first_region) {
                multi_hist = hist;
                flag_first_region = false;
            }
            else
                try {
                    cv::vconcat(multi_hist, hist, multi_hist);
                }
                catch (const std::exception&){
                    std::cout << " lbp hist size: " << hist.size() << "\t";
                    std::cout << " lbp MULTI_hist size: " << multi_hist.size() << std::endl;
                    cv::Mat tmp = img.clone();
                    for (int i = 0; i < 1/*regions.size()*/; i++)
                        if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
                            rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
                    cv::imshow("mser", tmp);
                    cv::waitKey(0);
                }
        }
    }
    if (single_hist) {
        hist = hist.reshape(1, 1);

        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

        return hist;
    }
    else
        return multi_hist;
}
/**     
 * color descriptor
 * This function is used to compute the color histogram in the HSV color space.
 *
 * @param image: input image.
 */
cv::Mat color_histogram(const cv::Mat& image) { //const because we don't want to modify the image
    //representation conversion to HSV
    cv::Mat hsv_img;
    cv::cvtColor(image, hsv_img, cv::COLOR_BGR2HSV);

    //split the three channels
    std::vector<cv::Mat> channels;
    cv::split(hsv_img, channels);

    //The Hue channel has a smaller range

    int histSize_hue = 100;         //typically 180 //# bins for Hue channel
    int histSize_saturation = 100;  //typically 256 //# bins for saturation channel

    float range_hue[] = { 0, histSize_hue };
    float range_saturation[] = { 0, histSize_saturation };

    const float* hist_rangeH = { range_hue };
    const float* hist_rangeS = { range_saturation };

    bool uniform = true, accumulate = false;

    //compute the histogram only for the hue and saturation channel (we are not considering value to be robust to illumination changes)
    //each histogram is a one dimensional vector representing each bins
    cv::Mat hist_hue, hist_saturation;
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), hist_hue, 1, &histSize_hue, &hist_rangeH, uniform, accumulate);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), hist_saturation, 1, &histSize_saturation, &hist_rangeS, uniform, accumulate);

    //we need row vectors
    hist_hue = hist_hue.reshape(1, 1);
    hist_saturation = hist_saturation.reshape(1, 1);

    //concatenate the channel histograms horizontally to form the color descriptor (one row vector)
    cv::Mat color_descriptor;
    color_descriptor = hist_saturation;
    cv::hconcat(hist_hue, hist_saturation, color_descriptor); //descriptor is one row vector

    //normalization of the descriptor
    //NORM_MINMAX such that the min value is 0 and the max value is 1
    cv::normalize(color_descriptor, color_descriptor, 0, 1, cv::NORM_MINMAX);
    
    return color_descriptor;
}
/**
 * color descriptor
 * This function is an overload of the previous one. It computes the histogram in RGB color space
 * and return the concatenation of all the three histograms into a single row vector.
 *
 * @param iamge: input image.
 * @param rgb_case: flag to select this as "color_histogram" method to call
 * @param rgb_histSize: dimension of the bins for each of the three channels
 */
cv::Mat color_histogram(const cv::Mat& image, bool rgb_case, int rgb_histSize) { //const because we don't want to modify the image
    //FULL BGR
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Set the histogram parameters
    int histSize = rgb_histSize; //256;  // Number of bins for each channel
    float range[] = { 0, histSize };  // Pixel value range (0-255)
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    // Calculate the histogram for each color channel
    cv::Mat bHist, gHist, rHist;
    cv::calcHist(&channels[0], 1, nullptr, cv::Mat(), bHist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&channels[1], 1, nullptr, cv::Mat(), gHist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&channels[2], 1, nullptr, cv::Mat(), rHist, 1, &histSize, &histRange, uniform, accumulate);
    bHist = bHist.reshape(1, 1);
    gHist = gHist.reshape(1, 1);
    rHist = rHist.reshape(1, 1);
    cv::Mat color_descriptor;
    cv::hconcat(bHist, gHist, color_descriptor); //descriptor is one row vector
    cv::hconcat(color_descriptor, rHist, color_descriptor); //descriptor is one row vector

    //normalization of the descriptor (also important for the training process see Alessandro Closed detto il Bello)
    //NORM_MINMAX such that the min value is 0 and the max value is 1
    cv::normalize(color_descriptor, color_descriptor, 0, 1, cv::NORM_MINMAX);

    return color_descriptor;
}

/**     
 * MSER feature extractor 
 * This function is used to extract the MSER feature and 
 * compute the LBP descriptor and color histogram descriptor in all the regions.
 *
 * @param img: input image.
 * @param color_descr: output parameter that contains the color descriptor of the regions.
 * @param lbp_descr: output parameter that contains the LBP descriptor of the regions.
 * @param col_on: flag to decide the inclusion of the color histogram.
 * @param lbp_on: flag to decide the inclusion of the lbp descriptor.
 * @param hist_color_size: dimension of the x axis of the color histogram.
 * @param hist_lbp_size: dimension of the x axis of the lbp histogram.
 * @param radius: radius of the elbp descriptor.
 * @param neigh: neighbor of the elbp descriptor.
 */
void MSER_descriptor(const cv::Mat img, cv::Mat& color_descr, cv::Mat& lbp_descr, bool col_on, bool lbp_on, int hist_color_size, int hist_lbp_size, const int radius, const int neigh) {
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    ms->setMinArea(30);
    ms->setMaxArea(2500);
    std::vector< std::vector< cv::Point> > regions;
    std::vector<cv::Rect> mser_bbox;
    ms->detectRegions(img, regions, mser_bbox);

    /* if needed this section plots the MSER regions */ 
    /*cv::Mat tmp = img.clone();
    for (int i = 0; i < regions.size(); i++)
        if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
            rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
    cv::imshow("mser", tmp);*/

    /* single channel histogram variables */
    float range_gray[] = { 0, hist_lbp_size };
    const float* hist_range = { range_gray };
    bool uniform = true, accumulate = false;

    cv::Mat hist_lbp, multi_hist_color, multi_hist_lbp;

    bool flag_first_region = true;

    for (int i = 0; i < regions.size(); i++) {
        if (mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10)
            continue;

        cv::Mat sub_img = (img.clone())(mser_bbox[i]);          //sub_region creation

        cv::Mat sub_elbp = elbp(sub_img, radius, neigh);        //lbp descriptor in sub_image
        cv::calcHist(&sub_elbp, 1, 0, cv::Mat(), hist_lbp, 1, &hist_lbp_size, &hist_range, uniform, accumulate);
        hist_lbp = hist_lbp.reshape(1, 1);
        cv::normalize(hist_lbp, hist_lbp, 0, 1, cv::NORM_MINMAX);

        /* HUE-saturation histogram */
        cv::Mat hist_color = color_histogram(sub_img);

        //std::cout << "--   lbp hist size: " << multi_hist_lbp.size() << std::endl;
        //std::cout << "--   color hist size: " << multi_hist_color.size() << std::endl;

        /* concatenation of all the regions's descriptors */
        if (flag_first_region) {
            flag_first_region = false;
            multi_hist_lbp = hist_lbp.clone();
            multi_hist_color = hist_color.clone();
        }
        else
            try {
                cv::vconcat(multi_hist_lbp, hist_lbp, multi_hist_lbp);
                cv::vconcat(multi_hist_color, hist_color, multi_hist_color);
            }
            catch (const std::exception&) {
                std::cout << " lbp hist size: " << hist_lbp.size() << "\t";
                std::cout << " lbp MULTI_hist size: " << multi_hist_lbp.size() << std::endl;
                std::cout << " lbp hist size: " << hist_color.size() << "\t";
                std::cout << " lbp MULTI_hist size: " << multi_hist_color.size() << std::endl;
                cv::Mat tmp = img.clone();
                for (int i = 0; i < regions.size(); i++)
                    if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
                        rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
                cv::imshow("mser", tmp);
                cv::waitKey(0);
            }
    }
    color_descr = multi_hist_color.clone();
    lbp_descr = multi_hist_lbp.clone();
}

/**
 * MSER feature extractor
 * This function is an overload of the above one. It computes also the rgb histogram descriptor
 *
 * @param img: input image.
 * @param color_descr: output parameter that contains the color descriptor of the regions.
 * @param lbp_descr: output parameter that contains the LBP descriptor of the regions.
 * @param rgb_descr: output parameter that contains the RGB color histogram of the regions.
 * @param col_on: flag to decide the inclusion of the color histogram.
 * @param lbp_on: flag to decide the inclusion of the lbp descriptor.
 * @param rgb_on: flag to decide the inclusion of the rgb descriptor.
 * @param hist_color_size: dimension of the x axis of the color histogram.
 * @param hist_lbp_size: dimension of the x axis of the lbp histogram.
 * @parma hist_rgb_size: dimension of the x axis of the rgb histogram.
 * @param radius: radius of the elbp descriptor.
 * @param neigh: neighbor of the elbp descriptor.
 */
void MSER_descriptor(const cv::Mat img, cv::Mat& color_descr, cv::Mat& lbp_descr, cv::Mat& rgb_descr, bool col_on, bool lbp_on, bool rgb_on, int hist_color_size, int hist_lbp_size, int hist_rgb_size, const int radius, const int neigh) {
    //std::cout << "MSER dio ca" << std::endl;
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    ms->setMinArea(30);
    ms->setMaxArea(2500);
    std::vector< std::vector< cv::Point> > regions;
    std::vector<cv::Rect> mser_bbox;
    ms->detectRegions(img, regions, mser_bbox);

    /*cv::Mat tmp = img.clone();
    for (int i = 0; i < regions.size(); i++)
        if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
            rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
    cv::imshow("mser", tmp);
    std::cout << "plotted regions... madonnola" << std::endl;*/

    float range_gray[] = { 0, hist_lbp_size };
    const float* hist_range = { range_gray };
    bool uniform = true, accumulate = false;

    cv::Mat hist_lbp, multi_hist_color, multi_hist_lbp, multi_hist_rgb;

    bool flag_first_region = true;

    for (int i = 0; i < regions.size(); i++) {
        if (mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10)
            continue;

        cv::Mat sub_img = (img.clone())(mser_bbox[i]);          //sub_region creation

        cv::Mat sub_elbp = elbp(sub_img, radius, neigh);        //lbp descriptor in sub_image
        cv::calcHist(&sub_elbp, 1, 0, cv::Mat(), hist_lbp, 1, &hist_lbp_size, &hist_range, uniform, accumulate);
        hist_lbp = hist_lbp.reshape(1, 1);
        cv::normalize(hist_lbp, hist_lbp, 0, 1, cv::NORM_MINMAX);

        /* rgb case */
        //cv::Mat hist_color = color_histogram(sub_img, true/*, hist_color_size*/);
        /* HUE-saturation */
        cv::Mat hist_color = color_histogram(sub_img);
        cv::Mat hist_rgb = color_histogram(sub_img, true, 100);

        //std::cout << "--   lbp hist size: " << multi_hist_lbp.size() << std::endl;
        //std::cout << "--   color hist size: " << multi_hist_color.size() << std::endl;

        if (flag_first_region) {
            flag_first_region = false;
            multi_hist_lbp = hist_lbp.clone();
            multi_hist_color = hist_color.clone();
            multi_hist_rgb = hist_rgb.clone();
        }
        else
            try {
            cv::vconcat(multi_hist_lbp, hist_lbp, multi_hist_lbp);
            cv::vconcat(multi_hist_color, hist_color, multi_hist_color);
            cv::vconcat(multi_hist_rgb, hist_rgb, multi_hist_rgb);
        }
        catch (const std::exception&) {
            std::cout << " lbp hist size: " << hist_lbp.size() << "\t";
            std::cout << " lbp MULTI_hist size: " << multi_hist_lbp.size() << std::endl;
            std::cout << " lbp hist size: " << hist_color.size() << "\t";
            std::cout << " lbp MULTI_hist size: " << multi_hist_color.size() << std::endl;
            cv::Mat tmp = img.clone();
            for (int i = 0; i < regions.size(); i++)
                if (!(mser_bbox[i].width <= 10 || mser_bbox[i].height <= 10))
                    rectangle(tmp, mser_bbox[i], CV_RGB(0, 255, 0));
            cv::imshow("mser", tmp);
            cv::waitKey(0);
        }
    }
    color_descr = multi_hist_color.clone();
    lbp_descr = multi_hist_lbp.clone();
    rgb_descr = multi_hist_rgb.clone();
}