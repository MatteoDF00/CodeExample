#ifndef DESCRIPTORSUTILITIES_H
#define DESCRIPTORSUTILITIES_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

/*SIFT*/
cv::Mat SIFTkpPlot(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, const std::string winName);

cv::Mat SIFTdescriptor(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);

/*UTILS*/
std::string type2str(int type);

cv::Mat img_preprocessing(cv::Mat img, const int final_size = 500);

/*LBP*/
cv::Mat elbp(cv::Mat src_, int radius, int neighbors); //extended LBP

cv::Mat lbp_descriptor(const cv::Mat img, const int radius, const int neigh, int gray_range = 255, bool single_hist = true);

/*COLOR DESCRIPTORS*/
cv::Mat color_histogram(const cv::Mat& image);

cv::Mat color_histogram(const cv::Mat& image, bool rgb_case, int rgb_histSize);

/*MSER FEATURE EXTRACTOR*/
void MSER_descriptor(const cv::Mat img, cv::Mat& color_descr, cv::Mat& lbp_descr, bool col_on, bool lbp_on, int hist_color_size, int hist_lbp_size, const int radius, const int neigh);

void MSER_descriptor(const cv::Mat img, cv::Mat& color_descr, cv::Mat& lbp_descr, cv::Mat& rgb_descr, bool col_on, bool lbp_on, bool rgb_on, int hist_color_size, int hist_lbp_size, int hist_rgb_size, const int radius, const int neigh);

#endif