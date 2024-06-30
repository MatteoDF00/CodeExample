#ifndef CLASSIFICATOR_H
#define CLASSIFICATOR_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

#include "descriptorsUtilities.h"

class BoW{
public:
	BoW(std::map<int, std::string> class_names, const int img_resize = 400, const int number_clusters = -1, const bool sift_descr = true, const bool color_descr = false, const bool lbp_descr = false, bool rgb_descr = false);
	
	void setCriteria(int num_iter, int epsi);

	int SVMpredict(cv::Mat img);
	int SVMpredict(cv::Mat img, float& out);
	void BOWtrain();
	std::string label2class(int label);

	void train_classifier(std::map<int, std::string> food_paths, std::map<int, int> map_num_train, std::map<int, int> map_start_index);
	int test_classifier(const cv::Mat& img, std::string& string_ans, bool plot = false, std::string img_name = "image");
	void test_classifier(const cv::Mat& img, int& label, std::string& string_ans, float& conf, bool plot = false, std::string img_name = "image");
	
	int n_samples = 0, n_clusters = 0;
private:
	cv::Mat allDescr, allLBPDescr, allColorDescr, allRGBDescr;
	std::vector<cv::Mat> allDescrxImg, allColorDescrxImg, allLBPDescrxImg, allRGBDescrxImg;
	std::vector<int> allLabels;
	cv::Ptr<cv::ml::SVM> svm;
	cv::Mat Kcentroids, Klabels;
	cv::Mat COLcentroids, COLlabels;
	cv::Mat LBPcentroids, LBPlabels;
	cv::Mat RGBcentroids, RGBlabels;
	cv::Mat inputData, inputDataLables;

	int n_iter = 20, epsilon = 0.01, attempts = 8;
	int dict_size = 20, images_resize = 400;
	int color_hist_size = 100, lbp_hist_size = 200, rgb_hist_size = 100;
	std::vector<std::string> classes;
	bool flag_color_descriptor = false, flag_mser_lbp_descriptor = false, flag_sift_descriptor = true, flag_rgb_descr = false;
	/* SVM variables */
	//bool save_train = false, load_train = false;
	//std::string save_train_name = "pretrained.xml", load_train_name = "pretrained.xml";

	void compute_descriptors(const int label_name, const std::string img_folder, const std::string img_name, const std::string img_extension, const int n_images, const int start_index = 0);
	void kmeans();

	cv::Mat getDataVector(cv::Mat descriptors, const char kmeans_type = 's');

	cv::Mat concatenate_descriptors(int descriptor_index);
	cv::Mat concatenate_descriptors(cv::Mat img);

	void getHistogram();
	void SVMtrain();
};

#endif