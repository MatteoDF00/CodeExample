#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <fstream>

struct output_classificator {
    float confidence;
    int label;
    std::string label_string;
};
struct SaladBox {
    cv::Rect box;
    double confidence;
    int index;
};

const  std::vector<cv::Scalar> color_vector = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 128, 255), cv::Scalar(204, 0, 0), cv::Scalar(102, 0, 204), cv::Scalar(0, 204, 0), cv::Scalar(204, 0, 102), cv::Scalar(0, 0, 204) };

std::vector<cv::Rect> find_plates(const cv::Mat& image, std::vector<cv::Mat>& circle_images, bool leftover_flag);

std::vector<cv::Point> find_extreme_points(const cv::Mat& mask);

cv::Rect box_mask_food(const cv::Mat& image, cv::Mat& mask, cv::Rect plate_box);

cv::Rect box_mask_bread(const cv::Mat& image, cv::Mat& mask, cv::Rect large_box);

cv::Mat process_image(const cv::Mat& image);

double calculateSilhouette(const cv::Mat& labels, const cv::Mat& distances);

int estimate_opt_clusters(const cv::Mat image);

cv::Mat create_segm_image(const cv::Mat& image, int num_clusters);

int segment_food(const cv::Mat& image, cv::Mat& mask, cv::Mat& segmented);

cv::Mat createColorMask(const cv::Mat& image, cv::Scalar color);

std::vector<cv::Mat> find_mask_box(const cv::Mat& image, int total_food, std::vector<cv::Rect>& food_box_, const cv::Rect plate_box);

bool intersection_boxes(cv::Mat image, cv::Rect rect1, cv::Rect rect2, double threshold, int& num_white_union, int& num_white_inters, bool verbose_plot);

void segmentation_task(const cv::Mat& circle_images, const cv::Rect& plate_box, const std::string& folder_output_path, const std::string& prefix_output, std::vector<cv::Mat>& mask_vect, std::vector<cv::Rect>& box_vect, std::vector<cv::Mat>& cut_image_vect, const cv::Mat& reference_image);

cv::Rect find_bread(cv::Mat& image, std::vector<cv::Rect> bounding_boxes, bool leftover_flag);

SaladBox find_salad(std::vector<cv::Rect>& bounding_boxes);

void draw_box(cv::Mat image, cv::Mat& modified_img, const std::string& label, const cv::Scalar& color, const cv::Rect& rect);

double leftover_calculation(const cv::Mat& before_mask, const cv::Mat& leftover_mask);

// CLASS OUTPUT
class outtxt {
public:
    std::string text_file = "data.txt";
    outtxt();
    outtxt(std::string newname);
    void separator(std::string msg);
    void write(std::string msg);
    void write(cv::Rect bbox, output_classificator res);
    void write(cv::Rect bbox, output_classificator res, double leftov);
    void write(cv::Rect bbox, std::string str, float conf); //AGGIUNTA
    void write(cv::Rect bbox, std::string str, float conf, double leftov);
    void write(std::vector<cv::Rect> bbox, std::vector<output_classificator> res);
    void write(std::vector<cv::Rect> bbox, std::vector<output_classificator> res, std::vector<double> leftov);
    void close();
private:
    std::ofstream outputfile;
};

#endif // !SEGMENTATION_H

