#include "segmentation.h"


/**
 * This function is used to localize the plates and salad inside the tray image. It finds the circular shapes consistent in dimension with
 * the plates and salad using the Hough Transform, and then computes the rectangular bounding boxes for each circle found.
 *
 * @param image: reference to the image where we want to localize the plates and salad.
 * @param circle_images: vector to store the individual images of the circles found.
 * @param leftover_flag: flag to indicate if the image is taken from the starting image or from the leftovers images.
 *
 * @return bounding_boxes: vector of rectangles indicating the bounding boxes found by the function.
 */
std::vector<cv::Rect> find_plates(const cv::Mat& image, std::vector<cv::Mat>& circle_images, bool leftover_flag)
{
    // Apply Bilateral filter to denoise the image while keeping the edges
    cv::Mat filtered;
    cv::bilateralFilter(image, filtered, 5, 75, 75);

    // Find the circles in the value channel of the HSV
    cv::Mat hsv;
    cv::cvtColor(filtered, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat value = channels[2];

    // Apply Hough Transform to find the circles
    std::vector<cv::Vec3f> circles; // Vector of circles
    if (leftover_flag) {
        // Apply histogram equalization to the value channel 
        cv::equalizeHist(value, value);
        cv::HoughCircles(value, circles, cv::HOUGH_GRADIENT, 3.5, 370, 275, 400, 180, 400);
    }
    else {
        cv::HoughCircles(value, circles, cv::HOUGH_GRADIENT, 3.5, 370, 275, 400, 180, 470);
    }

    // Now find rectangular bounding-boxes of each circle
    std::vector<cv::Rect> bounding_boxes;

    for (int i = 0; i < circles.size(); i++)
    {
        cv::Vec3f circle = circles[i];
        cv::Point center(cvRound(circle[0]), cvRound(circle[1])); // Coordinates of the center
        int radius = cvRound(circle[2]); // Radius of the circle

        // Ensure that the circle is within the image bounds
        int topleft_x = std::max(center.x - radius, 0);
        int topleft_y = std::max(center.y - radius, 0);
        int width = std::min(radius * 2, image.cols - topleft_x);
        int height = std::min(radius * 2, image.rows - topleft_y);

        // Create a black image with a white circle
        cv::Mat circle_image = cv::Mat::zeros(image.size(), CV_8UC3);
        cv::circle(circle_image, center, radius, cv::Scalar(255, 255, 255), -1);

        // Extract the circle region from the original image using bitwise AND
        cv::bitwise_and(image, circle_image, circle_image);

        // Add the circle image to the vector
        circle_images.push_back(circle_image);

        // Update the bounding box with the circle region
        cv::Rect bounding_box(topleft_x, topleft_y, width, height);
        bounding_boxes.push_back(bounding_box);
    }

    return bounding_boxes;
}

/**
 * This function finds the extreme points (minimum and maximum) of white pixels in a given binary mask.
 *
 * @param maschera: The binary mask (input image) represented by a cv::Mat object.
 * @return extreme_points: A vector of cv::Point containing the extreme points.
 */
std::vector<cv::Point> find_extreme_points(const cv::Mat& mask) {

    // Initialize an empty vector to store the extreme points.
    std::vector<cv::Point> extreme_points;

    // Create a temporary cv::Mat object to hold the coordinates of non-zero (white) pixels.
    cv::Mat white_pixels;

    // Find the coordinates of non-zero (white) pixels in the binary mask and store them in white_pixels.
    cv::findNonZero(mask, white_pixels);

    // Initialize two cv::Point variables to keep track of the minimum and maximum coordinates.
    cv::Point xmin_ymin = white_pixels.at<cv::Point>(0);
    cv::Point xmax_ymax = white_pixels.at<cv::Point>(0);

    // Loop through all the white pixels found in white_pixels.
    for (size_t i = 0; i < white_pixels.total(); ++i) {
        // Get the current white pixel's coordinates.
        cv::Point p = white_pixels.at<cv::Point>(i);

        // Update the xmin_ymin point by taking the minimum x and y coordinates.
        xmin_ymin.x = std::min(xmin_ymin.x, p.x);
        xmin_ymin.y = std::min(xmin_ymin.y, p.y);

        // Update the xmax_ymax point by taking the maximum x and y coordinates.
        xmax_ymax.x = std::max(xmax_ymax.x, p.x);
        xmax_ymax.y = std::max(xmax_ymax.y, p.y);
    }

    // Add the minimum and maximum points to the vector extreme_points.
    extreme_points.push_back(xmin_ymin);
    extreme_points.push_back(xmax_ymax);

    // Return the vector containing the extreme points.
    return extreme_points;
}

/**
 * This function segments the food region in the input image based on color thresholding and morphological operations,
 * then returns a bounding box around the segmented food area.
 *
 * @param image: reference to the input image to be segmented.
 * @param mask: The binary mask representing the area of interest for segmentation.
 * @param large_box: The bounding box of the plate where is located the plate.
 * @return bounding_box: The rectangle representing the bounding box around the segmented food area.
 */
cv::Rect box_mask_food(const cv::Mat& image, cv::Mat& mask, cv::Rect plate_box)
{
    // Convert the input image from BGR to HSV color space.
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Define lower and upper bounds for the color range to be segmented (food).
    cv::Scalar lowerColorBound(0, 65, 5);
    cv::Scalar upperColorBound(255, 255, 255);

    // Create an empty binary mask with the same size as the input image.
    mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // Apply color thresholding to the HSV image using the defined color range, 
    // setting pixels within the range to white and others to black in the mask.
    cv::inRange(hsvImage, lowerColorBound, upperColorBound, mask);

    // Perform morphological operations to clean up the mask.
    // Close operation fills small holes and gaps in the white regions.
    // Open operation removes small noise points from the white regions.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Perform another set of morphological operations with a larger kernel to further clean up the mask.
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Find extreme points (top-left and bottom-right) of the white regions in the mask.
    // These points are used to create a bounding box around the segmented food area.
    std::vector<cv::Point> extream_points = find_extreme_points(mask);

    // Create a rectangle using the extreme points to form a bounding box around the segmented food area.
    cv::Rect bounding_box(extream_points[0], extream_points[1]);
    bounding_box.x = bounding_box.x + plate_box.x;
    bounding_box.y = bounding_box.y + plate_box.y;
    // Return the bounding box, which can be used to extract the segmented food region from the original image.
    return bounding_box;

}

/**
 * This function segments the bread in the input image based on color thresholding and morphological operations,
 * then returns a bounding box around the segmented bread.
 *
 * @param image: Reference to the input image where the bread is to be segmented.
 * @param mask: The binary mask representing the area of interest for segmentation. This will be filled with the segmented bread region.
 * @param large_box: The bounding box of the plate where the bread is located.
 * @return bounding_box: The rectangle representing the bounding box around the segmented bread area.
 */
cv::Rect box_mask_bread(const cv::Mat& image, cv::Mat& mask, cv::Rect large_box)
{

    // Convert the extracted foreground from BGR to HSV color space.
    cv::Mat hsvImage;
    cv::cvtColor(image(large_box), hsvImage, cv::COLOR_BGR2HSV);

    // Define lower and upper bounds for the color range to be segmented (bread).
    cv::Scalar lowerColorBound(15, 75, 80);
    cv::Scalar upperColorBound(21, 200, 230);

    // Create an empty binary mask with the same size as the large_box region.
    mask = cv::Mat::zeros(image(large_box).size(), CV_8UC1);

    // Apply color thresholding to the HSV image using the defined color range, 
    // setting pixels within the range to white and others to black in the mask.
    cv::inRange(hsvImage, lowerColorBound, upperColorBound, mask);

    // Perform morphological operations to clean up the mask.
    // The close operation fills small holes and gaps in the white regions.
    // The open operation removes small noise points from the white regions.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(45, 45));

    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    if (!mask.empty()) {
        // Find extreme points (top-left and bottom-right) of the white regions in the mask.
        // These points are used to create a bounding box around the segmented bread area.
        std::vector<cv::Point> extream_points = find_extreme_points(mask);

        // Create a rectangle using the extreme points to form a bounding box around the segmented bread area.
        cv::Rect bounding_box(extream_points[0], extream_points[1]);
        // Translate the bounding box coordinates to the original image space (large_box might be a subregion of the original image).
        bounding_box.x = bounding_box.x + large_box.x;
        bounding_box.y = bounding_box.y + large_box.y;

        // Return the bounding box, which can be used to extract the segmented bread region from the original image.
        return bounding_box;
    }
    else
    {
        //If the mask is empty (no bread region detected), return the original large_box as a fallback.
        return large_box;
    }

}

/**
 * This function processes the input image by extracting the Luv channel and applying mean shift filtering.
 *
 * @param image: reference to the input image to be processed.
 */
cv::Mat process_image(const cv::Mat& image) {
    // Convert the input image to Luv color space
    cv::Mat luv_image;
    cv::cvtColor(image, luv_image, cv::COLOR_BGR2Luv);

    // Split the Luv image into channels
    std::vector<cv::Mat> luv_channels;
    cv::split(luv_image, luv_channels);

    // Set the H and V channels to zero
    luv_channels[2] = cv::Mat::zeros(luv_channels[2].size(), luv_channels[2].type());
    luv_channels[1] = cv::Mat::zeros(luv_channels[1].size(), luv_channels[1].type());

    // Merge the channels back to Luv image
    cv::Mat filtered_luv;
    cv::merge(luv_channels, filtered_luv);

    // Apply mean shift filtering
    cv::Mat meanShift;
    cv::pyrMeanShiftFiltering(filtered_luv, meanShift, 90, 90);
    cv::split(meanShift, luv_channels);
    for (int y = 0; y < luv_channels[0].rows; ++y) {
        for (int x = 0; x < luv_channels[0].cols; ++x) {
            // Se il valore del pixel è inferiore a 10, impostalo a zero
            if (luv_channels[0].at<uchar>(y, x) < 18) {
                luv_channels[0].at<uchar>(y, x) = 0;
            }
        }
    }
    cv::merge(luv_channels, meanShift);

    // Update the image with the mean shifted result
    return meanShift;
}

/**
 *
 * This function calculates the average silhouette value for a clustering result based on the provided labels and distances.
 *
 *
 * @param labels: A matrix containing the cluster labels assigned to each sample. It has dimensions (n_samples, 1),
 * where numSamples is the number of samples.
 * @param distance: A matrix containing the pairwise distances between samples. It has dimensions (n_samples, n_clusters),
 * where numClusters is the number of clusters.
 * @return  silhouetteAVG: The average silhouette value, which indicates the quality of the clustering result.
 * Higher values indicate better separation and coherence within clusters.
 */
double calculateSilhouette(const cv::Mat& labels, const cv::Mat& distances) {
    double sum_sil = 0;
    int n_samples = labels.rows;

    for (int i = 0; i < n_samples; i++) {

        int cluster_label = labels.at<int>(i);
        double intra_cdistance = distances.at<float>(i, cluster_label);

        // Calculate the average intra-cluster distance
        double avg_intra_cdistance = intra_cdistance / (n_samples - 1);

        // Initialize the minimum inter-cluster distance to a maximum value
        double min_inter_cdistance = std::numeric_limits<double>::max();

        // Find the minimum inter-cluster distance for the current sample
        for (int j = 0; j < distances.cols; j++) {
            if (j != cluster_label) {
                double interClusterDistance = distances.at<float>(i, j);
                min_inter_cdistance = std::min(min_inter_cdistance, interClusterDistance);
            }
        }

        // Calculate the silhouette value for the current sample
        double value_sil = 0;
        if (n_samples > 1)
            value_sil = (min_inter_cdistance - avg_intra_cdistance) / std::max(avg_intra_cdistance, min_inter_cdistance);
        // Accumulate the silhouette value for all samples
        sum_sil += value_sil;
    }
    // Calculate the average silhouette value
    return sum_sil / n_samples;
}

/**
 *
 * This function estimates the optimal number of clusters for a given image using the k-means clustering algorithm and the silhouette score.
 *
 *
 * @param image: The input image for clustering.
 * @return optimal_clusters: The estimated optimal number of clusters for the given image. This value indicates the number of distinct regions
 * or groups present in the image that best capture its structure or content.
 */
int estimate_opt_clusters(const cv::Mat image) {

    // Convert the image to a matrix of floating-point samples
    cv::Mat samples;
    image.convertTo(samples, CV_32F);
    samples = samples.reshape(1);

    // Set the maximum number of clusters(max three food for each image)
    int max_clusters = 4;

    // Initialize variables for tracking the best silhouette score and optimal number of clusters
    cv::Mat labels, centers;
    double best_sil = -1;
    int optimal_clusters = 0;

    for (int k = 2; k <= max_clusters; k++) {

        // Set the convergence criteria for k-means algorithm
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);

        // Set the number of attempts and center initialization method for k-means algorithm
        int attempts = 50;
        int flags = cv::KMEANS_RANDOM_CENTERS;

        // Apply k-means clustering to the samples
        cv::kmeans(samples, k, labels, criteria, attempts, flags, centers);

        // Calculate pairwise distances between samples and cluster centers
        cv::Mat distances(samples.rows, k, CV_32F);
        for (int i = 0; i < samples.rows; i++) {
            for (int j = 0; j < k; j++) {
                distances.at<float>(i, j) = cv::norm(samples.row(i) - centers.row(j));
            }
        }

        // Calculate the mean silhouette score for the current number of clusters
        double mean_sil = calculateSilhouette(labels, distances);

        // Update the best silhouette score and optimal number of clusters if necessary
        if (mean_sil - best_sil > 0.005) {
            best_sil = mean_sil;
            optimal_clusters = k;
        }
    }
    // Return the optimal cluster
    return optimal_clusters;
}

/**
 *
 * This function creates a segmented image by applying k-means clustering to the input image and assigning colors to the clusters.
 *
 *
 * @param image: reference to the input image to be segmented.
 * @param num_clusters: The desired number of clusters for segmentation.
 * @return result_segmented: The segmented image where each pixel is assigned a color based on the cluster it belongs to.
 *
 */
cv::Mat create_segm_image(const cv::Mat& image, int num_clusters) {

    // Reshape and convert the input image to the appropriate format
    cv::Mat sample_image = image.reshape(1, image.rows * image.cols);
    sample_image.convertTo(sample_image, CV_32F);

    // Perform k-means clustering on the image pixels    
    cv::Mat labels, centres;
    cv::kmeans(sample_image, num_clusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 50, cv::KMEANS_RANDOM_CENTERS, centres);

    // Create a blank segmented image of the same size as the input image   
    cv::Mat result_segmented = cv::Mat::zeros(image.size(), CV_8UC3);

    // Assign colors to the clusters in the segmented image
    cv::Scalar color;
    for (int i = 0; i < num_clusters; i++) {
        color = color_vector[i];
        for (int j = 0; j < labels.rows; j++) {
            if (labels.at<int>(j) == i) {
                int row = j / image.cols;
                int col = j % image.cols;
                result_segmented.at<cv::Vec3b>(row, col) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
    // Return the segmented image
    return result_segmented;
}

/**
 * This function performs the segmentation of an input image based on a given mask indicating the region of interest (ROI).
 *
 * @param image: reference to the input image to be segmented.
 * @param mask: a binary mask indicating the region of interest (ROI) in the image where segmentation will be applied.
 * @param segmented: output matrix where the segmented image will be stored.
 * @return num_clusters: the estimated number of clusters used for the segmentation, including the background.
 */
int segment_food(const cv::Mat& image, cv::Mat& mask, cv::Mat& segmented) {

    // Create a copy of the input image to work with.
    cv::Mat temp = image.clone();

    // Preprocess the image by applying the 'process_image' function.
    cv::Mat filter_image = process_image(temp);

    // Copy the processed image back to the original image, but only in the masked area (ROI).
    // This step replaces the pixels in the region of interest with the processed values.
    filter_image.copyTo(temp, mask);

    // Estimate the optimal number of clusters for segmentation, taking into account both the main course and the background.
    // The '+1' is added to consider the background as one of the clusters.
    int num_clusters = estimate_opt_clusters(temp) + 1;

    // Perform the final image segmentation based on the estimated optimal number of clusters.
    // The resulting segmented image will contain regions corresponding to different components, including the main course and background.
    segmented = create_segm_image(temp, num_clusters);

    // Return the estimated number of clusters used for segmentation, which includes the background.
    return num_clusters;
}

/**
 * This function creates a binary mask for a specific color in the input image.
 *
 * @param image: The input image for which to create the color mask.
 * @param
 * @return mask: A binary mask where pixels corresponding to the specific color are set to 255 (white),
 *               and all other pixels are set to 0 (black).
 */
cv::Mat createColorMask(const cv::Mat& image, cv::Scalar color) {

    // Create an empty mask of the same size as the input image
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

    // Convert the color from BGR to Scalar (Blue, Green, Red) format
    cv::Scalar targetColorBGR(color[0], color[1], color[2]);

    // Create a binary mask where pixels corresponding to the specified color are set to 255 (white)
    // and all other pixels are set to 0 (black)
    cv::inRange(image, targetColorBGR, targetColorBGR, mask);

    return mask;
}
/**
 * This function finds masks for the main course foods based on specified colors.
 *
 * @param image: The input image of the plate with food.
 * @param total_food: The total number of food items to find masks for.
 * @param food_box_: A vector of Rectangles representing the bounding boxes of the  foods in the image (output parameter).
 * @param plate_box: A Rect representing the bounding box of the plate in the image.
 * @return food_mask_mainCourse: A vector of Mat containing the masks for the main course foods.
 */
std::vector<cv::Mat> find_mask_box(const cv::Mat& image, int total_food, std::vector<cv::Rect>& food_box_, const cv::Rect plate_box) {

    std::vector<cv::Point> extreme_points;
    std::vector<cv::Mat> food_mask_;
    food_box_.clear();

    // Loop over the total_food items (colors) specified
    for (int j = 0; j < total_food; j++) {
        cv::Vec3b pixel_color = image.at<cv::Vec3b>(0, 0);
        cv::Scalar color = color_vector[j];

        // Check if the pixel color matches the chosen color
        if (!(pixel_color[0] == color[0] && pixel_color[1] == color[1] && pixel_color[2] == color[2])) {
            // If the pixel color doesn't match, create a color mask for the specific color
            cv::Mat temp = createColorMask(image, color);

            // Store the color mask in the food_mask_mainCourse vector
            food_mask_.push_back(temp);

            // Find extreme points of the mask to create a bounding box
            extreme_points = find_extreme_points(temp);

            // Create a bounding box for the main course food item and adjust it with the plate_box
            cv::Rect box(extreme_points[0], extreme_points[1]);
            ;
            box.x = box.x + plate_box.x;
            box.y = box.y + plate_box.y;

            // Store the bounding box in the food_box_mainCourse vector
            food_box_.push_back(box);


        }
    }

    // Return the vector of food masks for the main course items
    return food_mask_;
}


bool intersection_boxes(cv::Mat image, cv::Rect rect1, cv::Rect rect2, double threshold, int& num_white_union, int& num_white_inters, bool verbose_plot) {
    cv::Mat mask1 = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::rectangle(mask1, rect1, cv::Scalar(255, 255, 255), -1);
    cv::Mat mask2 = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::rectangle(mask2, rect2, cv::Scalar(255, 255, 255), -1);

    cv::Mat mask_union, mask_inters;
    cv::bitwise_or(mask1, mask2, mask_union);
    cv::bitwise_and(mask1, mask2, mask_inters);

    if (verbose_plot) {
        cv::imshow("mask1", mask1);
        cv::imshow("mask2", mask2);
        cv::imshow("mask_union", mask_union);
        cv::imshow("mask_inters", mask_inters);
    }

    num_white_union = cv::countNonZero(mask_union);
    num_white_inters = cv::countNonZero(mask_inters);

    std::cout << "#white union: " << num_white_union << "\t #white inters: " << num_white_inters << std::endl;
    float ratio = (float)num_white_inters / (float)num_white_union;
    std::cout << "ratio:" << ratio << std::endl;
    if (ratio >= threshold)
        return true;
    else
        return false;
}


/**
* This function performs image segmentation on the given circular region of interest (ROI) to identify food items within it.
* It also cuts out the individual food items and stores them in a separate vector.
*
* @param circle_images The input image containing circular shapes with food items.
* @param plate_box The bounding box representing the circular region of interest (plate) in the input image.
* @param folder_output_path The output folder where the segmented food items will be saved as images.
* @param prefix_output The prefix to be used for generating output image filenames.
* @param mask_vect A vector to store the masks of the segmented food items.
* @param box_vect A vector to store the bounding boxes of the segmented food items.
* @param cut_image_vect A vector to store the cut-out images of the segmented food items.
* @param
*/
void segmentation_task(const cv::Mat& circle_images, const cv::Rect& plate_box, const std::string& folder_output_path, const std::string& prefix_output, std::vector<cv::Mat>& mask_vect, std::vector<cv::Rect>& box_vect, std::vector<cv::Mat>& cut_image_vect, const cv::Mat& reference_image)
{
    // Extract the region of interest (ROI) from the input circle_images using the provided plate_box
    cv::Mat temp = circle_images(plate_box);
    //cv::imshow("plate", circle_images(plate_box));
    cv::waitKey(500);
    // Initialize a mask and compute a region of interest (ROI) using box_mask_food function
    cv::Mat mask;
    cv::Rect roi = box_mask_food(temp, mask, plate_box);

    cv::Mat image;
    temp.copyTo(image, mask);

    // Perform image segmentation on the ROI using the segment function
    cv::Mat segmented;
    int num_clusters = segment_food(image, mask, segmented);


    // Initialize a vector to store bounding boxes for the segmented objects (food)
    std::vector<cv::Rect> food_box;

    // Find masks for the individual objects using find_mask_box function
    std::vector<cv::Mat> food_mask = find_mask_box(segmented, num_clusters, food_box, plate_box);


    int un, inter;
    if (intersection_boxes(reference_image, food_box[0], food_box[1], 0.5, un, inter, false))
    {
        cut_image_vect.push_back(image);
        mask_vect.push_back(mask);
        box_vect.push_back(roi);

        // Generate the output filename using the provided folder_output_path, prefix_output, and index (j)
        std::string outputFilename = folder_output_path + prefix_output + ".jpg";

        // Save the segmented object as an image using OpenCV's imwrite function
        cv::imwrite(outputFilename, mask);
    }
    else {

        // Loop through the segmented objects
        for (int j = 0; j < food_mask.size(); j++)
        {
            // Create a cut-out image of the segmented food item using the mask
            cv::Mat cut;
            image.copyTo(cut, food_mask[j]);

            // Store the cut image, the mask, and bounding box for each segmented object in the corresponding vectors
            cut_image_vect.push_back(cut);
            mask_vect.push_back(food_mask[j]);
            box_vect.push_back(food_box[j]);

            // Generate the output filename using the provided folder_output_path, prefix_output, and index (j)
            std::string outputFilename = folder_output_path + prefix_output + "_" + std::to_string(j) + ".jpg";

            // Save the segmented object as an image using OpenCV's imwrite function
            cv::imwrite(outputFilename, food_mask[j]);
            //cv::imshow("mask", food_mask[j]);
            cv::waitKey(500);
        }
    }
}


cv::Rect find_bread(cv::Mat& image, std::vector<cv::Rect> bounding_boxes, bool leftover_flag)
{

    //work in the second channel of the LAB color space
    cv::Mat lab_img;
    cv::cvtColor(image, lab_img, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab_img, channels);
    cv::Mat a_channel = channels[1];

    //create a white mask with the same size as the original image
    cv::Mat white_mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;

    for (const cv::Rect& bbox : bounding_boxes)
    {
        // set the pixels inside the rect to black
        white_mask(bbox) = 0;
    }

    //apply the mask to the original image
    cv::Mat no_plates_img;
    cv::bitwise_and(a_channel, white_mask, no_plates_img);

    //apply a threshold consistent with the range of values of bread
    cv::inRange(no_plates_img, 140, 150, no_plates_img);

    // remove small regions with morphological erosion
    cv::Mat eroded_img;
    if (leftover_flag) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::erode(no_plates_img, eroded_img, kernel);
    }
    else {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
        cv::erode(no_plates_img, eroded_img, kernel);
    }

    //check if bread (ideally) was found
    if (cv::countNonZero(eroded_img)) {
        cv::Moments moments = cv::moments(eroded_img, true);

        //calculate the centroid coordinates ("centre of mass")
        cv::Point centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

        int square_size = 300;  // Square size, consistent with bread dimensions

        //calculate the top-left corner of the roi
        cv::Point roi_top_left(centroid.x - square_size / 2, centroid.y - square_size / 2);

        //make sure that the rect is inside the image bound
        int image_width = eroded_img.cols;
        int image_height = eroded_img.rows;

        roi_top_left.x = std::max(roi_top_left.x, 0);
        roi_top_left.y = std::max(roi_top_left.y, 0);

        roi_top_left.x = std::min(roi_top_left.x, image_width - square_size);
        roi_top_left.y = std::min(roi_top_left.y, image_height - square_size);

        // Create the roi rectangle
        cv::Rect roi_rect(roi_top_left, cv::Size(square_size, square_size));

        return roi_rect;
    }

    //return an empty rectangle if bread is not found
    return cv::Rect();

}


/**
 *
 * This function is used to find the smallest bounding box (representing salad) when in the original image, more then three round objects are found.
 * It then calculates the confidence score of the prediction using the a metric based on the ratio between the area of the smallest box (identified as salad)
 * and the area of the second smallest box.
 *
 * @param bounding_boxes: reference to a vector of bounding_boxes calculated by the find_plates function.
 * @return SaladBox structure, with the rectangular bounding box of the salad and its relative confidence score.
 */
SaladBox find_salad(std::vector<cv::Rect>& bounding_boxes)
{
    //salad can be found only if there are at least three boxes in the tray
    if (bounding_boxes.size() < 3)
    {
        throw std::invalid_argument("There must be at least three boxes in order to have salad");
    }

    //sort the bounding boxes based on area
    std::vector<cv::Rect> sorted_boxes = bounding_boxes;
    //std::sort standard function that sorts a range of elements
    //sorted_boxes.begin(),sorted_boxes.end() range of elements to be sorted (pointers are returned)
    //[](const cv::Rect& a, const cv::Rect& b) {return a.area() < b.area();} lambda function that defines the criterion for the sorting algo.
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() < b.area();
        });

    //find the smallest and the second smallest bounding box and calculate confidence score
    cv::Rect smallestBox = sorted_boxes[0];
    cv::Rect secondSmallestBox = sorted_boxes[1];

    //calculate the confidence score based on the size difference between the smallest and second smallest boxes
    double confidence = 1.0 - pow(static_cast<double>(smallestBox.area()) / secondSmallestBox.area(), 2);

    int index = -1;
    for (int i = 0; i < bounding_boxes.size(); ++i) {
        if (bounding_boxes[i].x == smallestBox.x && bounding_boxes[i].y == smallestBox.y) {
            index = i;
        }
    }
    //return the smallest bounding box and confidence score
    return { smallestBox, confidence, index };
}
void draw_box(cv::Mat image, cv::Mat& modified_img, const std::string& label, const cv::Scalar& color, const cv::Rect& rect)
{
    modified_img = image.clone();  //copy the original image

    //draw the rectangle in the original image
    cv::rectangle(modified_img, rect, color, 2);

    //draw the text inside the box
    cv::putText(modified_img, label, cv::Point(rect.x + 20, rect.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2); //normal size sans-serif font (era il primo che c'era e sembra ok)
}
/**
*
* Function that calculates the leftover estimate.
*
* @param before_mask: mask of the before food
* @param leftover_mask: mask of the leftover food
* @return leftover estimation percentadge
*/

double leftover_calculation(const cv::Mat& before_mask, const cv::Mat& leftover_mask) {

    // count white pixels in the first and second image
    double num_white_1 = static_cast<double>(cv::countNonZero(before_mask));
    double num_white_2 = static_cast<double>(cv::countNonZero(leftover_mask));

    // check if the number of pixels in the second image is smaller
    if (num_white_2 < num_white_1) {
        // return the pixel count in the first image divided by the pixel count in the second image
        return num_white_2 / num_white_1;
    }
    else {
        // return -1.0 when the number of pixels in the second image is not smaller
        return -1.0;
    }
}

outtxt::outtxt() {
    outputfile.open(text_file);
}
outtxt::outtxt(std::string newname) {
    outputfile.open(newname);
}
void outtxt::write(std::string msg) {
    outputfile << msg << std::endl;
}
void outtxt::separator(std::string msg) {
    outputfile << "----------" << msg << "----------" << std::endl;
}
void outtxt::write(cv::Rect bbox, output_classificator res) {
    outputfile << "BOX: [";
    outputfile << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]; ";
    outputfile << "LABEL: " << res.label_string;
    outputfile << "; CONFIDENCE: " << res.confidence << std::endl;
}
void outtxt::write(cv::Rect bbox, output_classificator res, double leftov) {
    outputfile << "BOX: [";
    outputfile << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]; ";
    outputfile << "LABEL: " << res.label_string;
    outputfile << "; CONFIDENCE: " << res.confidence;
    outputfile << "; LEFTOVER RATIO: " << leftov << std::endl;
}
void outtxt::write(std::vector<cv::Rect> bbox, std::vector<output_classificator> res) {
    CV_Assert(bbox.size() == res.size());
    for (int i = 0; i < bbox.size(); i++)
        write(bbox[i], res[i]);
}
void outtxt::write(std::vector<cv::Rect> bbox, std::vector<output_classificator> res, std::vector<double> leftov) {
    CV_Assert(bbox.size() == res.size());
    for (int i = 0; i < bbox.size(); i++)
        write(bbox[i], res[i], leftov[i]);
}
void outtxt::write(cv::Rect bbox, std::string str, float conf) {
    outputfile << "BOX: [";
    outputfile << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]; ";
    outputfile << "LABEL: " << str;
    if (conf != -1)
        outputfile << "; CONFIDENCE: " << conf << std::endl;
    else
        outputfile << std::endl;
}
void outtxt::write(cv::Rect bbox, std::string str, float conf, double leftov) {
    outputfile << "BOX: [";
    outputfile << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]; ";
    outputfile << "LABEL: " << str;
    if (conf != -1)
        outputfile << "; CONFIDENCE: " << conf;
    outputfile << "; LEFTOVER RATIO: " << leftov << std::endl;
}
void outtxt::close() {
    outputfile.close();
}
