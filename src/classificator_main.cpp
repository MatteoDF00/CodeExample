// classificator_main.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <fstream>

#include "classificator.h"
#include "descriptorsUtilities.h"
#include "segmentation.h"

const std::string trays_path("../../Food_leftover_dataset/tray");

const std::string beans_path("../../training_set/beans/");
const std::string basilpotatoes_path("../../training_set/basil_potatoes/");
const std::string salad_path("../../training_set/salad/");
const std::string pastapesto_path("../../training_set/pasta_with_pesto/");
const std::string pastatomato_path("../../training_set/pasta_with_tomato_sauce/");
const std::string rice_path("../../training_set/pilaw_rice_with_peppers_and_peas/");
const std::string fish_path("../../training_set/fish_cutlet/");
const std::string grilledpork_path("../../training_set/grilled_pork_cutlet/");
const std::string seafoodsalad_path("../../training_set/seafood_salad/");
const std::string rabbit_path("../../training_set/rabbit/");
const std::string pastameat_path("../../training_set/pasta_with_meat_sauce/");
const std::string pastaclams_path("../../training_set/pasta_with_clams_and_mussels/");

std::map<std::string, int> find_label_from_string = { {"pasta with tomato",0,},
                                                        {"pasta with pesto",1,},
                                                        {"rice",2,},
                                                        {"clams",3,},
                                                        {"fish cutlet",4,},
                                                        {"potatoes",5,},
                                                        {"grilled pork",6,},
                                                        {"beans",7,},
                                                        {"rabbit",8,},
                                                        {"seafood salad",9,},
                                                        {"salad",10,} };
std::map<int, std::string> food_map_classes = { {0,"pasta with tomato",},
                                                {1,"pasta with pesto",},
                                                {2,"rice",},
                                                {3,"clams",},
                                                {4,"fish cutlet",},
                                                {5,"potatoes",},
                                                {6,"grilled pork",},
                                                {7,"beans",},
                                                {8,"rabbit",},
                                                {9,"seafood salad",},
                                                {10,"salad",} };
std::map<int, std::string> food_map_paths = { {0,pastatomato_path,},
                                                {1,pastapesto_path,},
                                                {2,rice_path,},
                                                {3,pastaclams_path,},
                                                {4,fish_path,},
                                                {5,basilpotatoes_path,},
                                                {6,grilledpork_path,},
                                                {7,beans_path,},
                                                {8,rabbit_path,},
                                                {9,seafoodsalad_path,},
                                                {10,salad_path,} };
std::map<int, int> map_num_train = { {0,8,},    // key, number of training images for each class
                                     {1,6,},
                                     {2,7,},
                                     {3,7,},
                                     {4,5,},
                                     {5,8,},
                                     {6,8,},
                                     {7,8,},
                                     {8,6,},
                                     {9,5,},
                                     {10,9,} };
std::map<int, int> map_start_index = { {0,6,},    // key, starting index of the training images
                                       {1,5,},
                                       {2,1,},
                                       {3,1,},
                                       {4,1,},
                                       {5,1,},
                                       {6,1,},
                                       {7,6,},
                                       {8,1,},
                                       {9,1,},
                                       {10,10,} };

template<typename Map>
void PrintMap(Map& m) {
    std::cout << "[ ";
    for (auto& item : m) {
        std::cout << item.first << ":" << item.second << " ";
    }
    std::cout << "]\n";
}
void sublabel2global(std::vector<std::string> label_sub_string, std::vector<int> &label_glob_int) {
    for (int i = 0; i < label_sub_string.size(); i++) 
        label_glob_int.push_back(find_label_from_string[label_sub_string[i]]);
}
void getSubSets(std::vector<std::string> label_sub_string, std::map<int, std::string>& param_sub_food_map, std::map<int, std::string>& param_sub_path_map, std::map<int, int>& param_sub_train, std::map<int, int>& param_sub_index) {
    std::vector<int> param_ans;
    sublabel2global(label_sub_string, param_ans);
    for (int i = 0; i < param_ans.size(); i++) {
        param_sub_food_map.insert({ i,food_map_classes[param_ans[i]] });
        param_sub_path_map.insert({ i,food_map_paths[param_ans[i]] });
        param_sub_train.insert({ i,map_num_train[param_ans[i]] });
        param_sub_index.insert({ i,map_start_index[param_ans[i]] });
    }
}
int main(int argc, char** argv)
{

    //check if at least two paths are passed (the name of the program is included)
    if (argc < 3) {
        std::cout << "Please provide two image paths, one initial image and one leftover image." << std::endl;
        return -1;
    }

    //read the paths from command line
    //std::string tray = "tray4";
    //std::string before_path = "../../Food_leftover_dataset/" + tray + "/food_image.jpg";//argv[1];
    //std::string leftover_path = "../../Food_leftover_dataset/" + tray + "/leftover1.jpg"; //argv[2];
    std::string before_path = argv[1];
    std::string leftover_path = argv[2];

    //load the images
    cv::Mat before_img = cv::imread(before_path, cv::IMREAD_COLOR);
    cv::Mat leftover_img = cv::imread(leftover_path, cv::IMREAD_COLOR);

    //images with bounding boxes and labels drawn
    cv::Mat output_before_img = before_img.clone();
    cv::Mat output_leftover_img = leftover_img.clone();

    //check the validity of the provided paths
    if (before_img.empty() || leftover_img.empty()) {
        std::cout << "Wrong paths";
        return -1;
    }



    outtxt out("data.txt");

    //-----------------------------------------------FIND BOUNDING BOXES OF THE PLATES AND BOWL FOR BOTH IMAGES-------------------------------------------------------------------------

    std::vector<cv::Mat> before_circle_images; //vector of cropped plates
    std::vector<cv::Rect> before_plate_box = find_plates(before_img, before_circle_images, false); //returns the bounding boxes of the plates

    std::vector<cv::Mat> leftover_circle_images;
    std::vector<cv::Rect> leftover_plate_box = find_plates(leftover_img, leftover_circle_images, true);

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    //----------------------------------------------CHECK FOR SALAD LOCALIZATION AND FIND IT--------------------------------------------------------------------------------------------


        //we check for salad only if at least three circle objects were detected
    std::cout << "start salad localization" << std::endl;

    if (before_plate_box.size() >= 3) { //before image has salad

        SaladBox salad_rect = find_salad(before_plate_box);
        int index_before = salad_rect.index; //index of rect/circle image corresponding to salad

        cv::Mat salad_mask; //output mask of salad
        
        cv::Rect final_salad_box = box_mask_food(before_circle_images[index_before](before_plate_box[index_before]), salad_mask, before_plate_box[index_before]); //segmentation and output bounding box of salad

        cv::imwrite("before_salad_mask.jpg", salad_mask);

        float salad_confidence = salad_rect.confidence;
        before_plate_box.erase(before_plate_box.begin() + index_before); //remove the salad from the vector of things to segment and classify
        before_circle_images.erase(before_circle_images.begin() + index_before);

        std::string label_salad = "salad";
        draw_box(output_before_img, output_before_img, label_salad, cv::Scalar(0, 255, 255), final_salad_box);

        if (leftover_plate_box.size() >= 3) { //salad is still present
            SaladBox salad_rect_leftover = find_salad(leftover_plate_box);
            int index_leftover = salad_rect_leftover.index; //index of rect/circle image corresponding to salad
            float salad_confidence_leftover = salad_rect_leftover.confidence;
            cv::Mat salad_mask_leftover; //output mask of salad
            
            cv::Rect final_salad_box_leftover = box_mask_food(leftover_circle_images[index_leftover](leftover_plate_box[index_leftover]), salad_mask_leftover, leftover_plate_box[index_leftover]); //segmentation and output bounding box of salad
            cv::imwrite("leftover_salad_mask.jpg", salad_mask_leftover);
            draw_box(output_leftover_img, output_leftover_img, label_salad, cv::Scalar(0, 255, 255), final_salad_box_leftover);


            double leftover_salad = leftover_calculation(salad_mask, salad_mask_leftover);

            out.separator("Salad before");
            out.write(final_salad_box, label_salad, salad_confidence, leftover_salad);
            out.separator("Salad leftover");
            out.write(final_salad_box_leftover, label_salad, salad_confidence_leftover);

            leftover_plate_box.erase(leftover_plate_box.begin() + index_leftover); //remove the salad from the vector of things to segment and classify
            leftover_circle_images.erase(leftover_circle_images.begin() + index_leftover);
        }
        else { //salad not present anymore

            out.separator("Salad before");
            out.write(final_salad_box, label_salad, salad_confidence, 0);
        }

    }


    /*cv::imshow("Check insalata before image", output_before_img);
    cv::imshow("Check insalata leftover image", output_leftover_img);
    cv::waitKey(500);
    cv::destroyAllWindows();*/


    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   


    //-------------------------------------------------------------------------------BREAD--------------------------------------------------------------------------------------------------------
    
    //try to find bread in the before image
    std::cout << "start bread localization" << std::endl;

    cv::Rect large_box_bread = find_bread(before_img, before_plate_box, false); //bounding box of bread 

    if (!large_box_bread.empty()) { //potential bread found
        cv::Mat bread_mask;
        cv::Rect bread_box = box_mask_bread(before_img, bread_mask, large_box_bread); //refined bread box
        std::string label_bread = "bread";
        draw_box(output_before_img, output_before_img, label_bread, cv::Scalar(128, 0, 255), bread_box);
        cv::imwrite("before_bread_mask.jpg", bread_mask);

        cv::Rect large_box_bread_leftover = find_bread(leftover_img, leftover_plate_box, true);
        if (!large_box_bread_leftover.empty()) { //bread still present
            cv::Mat bread_mask_leftover;
            cv::Rect bread_box_leftover = box_mask_bread(leftover_img, bread_mask_leftover, large_box_bread_leftover); //search for bread in the leftover image
            draw_box(output_leftover_img, output_leftover_img, label_bread, cv::Scalar(128, 0, 255), bread_box_leftover);
            cv::imwrite("leftover_bread_mask.jpg", bread_mask_leftover);

            double leftover_bread = leftover_calculation(bread_mask, bread_mask_leftover);
            out.separator("Bread before");
            out.write(bread_box, label_bread, -1, leftover_bread);
            out.separator("Bread leftover");
            out.write(bread_box_leftover, label_bread, -1);
        }
        else { //leftover to zero
            out.separator("Salad before");
            out.write(bread_box, label_bread, -1, 0);
        }

    }


    /*cv::imshow("Check pane before image", output_before_img);
    cv::imshow("Check pane leftover image", output_leftover_img);
    cv::waitKey(2000);
    cv::destroyAllWindows();*/
    
    //-------------------------------------------------------------------------------SEGMENTATION-------------------------------------------------------------------------------

    std::vector<cv::Rect> before_box;
    std::vector<cv::Mat> before_mask;
    std::vector<cv::Mat> before_cut_food;

    std::vector<cv::Rect> leftover_box;
    std::vector<cv::Mat> leftover_mask;
    std::vector<cv::Mat> leftover_cut_food;

    std::string folder_output_path = ""; //da cambiare
    std::string prefix_output = "before_mask_food";

    for (int i = 0; i < before_circle_images.size(); i++)
        segmentation_task(before_circle_images[i], before_plate_box[i], folder_output_path, prefix_output + std::to_string(i), before_mask, before_box, before_cut_food, before_img);
    CV_Assert(before_cut_food.size() == before_box.size());
    CV_Assert(before_mask.size() == before_box.size());
    std::cout << "END SEGMENTATION ---- START TRAINING" << std::endl;


    // ordine dei flag (partendo dal 4° parametro):
    // flag SIFT descriptor    ->    consigliato a false (allunga il training di molto, senza notevoli improvements)
    // flag HSV descriptor     ->    true
    // flag LBP descriptro     ->    true
    // flag RGB descriptor     ->    false (non migliora, conviene risparmiare conti)
    BoW all_food_classifier(food_map_classes, 300, -1, false, true, true, false);
    all_food_classifier.train_classifier(food_map_paths, map_num_train, map_start_index);
    std::cout << "push to go" << std::endl;
    cv::waitKey(500);
    std::vector<output_classificator> before_results;
    for (int i = 0; i < before_cut_food.size(); i++) {
        //cv::imshow("image" + std::to_string(i), before_cut_food[i]);
        output_classificator tmp_data;
        all_food_classifier.test_classifier(before_cut_food[i], tmp_data.label, tmp_data.label_string, tmp_data.confidence, false, "classified");
        before_results.push_back(tmp_data);
    }
    out.separator("BEFORE IMAGE");
    out.write(before_box, before_results);
    for (int i = 0; i < before_results.size(); i++) {
        std::cout << "---- IMAGE " << i << " ----" << std::endl;
        std::cout << "label: " << before_results[i].label_string << " ---- confidence: " << before_results[i].confidence << std::endl;
    }
    if (before_results.size() != before_box.size())
        std::cout << "A food is missing in the before image" << std::endl;
    /* PLOT BOXES con label */
    for (int i = 0; i < before_box.size(); i++) 
        draw_box(output_before_img, output_before_img, before_results[i].label_string, color_vector[i+5], before_box[i]);
    
    cv::imshow("before_image", output_before_img);
    cv::imwrite("before_image.jpg", output_before_img);
    cv::waitKey(500);
    //cv::destroyAllWindows();


    std::cout << " ---- START LEFTOVER SEGMENTATION ----" << std::endl;

    folder_output_path = "";
    prefix_output = "leftover_mask_food";

    for (int i = 0; i < leftover_circle_images.size(); i++)
        segmentation_task(leftover_circle_images[i], leftover_plate_box[i], folder_output_path, prefix_output + std::to_string(i), leftover_mask, leftover_box, leftover_cut_food, leftover_img);

    std::cout << "END SEGMENTATION ---- BEGIN TRAINING" << std::endl;
    std::cout << "push to go" << std::endl;
    cv::waitKey(1000);
    
    std::vector<output_classificator> leftover_results, temp_results(before_results);
    std::map<int, std::string> sub_food_map, sub_path_map;
    std::map<int, int> sub_train, sub_index;
    std::vector<std::string> found_labels;
    for (int i = 0; i < temp_results.size(); i++)
        found_labels.push_back(temp_results[i].label_string);
    getSubSets(found_labels, sub_food_map, sub_path_map, sub_train, sub_index);
    std::cout << "food - ";PrintMap(sub_food_map);std::cout << std::endl;

    /* predict in the leftover */

    BoW sub_classifier(sub_food_map, 300, -1, false, true, true, false);
    sub_classifier.train_classifier(sub_path_map, sub_train, sub_index);

    for (int i = 0; i < leftover_cut_food.size(); i++) {
        output_classificator tmp_data;
        sub_classifier.test_classifier(leftover_cut_food[i], tmp_data.label, tmp_data.label_string, tmp_data.confidence, false);
        leftover_results.push_back(tmp_data);
    }

    out.separator("LEFTOVER IMAGE");
    out.write(leftover_box, leftover_results);

    /* debug, then in file */
    for (int i = 0; i < leftover_results.size(); i++) {
        std::cout << "---- IMAGE " << i << " ----" << std::endl;
        std::cout << "label: " << leftover_results[i].label_string << " ---- confidence: " << leftover_results[i].confidence << std::endl;
    }
    if (leftover_results.size() != leftover_box.size())
        std::cout << "A food is missing in the leftover image" << std::endl;

    /* PLOT BOXES with labels */
    for (int i = 0; i < leftover_box.size(); i++)
        draw_box(output_leftover_img, output_leftover_img, leftover_results[i].label_string, color_vector[i+5], leftover_box[i]);

    cv::imshow("leftover_image", output_leftover_img);
    cv::imwrite("leftover_image.jpg", output_leftover_img);
    cv::waitKey(500);

    //----------------------------------------------------LEFT OVER ESTIMATION-------------------------------------------------------------------------------------

    std::cout << "*********** START LEFTOVER ESTIMATION ***********" << std::endl;

    //we create a vector corresponding_foods of the same dimensions of the leftover_results. Each element in this vector is the corresponding best confidence match present in the original img. If we have two identical labels
    //in the leftover image, only one gests the food estimation with the best match. The other types of foods that do not have a match are considered as if they are not present anymore (leftover = 0)

    std::vector<output_classificator> corresponding_foods;
    std::vector<double> leftovers_ratio;
    std::vector<int> index_used;
    std::vector<output_classificator> copyresults(before_results); //copy of before_results that will contain the foods that are not found again in the leftover tray


    for (int i = 0; i < leftover_results.size(); ++i) {


        output_classificator best_match;
        best_match.confidence = 0;
        int best_index = -1;


        for (int j = 0; j < before_results.size(); ++j) {


            if (leftover_results[i].label_string == before_results[j].label_string && before_results[j].confidence > best_match.confidence) {
                best_match = before_results[j];
                best_index = j;
                std::string label_to_remove = before_results[j].label_string;
                copyresults.erase(std::remove_if(copyresults.begin(), copyresults.end(), [&label_to_remove](const output_classificator& classificator) {
                    return classificator.label_string == label_to_remove;
                    }), copyresults.end());
            }


        }

        corresponding_foods.push_back(best_match);
        double leftover = leftover_calculation(before_mask[best_index], leftover_mask[i]);
        leftovers_ratio.push_back(leftover);

    }


    //print check of matches (ci dovrebbe essere una print per ogni leftover food e ogni print dovrebbe avere una corrispondenza di label)


    //ci potrebbero essere leftovers_ratio = -1 che devo capire come gestire e basta setttare sopra nella funzione a 0 invece che -1 (per ora lascio così in modo che capisco se sta funzionando)
    out.separator("LEFTOVER ESTIMATION");
    for (int i = 0; i < leftover_results.size(); ++i) {

        std::cout << leftover_results[i].label_string << " matched with " << corresponding_foods[i].label_string << " with leftover ratio = " << leftovers_ratio[i] << "\n";
        std::string str = leftover_results[i].label_string + " matched with " + corresponding_foods[i].label_string + " with leftover ratio = " + std::to_string(leftovers_ratio[i]);
        out.write(str);
    }
    for (int i = 0; i < copyresults.size(); ++i) {
        std::cout << copyresults[i].label_string << " not found again with leftover ratio = 0" << "\n";
        std::string str = copyresults[i].label_string + " not found again with leftover ratio = 0";
        out.write(str);
    }
    out.close();

    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}