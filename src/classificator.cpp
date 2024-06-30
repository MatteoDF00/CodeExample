#include "classificator.h"

/**
 * BoW constructor
 *
 * @param class_names: dictionary of key = label, value = label_string
 * @param img_resize: dimension to "normalize" all the training images
 * @param number_clusters: if needed change the number of clusters of kmeans
 * @param sift_descr: activate or not the sift descriptor
 * @param color_descr: include or not hsv color descriptor
 * @param lbp_descr: include or not lbp descriptor
 * @param rgb_descr: include or not rgb histogram descriptor
 */
BoW::BoW(std::map<int, std::string> class_names, const int img_resize, const int number_clusters, const bool sift_descr, const bool color_descr, const bool lbp_descr, bool rgb_descr) {
    for (int i = 0; i < class_names.size(); i++)
        classes.push_back(class_names[i]);

    images_resize = img_resize;
    flag_sift_descriptor = sift_descr;
    flag_color_descriptor = color_descr;
    flag_mser_lbp_descriptor = lbp_descr;
    flag_rgb_descr = rgb_descr;

    if (number_clusters == -1)
        dict_size = classes.size();
    else
        dict_size = number_clusters;
}

void BoW::compute_descriptors(const int label_name, const std::string img_folder, const std::string img_name, const std::string img_extension, const int n_images, const int start_index) {
    std::cout << "training food: " << img_folder << std::endl;
    for (int i = start_index; i < n_images + start_index; i++) {
        std::vector<cv::KeyPoint> keypoints_SIFT;
        std::string img_path = img_folder + img_name + std::to_string(i) + img_extension;
        cv::Mat img = cv::imread(img_path);

        if (img.empty()) {
            std::cout << " compute descriptor ---- NO IMAGE: " << std::endl;
            break;
        }

        img = img_preprocessing(img, images_resize);

        n_samples++;
        allLabels.push_back(label_name);

        if (flag_sift_descriptor) {
            cv::Mat descr_SIFT = SIFTdescriptor(img, keypoints_SIFT);

            allDescr.push_back(descr_SIFT);
            allDescrxImg.push_back(descr_SIFT);
        }
        if (flag_color_descriptor && flag_mser_lbp_descriptor) {
            cv::Mat descr_color, descr_lbp, descr_rgb;
            /*MODIFIED*/
            if (flag_rgb_descr)
                MSER_descriptor(img, descr_color, descr_lbp, descr_rgb, true, true, true, color_hist_size, lbp_hist_size, rgb_hist_size, 2, 10);
            else
                MSER_descriptor(img, descr_color, descr_lbp, true, true, color_hist_size, lbp_hist_size, 2, 10);
            
            allColorDescrxImg.push_back(descr_color);
            allColorDescr.push_back(descr_color);

            allLBPDescrxImg.push_back(descr_lbp);
            allLBPDescr.push_back(descr_lbp);

            /*MODIFIED*/
            if (flag_rgb_descr) {
                allRGBDescrxImg.push_back(descr_rgb);
                allRGBDescr.push_back(descr_rgb);
            }
        }
    }
    n_clusters++;
}

void BoW::kmeans() {
    int flags = cv::KMEANS_PP_CENTERS;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, n_iter, epsilon);
    
    if (flag_sift_descriptor)
        cv::kmeans(allDescr, n_clusters, Klabels, criteria, attempts, flags, Kcentroids);

    if (flag_color_descriptor && flag_mser_lbp_descriptor) {
        cv::kmeans(allLBPDescr, n_clusters, LBPlabels, criteria, attempts, flags, LBPcentroids);

        cv::kmeans(allColorDescr, n_clusters, COLlabels, criteria, attempts, flags, COLcentroids);

        /*MODIFIED*/
        if (flag_rgb_descr)
            cv::kmeans(allRGBDescr, n_clusters, RGBlabels, criteria, attempts, flags, RGBcentroids);
    }
}

void BoW::setCriteria(int num_iter, int epsi) {
    n_iter = num_iter;
    epsilon = epsi;
}

cv::Mat BoW::getDataVector(cv::Mat descriptors, const char kmeans_type) {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    switch (kmeans_type) {
    case 's':
        matcher.match(descriptors, Kcentroids, matches);
        //std::cout << " get_data_vector sift " << matches.size() << std::endl;
        break;
    case 'l':
        matcher.match(descriptors, LBPcentroids, matches);
        //std::cout << " get_data_vector lbp " << matches.size() << std::endl;
        break;
    case 'c':
        matcher.match(descriptors, COLcentroids, matches);
        //std::cout << " get_data_vector lbp " << matches.size() << std::endl;
        break;
    /*MODIFIED*/
    case 'r':
        matcher.match(descriptors, RGBcentroids, matches);
        //std::cout << " get_data_vector lbp " << matches.size() << std::endl;
        break;
    }
    //Make a Histogram of visual words
    cv::Mat datai = cv::Mat::zeros(1, dict_size, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++) {
        datai.at<float>(0, matches.at(index).trainIdx) = datai.at<float>(0, matches.at(index).trainIdx) + 1;
    }
    cv::normalize(datai, datai, 0, 1, cv::NORM_MINMAX);
    //std::cout << " datai size: " << datai.size() << " type: " << kmeans_type << std::endl;
    CV_Assert(datai.cols == dict_size);
    return datai;
}

cv::Mat BoW::concatenate_descriptors(int descriptor_index) {
    cv::Mat sift_descriptor;
    cv::Mat color_descriptor;
    cv::Mat lbp_descriptor;
    /*MODIFIED*/
    cv::Mat rgb_descriptor;

    //add the descriptors that we want or just test each one of them and see their performance 
    //concatenate every descriptor
    cv::Mat concatenated_descriptor;
    if (flag_sift_descriptor)
        sift_descriptor = getDataVector(allDescrxImg[descriptor_index]);
    if (flag_color_descriptor)
        color_descriptor = getDataVector(allColorDescrxImg[descriptor_index],'c');
    if (flag_mser_lbp_descriptor)
        lbp_descriptor = getDataVector(allLBPDescrxImg[descriptor_index],'l');
    /*MODIFIED*/
    if (flag_rgb_descr)
        rgb_descriptor = getDataVector(allRGBDescrxImg[descriptor_index], 'r');

    if (flag_sift_descriptor && flag_color_descriptor && flag_mser_lbp_descriptor) {
        CV_Assert(sift_descriptor.rows == color_descriptor.rows);
        CV_Assert(sift_descriptor.rows == lbp_descriptor.rows);
        cv::hconcat(sift_descriptor, color_descriptor, concatenated_descriptor);
        cv::hconcat(concatenated_descriptor, lbp_descriptor, concatenated_descriptor);
    }
    else if (flag_color_descriptor && flag_mser_lbp_descriptor) {
        cv::hconcat(color_descriptor, lbp_descriptor, concatenated_descriptor);
        /*MODIFIED*/
        if (flag_rgb_descr)
            cv::hconcat(concatenated_descriptor, rgb_descriptor, concatenated_descriptor);
    }
    else if (flag_sift_descriptor && flag_color_descriptor)
        cv::hconcat(sift_descriptor, color_descriptor, concatenated_descriptor);
    else if (flag_sift_descriptor && flag_mser_lbp_descriptor)
        cv::hconcat(sift_descriptor, lbp_descriptor, concatenated_descriptor);
    else if (flag_color_descriptor && flag_mser_lbp_descriptor)
        cv::hconcat(color_descriptor, lbp_descriptor, concatenated_descriptor);
    else {
        if (flag_sift_descriptor)
            return sift_descriptor;
        if (flag_color_descriptor)
            return color_descriptor;
        if (flag_mser_lbp_descriptor)
            return lbp_descriptor;
    }
    //std::cout << " HCONCAT ---- descr size" << concatenated_descriptor.size() << std::endl;
    return concatenated_descriptor;
}

cv::Mat BoW::concatenate_descriptors(cv::Mat img) {
    cv::Mat sift_descriptor;
    cv::Mat color_descriptor;
    cv::Mat elbp_descriptor;
    /*MODIFIED*/
    cv::Mat rgb_descriptor;

    cv::Mat concatenated_descriptor;

    if (flag_sift_descriptor) {
        std::vector<cv::KeyPoint> kpts;
        cv::Mat descrSIFT = SIFTdescriptor(img, kpts);
        sift_descriptor = getDataVector(descrSIFT);
        if (sift_descriptor.empty())
            return sift_descriptor;
    }
    if (flag_color_descriptor && flag_mser_lbp_descriptor) {
        cv::Mat descr_col, descr_lbp, descr_rgb;
        /*MODIFIED*/
        if (flag_rgb_descr)
            MSER_descriptor(img, descr_col, descr_lbp, descr_rgb, true, true, true, color_hist_size, lbp_hist_size, rgb_hist_size, 2, 10);
        else
            MSER_descriptor(img, descr_col, descr_lbp, true, true, color_hist_size, lbp_hist_size, 2, 10);
        
        //std::cout << " LBP in concatenate_descriptors: " << descr_lbp.size() << std::endl;
        //std::cout << " COLOR in concatenate_descriptors: " << descr_col.size() << std::endl;
        
        color_descriptor = getDataVector(descr_col, 'c');
        elbp_descriptor = getDataVector(descr_lbp, 'l');
        /*MODIFED*/
        if (flag_rgb_descr) {
            //std::cout << " RGB in concatenate_descriptors: " << descr_rgb.size() << std::endl;
            rgb_descriptor = getDataVector(descr_rgb, 'r');
        }
    }
    if (flag_sift_descriptor && flag_color_descriptor && flag_mser_lbp_descriptor) {
        CV_Assert(sift_descriptor.rows == color_descriptor.rows);
        CV_Assert(sift_descriptor.rows == elbp_descriptor.rows);
        cv::hconcat(sift_descriptor, color_descriptor, concatenated_descriptor);
        cv::hconcat(concatenated_descriptor, elbp_descriptor, concatenated_descriptor);
    }
    else if (flag_color_descriptor && flag_mser_lbp_descriptor) {
        cv::hconcat(color_descriptor, elbp_descriptor, concatenated_descriptor);
    }
    else if (flag_sift_descriptor && flag_color_descriptor)
        cv::hconcat(sift_descriptor, color_descriptor, concatenated_descriptor);
    else if (flag_sift_descriptor && flag_mser_lbp_descriptor)
        cv::hconcat(sift_descriptor, elbp_descriptor, concatenated_descriptor);
    else if (flag_color_descriptor && flag_mser_lbp_descriptor)
        cv::hconcat(color_descriptor, elbp_descriptor, concatenated_descriptor);
    else {
        if (flag_sift_descriptor)
            return sift_descriptor;
        if (flag_color_descriptor)
            return color_descriptor;
        if (flag_mser_lbp_descriptor)
            return elbp_descriptor;
    }
    /*MODIFIED*/
    if (flag_rgb_descr)
        cv::hconcat(concatenated_descriptor, rgb_descriptor, concatenated_descriptor);

    //std::cout << "concatenated descriptor size: " << concatenated_descriptor.size() << std::endl;

    return concatenated_descriptor;
}

void BoW::getHistogram() {
    for (int i = 0; i < n_samples; i++) {
        cv::Mat data_vect = concatenate_descriptors(i);
        inputData.push_back(data_vect);
        inputDataLables.push_back(cv::Mat(1, 1, CV_32SC1, allLabels[i]));
    }
}

void BoW::SVMtrain() {
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC); // C_SVC default SVM type: allows imperfect separation of classes with multipliers C.
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1e3, 0.01));
    svm->setKernel(cv::ml::SVM::RBF); // Radial Basis Function kernel

    svm->trainAuto(inputData, cv::ml::ROW_SAMPLE, inputDataLables, 10); //second parameter: layout
    /* trainAuto automatically compute the optimal parameters for C,nu,gamma... */
}

int BoW::SVMpredict(cv::Mat img) {

    img = img_preprocessing(img, images_resize);

    cv::Mat dvector = concatenate_descriptors(img);

    if (dvector.empty())
        return -1;
    
    return svm->predict(dvector);
}

int BoW::SVMpredict(cv::Mat img, float &out) {  //overload of the previous method, it returns also the confidence

    img = img_preprocessing(img, images_resize);

    cv::Mat dvector = concatenate_descriptors(img);

    if (dvector.empty())
        return -1;

    out = svm->predict(dvector, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);

    return svm->predict(dvector);
}

void BoW::BOWtrain() {
    std::cout << "kmeans\t----\t" ;
    kmeans();
    std::cout << "getHistogram\t----\t";
    getHistogram();
    std::cout << "SVM train\t----\t" << std::endl;
    SVMtrain();
    std::cout << "train completed!" << std::endl;
}

std::string BoW::label2class(int label) {
    if (label == -1)
        return "no food";
    std::string string_label = classes[label];
    return string_label;
}

void BoW::train_classifier(std::map<int, std::string> food_paths, std::map<int, int> map_num_train, std::map<int, int> map_start_index) {
    
    int num_classes_to_train = food_paths.size();
    CV_Assert(num_classes_to_train == map_num_train.size());
    CV_Assert(num_classes_to_train == map_start_index.size());
    CV_Assert(num_classes_to_train == classes.size());
    CV_Assert(num_classes_to_train == dict_size);

    for (int i = 0; i < num_classes_to_train; i++) {
        compute_descriptors(i, food_paths[i], "", ".png", map_num_train[i], map_start_index[i]);
        //std::cout << "\nclasse #: " << i << " #samples_to_train: " << map_num_train[i] << " start index: " << map_start_index[i] << std::endl;
    }

    std::cout << "#clusters: " << n_clusters << " #samples: " << n_samples << std::endl;

    BOWtrain();
}

int BoW::test_classifier(const cv::Mat& img, std::string& string_ans, bool plot, std::string img_name) {
    if (img.empty()) {
        std::cout << "no immagine" << std::endl;
        return 0;
    }
    cv::Mat src = img_preprocessing(img.clone(), 400);
    
    int ans = SVMpredict(src);
    
    if (ans == -1) {
        std::cout << " no matches, image not in the training classes " << std::endl;
        return -1;
    }
    string_ans = label2class(ans);

    if (plot) {
        putText(src, string_ans, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::FILLED);
        cv::imshow(img_name, src);
    }
    return ans;
}

void BoW::test_classifier(const cv::Mat& img, int &label, std::string& string_ans, float &conf, bool plot, std::string img_name) {
    if (img.empty()) {
        std::cout << "no image" << std::endl;
        return;
    }
    cv::Mat src = img_preprocessing(img.clone(), 400);

    float decision;
    label = SVMpredict(src, decision);
    conf = 1.0 / (1.0 + exp(-decision));

    if (label == -1) {
        std::cout << " no matches, image not in the training classes " << std::endl;
        return;
    }
    string_ans = label2class(label);

    if (plot) {
        cv::putText(src, string_ans, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::FILLED);
        cv::imshow(img_name, src);
    }
}