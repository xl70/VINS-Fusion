//
// Created by chewgum on 19-8-6.
//

#include "my_test.h"

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "feature_tracker.h"
#include <iostream>
#include <ros/package.h>




using namespace std;

FeatureTracker featureTracker;
camodocal::CameraPtr m_camera;
std::vector<std::string> CAM_NAMES;


int main(int argc, char **argv)
{

    ros::init(argc, argv, "my_test");
    ros::NodeHandle n("~");

//    std::thread my_test_process{feature_process};
//    ros::spin();

    string config_file = ros::package::getPath("launch_pkg") +  "/camera_config/mynteye_stereo_imu_config.yaml";
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        cout<<"config_file dosen't exist; wrong config_file path"<<endl;
        ROS_BREAK();
        return 0;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);


    std::string cam1Calib;
    fsSettings["cam1_calib"] >> cam1Calib;
    std::string cam1Path = configPath + "/" + cam1Calib;
    CAM_NAMES.push_back(cam1Path);

 //   string cameraFile = ros::package::getPath("launch_pkg")  + "/camera_config/left_mynt_eye.yaml";

    featureTracker.readIntrinsicParameter(CAM_NAMES);


    //ros::Time t = ros::Time::now();
    cv::Mat image;
    double  t = 1563844522.0;
//    std::string image_dir = "../pic/";
//    string  image_path = image_dir  + "0_image.png";

    string  image_path = ros::package::getPath("my_test") + "/pic/0_image.png";
    image = cv::imread(image_path.c_str(), 0);


    cout<<"test"<<endl;
    featureTracker.trackImage(t,image);
    cv::Mat img= featureTracker.getTrackImage();
    cv::imshow("feature img",img);
    cv::waitKey(0);
    ros::spin();
    return 0;
}
