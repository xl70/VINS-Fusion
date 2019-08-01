#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <tf/transform_broadcaster.h>
#include "keyframe.h"
#include "../utility/tic_toc.h"
// #include "pose_graph.h"
// #include "../utility/CameraPoseVisualization.h"
#include "parameters.h"

#include "../ThirdParty/DBoW/DBoW2.h"
//#include "../ThirdParty/DBoW/TemplatedDatabase.h"
//#include "../ThirdParty/DBoW/TemplatedVocabulary.h"
using namespace std;



camodocal::CameraPtr m_camera;
std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
int ROW = 480;
int COL = 752;
int DEBUG_IMAGE = false;



queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
std::mutex m_buf;
std::mutex m_process;
std::mutex m_init;

BriefDatabase db;
BriefVocabulary* voc=NULL;
list<KeyFrame*> keyframelist;
map<int, cv::Mat> image_pool;
int global_index = 0;
Eigen::Vector3d last_t(-100, -100, -100);
Vector3d t_drift;
Matrix3d r_drift;
bool init = false;
ros::Publisher pub_odometry_map;



void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //if (init) return;
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{    
    if (init) 
    {
        static tf::TransformBroadcaster br; 
        tf::Transform transform;
        tf::Quaternion q;
        // body frame
        Vector3d correct_t;
        Quaterniond correct_q;
        m_init.lock();
        correct_t = t_drift;
        correct_q = r_drift;
        m_init.unlock();

        transform.setOrigin(tf::Vector3(correct_t(0),
                                        correct_t(1),
                                        correct_t(2)));
        q.setW(correct_q.w());
        q.setX(correct_q.x());
        q.setY(correct_q.y());
        q.setZ(correct_q.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, pose_msg->header.stamp, "map", "world"));
    }

    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    //if (init) return;
    //ROS_INFO("point_callback!");
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void odometry_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    if (!init)
        return;
    //ROS_INFO("odometry_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = r_drift * vio_t + t_drift;
    vio_q = r_drift * vio_q;

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "map";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    pub_odometry_map.publish(odometry);
}

void loadPoseGraph()
{
    TicToc tmp_t;
    FILE * pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("lode pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    pFile = fopen (file_path.c_str(),"r");
    if (pFile == NULL)
    {
        printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    int window_keypoints_num;
    Eigen::Matrix<double, 8, 1 > loop_info;
    while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d %d", 
                                    &index, &time_stamp, 
                                    &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                                    &PG_Tx, &PG_Ty, &PG_Tz, 
                                    &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                                    &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                                    &loop_index,
                                    &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                                    &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                                    &keypoints_num, &window_keypoints_num) != EOF) 
    {
        cv::Mat image;
        std::string image_path, descriptor_path;
       
        image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
        image = cv::imread(image_path.c_str(), 0);

        Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
        Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1 > loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        // load keypoints, brief_descriptors   
        string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        std::ifstream brief_file(brief_path, std::ios::binary);
        string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "r");
        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<BRIEF::bitset> brief_descriptors;
        for (int i = 0; i < keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            brief_file >> tmp_des;
            brief_descriptors.push_back(tmp_des);
            cv::KeyPoint tmp_keypoint;
            cv::KeyPoint tmp_keypoint_norm;
            double p_x, p_y, p_x_norm, p_y_norm;
            if(!fscanf(keypoints_file,"%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
                printf(" fail to load pose graph \n");
            tmp_keypoint.pt.x = p_x;
            tmp_keypoint.pt.y = p_y;
            tmp_keypoint_norm.pt.x = p_x_norm;
            tmp_keypoint_norm.pt.y = p_y_norm;
            keypoints.push_back(tmp_keypoint);
            keypoints_norm.push_back(tmp_keypoint_norm);
        }
        brief_file.close();
        fclose(keypoints_file);
        
        // load point_uv
        string window_brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_window_briefdes.dat";
        std::ifstream window_brief_file(window_brief_path, std::ios::binary);
        string window_keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_window_keypoints.txt";
        FILE *window_keypoints_file;
        window_keypoints_file = fopen(window_keypoints_path.c_str(), "r");
        vector<cv::Point3f> point_3d; 
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_norm;
        vector<BRIEF::bitset> window_brief_descriptors;
        for (int i = 0; i < window_keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            window_brief_file >> tmp_des;
            window_brief_descriptors.push_back(tmp_des);
            cv::Point3f tmp_point_3d;
            cv::Point2f tmp_point_2d_uv;
            cv::Point2f tmp_point_2d_norm;
            double p_3d_x, p_3d_y, p_3d_z, p_x_uv, p_y_uv, p_x_norm, p_y_norm;
            if(!fscanf(window_keypoints_file,"%lf %lf %lf %lf %lf %lf %lf", &p_3d_x, &p_3d_y, &p_3d_z, &p_x_uv, &p_y_uv, &p_x_norm, &p_y_norm))
                printf(" fail to load pose graph \n");
            tmp_point_3d.x = p_3d_x;
            tmp_point_3d.y = p_3d_y;
            tmp_point_3d.z = p_3d_z;
            tmp_point_2d_uv.x = p_x_uv;
            tmp_point_2d_uv.y = p_y_uv;
            tmp_point_2d_norm.x = p_x_norm;
            tmp_point_2d_norm.y = p_y_norm;
            point_3d.push_back(tmp_point_3d);
            point_2d_uv.push_back(tmp_point_2d_uv);
            point_2d_norm.push_back(tmp_point_2d_norm);
        }
        window_brief_file.close();
        fclose(window_keypoints_file);

        KeyFrame* keyframe = new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors,
            point_3d, point_2d_uv, point_2d_norm, window_brief_descriptors);
        cout << keyframe->point_3d.size() << "," 
            << keyframe->point_2d_uv.size() << "," 
            << keyframe->point_2d_norm.size() << ","
            << keyframe->window_brief_descriptors.size() << ","
            << keyframe->keypoints.size() << ","
            << keyframe->keypoints_norm.size() << ","
            << keyframe->brief_descriptors.size() << endl;
        keyframelist.push_back(keyframe);
        db.add(keyframe->brief_descriptors);
        global_index = index + 1;
        
        //debug image
        if (DEBUG_IMAGE)
        {
            cv::Mat compressed_image;
            int feature_num = keyframe->keypoints.size();
            cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
            cv::putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
            image_pool[keyframe->index] = compressed_image;
        }
    }
    fclose (pFile);
    printf("load pose graph time: %f s\n", tmp_t.toc()/1000);
    cout << "keyframelist size:" << keyframelist.size() << endl;
    cout << "db size:" << db.size() << endl;
}

KeyFrame* getKeyFrame(int index)
{
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

bool p_r_RANSAC(vector<Vector3d> &_i_p, vector<Matrix3d> &_i_r, Vector3d &_init_p, Matrix3d &_init_r)
{
    vector<int> max_inliers;
    for (int i = 0; i < _i_p.size(); i++)
    {
        vector<int> inliers;
        for (int j = 0; j < _i_p.size(); j++)
        {
            double dis = (_i_p[i] - _i_p[j]).norm();
            if (dis < 0.5)
            {
                inliers.push_back(j);
            }
        }
        
        if (inliers.size() > max_inliers.size())
        {
            max_inliers = inliers;
        }
    }

    if (max_inliers.size() < 3)
        return false;
    
    _init_p.x() = 0;
    _init_p.y() = 0;
    _init_p.z() = 0;
    Vector3d ypr{0, 0, 0};
    for (int i =0; i < max_inliers.size(); i++)
    {
        cout << _i_p[max_inliers[i]] << endl;
        cout << _i_r[max_inliers[i]] << endl;
        _init_p += _i_p[max_inliers[i]];
        ypr += Utility::R2ypr(_i_r[max_inliers[i]]);
    }
    _init_p = _init_p / max_inliers.size();
    _init_r = Utility::ypr2R(ypr / max_inliers.size());
    return true;
}

void process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;

        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();
        
        if (pose_msg != NULL)
        {           
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();

            if((T - last_t).norm() > 0)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }
                
                KeyFrame* cur_kf = new KeyFrame(pose_msg->header.stamp.toSec(), global_index, T, R, image, point_3d, point_2d_uv, point_2d_normal, point_id);
                global_index++;

                DBoW2::QueryResults ret;
                db.query(cur_kf->brief_descriptors, ret, 100, -1);
                
                if (DEBUG_IMAGE && false)
                {
                    cv::Mat compressed_image;
                    int feature_num = cur_kf->keypoints.size();
                    cv::resize(cur_kf->image, compressed_image, cv::Size(376, 240));
                    cv::putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    
                    cv::Mat loop_result;
                    loop_result = compressed_image.clone();
                    if (ret.size() > 0)
                        cv::putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));

                    for (unsigned int i = 0; i < ret.size(); i++)
                    {
                        int tmp_index = ret[i].Id;
                        auto it = image_pool.find(tmp_index);
                        cv::Mat tmp_image = (it->second).clone();
                        cv::putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
                        cv::hconcat(loop_result, tmp_image, loop_result);
                    }

                    cv::imshow("loop_result", loop_result);
                    cv::waitKey(5);
                }
                
                if (ret.size() >= 1 && ret[0].Score > 0.015)
                {
                    
                    Vector3d init_p{0, 0, 0};
                    Matrix3d init_r;
                    
                    Vector3d vio_P;
                    Matrix3d vio_R;
                    KeyFrame* old_kf;
                    
                    vector<Vector3d> i_p;
                    vector<Matrix3d> i_r;
                    int index = 0;
                    while (ret[index].Score > 0.015 && index < ret.size())
                    {
                        old_kf = getKeyFrame(ret[index].Id);
                        old_kf->getVioPose(vio_P, vio_R);
                        index++;
                        
                        Vector3d relative_t; 
                        Matrix3d relative_r;
                        m_process.lock();
                        if (cur_kf->findConnection(old_kf, relative_t, relative_r))
                        {
                            //cout << vio_R * relative_t + vio_P << endl;
                            //cout << vio_R * relative_r << endl;
                            i_p.push_back(vio_R * relative_t + vio_P);
                            i_r.push_back(vio_R * relative_r);
                        }
                        m_process.unlock();
                    }
                    
                    if (p_r_RANSAC(i_p, i_r, init_p, init_r))
                    {
                        cout << "++++" << endl;
                        cout << init_p << endl;
                        cout << init_r << endl;
                        cout << "====" << endl;
                        
                        //init_r = r_drift * R;
                        //init_p = r_drift * T + t_drift;
                        m_init.lock();
                        r_drift = init_r * R.transpose();
                        t_drift = init_p - r_drift * T;
                        m_init.unlock();
                        init = true;
                    }
                }
            }
        }
        
        
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "init");
    ros::NodeHandle n("~");
    
    if(argc > 2)
    {
        printf("please intput: rosrun init init [config file] \n"
               "for example: rosrun init init"
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }
    
    //加载字典文件
    std::string pkg_path = ros::package::getPath("init");
    string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
    cout << "vocabulary_file:\t" << vocabulary_file << endl;
    voc = new BriefVocabulary(vocabulary_file);
    db.setVocabulary(*voc, false, 0);
    
    //设置描述子yaml文件，keyframe中会用到
    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE:\t" << BRIEF_PATTERN_FILE << endl;
    
    if (argc == 2)
    {
        string config_file = argv[1];
        printf("config_file: %s\n", argv[1]);
        cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            std::cerr << "ERROR: Wrong path to settings" << std::endl;
            return 1;
        }
        
        //设置pose_graph保存路径
        fsSettings["init_pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
        cout << "POSE_GRAPH_SAVE_PATH:\t" << POSE_GRAPH_SAVE_PATH << endl;
    
        int pn = config_file.find_last_of('/');
        std::string configPath = config_file.substr(0, pn);
        std::string cam0Calib;
        fsSettings["cam0_calib"] >> cam0Calib;
        std::string cam0Path = configPath + "/" + cam0Calib;
        cout << "camera yaml:\t" << cam0Path << endl;
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());
    }
    else
    {
        //设置pose_graph保存路径
        POSE_GRAPH_SAVE_PATH = pkg_path + "/pose_graph/";
        cout << "POSE_GRAPH_SAVE_PATH:\t" << POSE_GRAPH_SAVE_PATH << endl;
        
        cout << "camera yaml:\t" << "/home/sxsqli/Desktop/test/yaml_new/left_c.yaml" << endl;
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile("/home/sxsqli/Desktop/test/yaml_new/left_c.yaml");
    }
    
    m_process.lock();
    loadPoseGraph();
    m_process.unlock();
    
    ros::Subscriber sub_image = n.subscribe("/mynteye/left/image_raw", 2000, image_callback);
    ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 2000, pose_callback);
    ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);
    
    ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
    
    ros::Subscriber sub_odometry = n.subscribe("/loop_fusion/odometry_rect", 2000, odometry_callback);
    
    pub_odometry_map = n.advertise<nav_msgs::Odometry>("odometry_map", 1000);
    
    std::thread measurement_process{process};
    
    ros::spin();
    return 0;
}