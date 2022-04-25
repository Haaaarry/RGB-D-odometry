//
// Created by zlj on 22-4-20.
//
// Eigen
# pragma once

// 各种头文件
// C++标准库
#include <fstream>
#include <vector>
#include <map>
using namespace std;

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

//Sophus
#include <sophus/se3.hpp>

//g2o
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace cv;
using namespace Eigen;
using namespace Sophus;

// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

class Reader{
public:
    Reader(string file="/home/zlj/slam project/RGBD-VO/parameters.txt"){
        ifstream fin(file);
        if(!fin){
            cout<<"parameter file not exist!"<<endl;
        }
        while(!fin.eof()){
            string str;
            getline(fin,str);

            if(str[0] == '#')
                continue;
            int pos = str.find('=');

            if(pos == -1)
                continue;
            mp[str.substr(0,pos)] = str.substr(pos+1,str.length());
//            if(fin.good()) {
//                cout<<"getout"<<endl;
//                break;
//            }
        }
    }
    string getData(string query){
        map<string,string>::iterator iter = mp.find(query);
        if(iter == mp.end()){
            cerr<<"parameter "<<query<<" not found"<<endl;
            return string("NOT FOUND");
        }
        return iter->second;
    }


private:
    map<string,string>mp;
};

struct FRAME{
    cv::Mat rgbImage;
    cv::Mat depthImage;
    cv::Mat descriptors;
    vector<cv::KeyPoint> keypoints;
};

class Camera{
public:
    Camera(){
        Reader reader;
        double fx = atof( reader.getData( "fx" ).c_str());
        double fy = atof( reader.getData( "fy" ).c_str());
        double cx = atof( reader.getData( "cx" ).c_str());
        double cy = atof( reader.getData( "cy" ).c_str());
        double scale = atof(reader.getData( "scale" ).c_str() );
        K = (Mat_<double>(3,3)<<fx , 0 , cx ,
                                0  , fy, cy,
                                0  , 0 , 1  );
        this->scale = scale;
    }
public:
    cv::Mat K;
    double scale;
};

void findDescriptorsAndKeypoints(FRAME &frame1);

void matchAndEstimatePnp(FRAME &frame1 , FRAME &frame2 ,SE3d &pose, Matrix3d &K);

void solvePnpG2o(const VecVector3d &points_3d , const VecVector2d &points_2d, Matrix3d &camera , SE3d &pose);

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, Camera& camera );

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T,Camera& camera ) ;

FRAME readFrame(int index , Reader reader);