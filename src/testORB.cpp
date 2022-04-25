//
// Created by zlj on 22-4-24.
//

//checked


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
void findDescriptorsAndKeypoints(FRAME &frame){
    assert(frame.depthImage != nullptr && frame.rgbImage != nullptr);
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    detector->detect(frame.rgbImage,frame.keypoints);
    descriptor->compute(frame.rgbImage,frame.keypoints,frame.descriptors);

    return;
}

FRAME readFrame(int index , Reader reader){
    FRAME f;
    string rgbDir   =   reader.getData("rgb_dir");
    string depthDir =   reader.getData("depth_dir");

    string rgbExt   =   reader.getData("rgb_extension");
    string depthExt =   reader.getData("depth_extension");

    string filename1 = rgbDir+to_string(index)+rgbExt;
    cout<<"filename1 = " <<filename1<<endl;
    f.rgbImage = cv::imread( filename1 );


    string filename2 = depthDir+to_string(index)+depthExt;
    cout<<"filename2 = " <<filename2<<endl;
    f.depthImage = cv::imread( filename2, -1 );
    return f;
}




int main(){
    Reader r;
    FRAME f = readFrame(1,r);

    findDescriptorsAndKeypoints(f);

    cout<<f.descriptors;
    for(int i = 0 ; i < f.keypoints.size() ; i ++){
        cout<<f.keypoints[i].pt;
    }

    return 0;
}