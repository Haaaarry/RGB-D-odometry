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

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, Camera& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    double fx = camera.K.at<double>(0,0) ,  cx = camera.K.at<double>(0,2);
    double fy = camera.K.at<double>(1,1) ,  cy = camera.K.at<double>(1,2);
    for (int m = 0; m < depth.rows; m+=2)
        for (int n=0; n < depth.cols; n+=2)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - cx) * p.z / fx;
            p.y = (m - cy) * p.z / fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

int main(){
    Camera c;
    cv::Mat rgb = imread("/home/zlj/slam project/RGBD-VO/data/rgb_png/1.png");
    cv::Mat depth = imread("/home/zlj/slam project/RGBD-VO/data/depth_png/1.png",-1);

    PointCloud::Ptr cloud = image2PointCloud(rgb,depth,c);
    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( cloud );
    while( !viewer.wasStopped() )
    {

    }

    return 0;
}