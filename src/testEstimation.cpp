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

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;
using namespace g2o;
// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

struct FRAME{
    cv::Mat rgbImage;
    cv::Mat depthImage;
    cv::Mat descriptors;
    vector<cv::KeyPoint> keypoints;
};

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



class VertexPose : public BaseVertex<6,SE3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate=SE3d();
    }
    virtual void oplusImpl(const double *update) override{
        Eigen::Matrix <double , 6 , 1 > update_eigen;
        update_eigen<<update[0],update[1],update[2],update[3],update[4],update[5];
        _estimate = SE3d::exp(update_eigen) * _estimate;
    }
    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override {}
};

class ProjectionEdge : public BaseUnaryEdge<2,Vector2d,VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionEdge(const Vector3d &point3d ,const Matrix3d &K):_point3d(point3d),_K(K){}

    virtual void computeError() override {
        VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        SE3d T = v->estimate();
        Vector3d pose_pixel = _K * ( T * _point3d);
        pose_pixel /= pose_pixel[2];
        _error = _measurement - pose_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        SE3d T = v->estimate();
        Vector3d pos_cam = T * _point3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
                << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
                0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override {}
private:
    Vector3d _point3d;
    Matrix3d _K;
};

void solvePnpG2o(const VecVector3d &points_3d , const VecVector2d &points_2d, Matrix3d &camera , SE3d &pose){
    typedef BlockSolver<BlockSolverTraits<6,3>> BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *v = new VertexPose();
    v->setId(0);
    v->setEstimate(SE3d());
    optimizer.addVertex(v);

    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        ProjectionEdge *edge = new ProjectionEdge(p3d, camera);
        edge->setId(index);
        edge->setVertex(0, v);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    pose = v->estimate();
}

void findDescriptorsAndKeypoints(FRAME &frame){
    assert(frame.depthImage != nullptr && frame.rgbImage != nullptr);
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    detector->detect(frame.rgbImage,frame.keypoints);
    descriptor->compute(frame.rgbImage,frame.keypoints,frame.descriptors);

    return;
}

void matchAndEstimatePnp(FRAME &frame1 , FRAME &frame2 , SE3d &pose , Matrix3d &K) {
    cout<<"firsthere"<<endl;
    Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    cout<<"f1 descriptor is"<<frame1.descriptors<<endl;
    cout<<"f1 descriptor is"<<frame2.descriptors<<endl;

    matcher->match(frame1.descriptors,frame2.descriptors,matches);
    cout<<"flag1"<<endl;
    auto min_max = minmax_element(matches.begin(),matches.end(),
                                  [](const DMatch &m1, const DMatch &m2){return m1.distance < m2.distance;});
    cout<<"flag2"<<endl;
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    cout<<"flag3"<<endl;

    vector<DMatch>good_matches;

    for(int i = 0 ; i< matches.size() ; i++){
        if(matches[i].distance <= max(2 * min_dist ,30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    cout<<"flag4"<<endl;

    VecVector3d points_3d;
    VecVector2d points_2d;


    for(auto m:good_matches){
        ushort d = frame1.depthImage.ptr<ushort>(int(frame1.keypoints[m.queryIdx].pt.y))[int(frame1.keypoints[m.queryIdx].pt.x)];
        if(d == 0)
            continue;
        double fx = K(0,0) ,  cx = K(0,2);
        double fy = K(1,1) ,  cy = K(1,2);
        double x = (frame1.keypoints[m.queryIdx].pt.x - cx)/fx;
        double y = (frame1.keypoints[m.queryIdx].pt.y - cy)/fy;
        points_3d.push_back(Vector3d(x*d,y*d,d));
        points_2d.push_back(Vector2d(frame2.keypoints[m.trainIdx].pt.x,frame2.keypoints[m.trainIdx].pt.y));
    }
    cout<<"flag5"<<endl;

    solvePnpG2o(points_3d,points_2d,K,pose);
    cout<<"flag6"<<endl;
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
    FRAME f1 = readFrame(1,r);
    FRAME f2 = readFrame(2,r);

    findDescriptorsAndKeypoints(f1);
    findDescriptorsAndKeypoints(f2);
    SE3d pose;
    Matrix3d  K;
    K<< 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    cout<<"flag0"<<endl;

    matchAndEstimatePnp(f1,f2,pose,K);


    return 0 ;
}