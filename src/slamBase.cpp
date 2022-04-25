//
// Created by zlj on 22-4-20.
//

#include "slamBase.h"
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace g2o;
using namespace Sophus;

void findDescriptorsAndKeypoints(FRAME &frame){
    assert(frame.depthImage != nullptr && frame.rgbImage != nullptr);
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    detector->detect(frame.rgbImage,frame.keypoints);
    descriptor->compute(frame.rgbImage,frame.keypoints,frame.descriptors);

    return;
}

void matchAndEstimatePnp(FRAME &frame1 , FRAME &frame2 , SE3d &pose , Matrix3d &K) {

    Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(frame1.descriptors,frame2.descriptors,matches);

    auto min_max = minmax_element(matches.begin(),matches.end(),
                                  [](const DMatch &m1, const DMatch &m2){return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    vector<DMatch>good_matches;

    for(int i = 0 ; i< matches.size() ; i++){
        if(matches[i].distance <= max(2 * min_dist ,30.0)){
            good_matches.push_back(matches[i]);
        }
    }

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

    solvePnpG2o(points_3d,points_2d,K,pose);

}

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

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, Camera & camera ) {
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgbImage, newFrame.rgbImage, camera );

    // 合并点云
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    static Reader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
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