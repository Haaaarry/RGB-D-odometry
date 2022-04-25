//
// Created by zlj on 22-4-20.
//
#include "slamBase.h"
using namespace cv;

int main(){
    Reader reader;
    int startIndex  =   atoi( reader.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( reader.getData( "end_index").c_str() );

    cout<<"startIndex= " <<startIndex<<" ,  endIndex = "<<endIndex<<endl;



    // initialize
    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex;
    FRAME lastFrame = readFrame( currIndex, reader );
    Camera camera ;
    Eigen::Matrix3d eigen_K ;
    eigen_K << camera.K.at<double>(0,0) , camera.K.at<double>(0,1) , camera.K.at<double>(0,2),
               camera.K.at<double>(1,0) , camera.K.at<double>(1,1) , camera.K.at<double>(1,2),
               camera.K.at<double>(2,0) , camera.K.at<double>(2,1) , camera.K.at<double>(2,2);
    cout<<"camera K is"<<eigen_K<<endl;

    findDescriptorsAndKeypoints(lastFrame);
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgbImage, lastFrame.rgbImage, camera );

    pcl::visualization::CloudViewer viewer("viewer");

    bool visualize = reader.getData("visualize_pointcloud")==string("yes");

    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,reader ); // 读取currFrame
        findDescriptorsAndKeypoints( currFrame);
        SE3d T;
        matchAndEstimatePnp(lastFrame,currFrame,T,eigen_K);

        Eigen::Isometry3d E_T(T.matrix());
        cloud = joinPointCloud( cloud, currFrame, E_T, camera );
        if ( visualize == true ){
            viewer.showCloud( cloud );
            cout<<"visualize is true"<<endl;
        }

        cout<<"here"<<endl;
        lastFrame = currFrame;
    }


    pcl::io::savePCDFile( "/home/zlj/slam project/RGBD-VO/data/result.pcd", *cloud );
    return 0;
}

