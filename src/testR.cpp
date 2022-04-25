//
// Created by zlj on 22-4-24.
//


//checked

#include "slamBase.h"
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

    Reader R;
    FRAME img = readFrame(1,R);
    cv::imshow("reader",img.rgbImage);
    cv::imshow("readerDepth",img.depthImage);

    cv::Mat im = cv::imread("/home/zlj/slam project/RGBD-VO/data/rgb_png/1.png",1);
    cv::imshow("imread",im);
    cv::Mat im_d = cv::imread("/home/zlj/slam project/RGBD-VO/data/depth_png/1.png",1);
    cv::imshow("imread_depth",im_d);
    cv::waitKey(0);

    return 0 ;
}