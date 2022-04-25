# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
FIND_PACKAGE( OpenCV 3 REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

#add g2o
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIRS})


# INCLUDE_DIRECTORIES( ${PROJECT_SOURSE_DIR}/include )
ADD_LIBRARY( slambase slamBase.cpp )
ADD_EXECUTABLE( RGBD_VO RGBD_VO.cpp)
TARGET_LINK_LIBRARIES( RGBD_VO
        g2o_core g2o_stuff
        slambase
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES} )

ADD_EXECUTABLE( testR testR.cpp)
TARGET_LINK_LIBRARIES( testR
        slambase
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})

ADD_EXECUTABLE( testORB testORB.cpp)
TARGET_LINK_LIBRARIES( testORB
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})

ADD_EXECUTABLE( testCloud testCloud.cpp)
TARGET_LINK_LIBRARIES( testCloud
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})

ADD_EXECUTABLE( testPnpG2o testPnpG2o.cpp)
TARGET_LINK_LIBRARIES( testPnpG2o
        g2o_core g2o_stuff
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})

ADD_EXECUTABLE( testEstimation testEstimation.cpp)
TARGET_LINK_LIBRARIES( testEstimation
        g2o_core g2o_stuff
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})