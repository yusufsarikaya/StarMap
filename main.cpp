//
//  main.cpp
//  StarMap
//
//  Created by Yusuf Sarıkaya on 28.07.2020.
//  Copyright © 2020 Yusuf Sarıkaya. All rights reserved.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void findMatchingRect(cv::Mat, cv::Mat);

int main(int argc, const char * argv[]) {
    cv::Mat image1, image2, image3;
    image1 = cv::imread("/Users/yusufsarikaya/Desktop/XCODE_PROJ/StarMap/Images/StarMap.png", cv::IMREAD_GRAYSCALE);
    image2 = cv::imread("/Users/yusufsarikaya/Desktop/XCODE_PROJ/StarMap/Images/Small_area.png", cv::IMREAD_GRAYSCALE);
    image3 = cv::imread("/Users/yusufsarikaya/Desktop/XCODE_PROJ/StarMap/Images/Small_area_rotated.png", cv::IMREAD_GRAYSCALE);
    if( !image1.data || !image2.data || !image3.data)
    {
      cerr << " Failed to load images." << endl;
      return -1;
    }

    findMatchingRect(image3, image1);
}


void findMatchingRect(cv::Mat image1, cv::Mat image2){
    //-- Step 1: Apply threshhold to images
    cv::Mat img_object, img_scene;
    cv::adaptiveThreshold(image1, img_object, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 17,1);
    cv::adaptiveThreshold(image2, img_scene, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 17,1);
    //-- Step 2: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 10;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    //-- Step 3: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.8f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( image1, keypoints_object, image2, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    
    Mat H = findHomography( obj, scene, RANSAC );
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    
    perspectiveTransform( obj_corners, scene_corners, H);
    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 0, 255), 2 );
    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 0, 255), 2 );
    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 0, 255), 2 );
    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 0, 255), 2 );
    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches );
    waitKey();
}
