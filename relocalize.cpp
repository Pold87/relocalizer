// No copyright - Volker

#include <stdio.h>

// Include standard libraries
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>
#include <chrono>

// Include OpenCV libraries

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

// Math libraries
#include <math.h>
#include <cmath>

#include "relocalize.h"

using namespace std;
using namespace cv;
const double pi = 3.14159265358979323846;

const float inlier_threshold = 100.0f; // Distance threshold to identify inliers
const float match_ratio = 0.8f;   // Nearest neighbor matching ratio

const bool draw_matches = true;
const bool draw_lines = false;
const bool use_good_matches = true;
const bool save_img = false;
const bool show_img = false;

Relocalizer::Relocalizer(std::string path) {

  ref_img_path = path;
  ref_img_c = cv::imread(ref_img_path);

  cv::cvtColor(ref_img_c,
               ref_img,
               cv::COLOR_BGR2GRAY);



  int minHessian = 400;
  detector = cv::xfeatures2d::SURF::create(minHessian);


  // Detect and compute keypoints of the reference image
  detector->detectAndCompute(ref_img,
                             cv::noArray(),
                             kp_ref,
                             des_ref);
}


std::vector<float> Relocalizer::calcLocation(cv::Mat query_img, int num) {

  std::vector<float> res(3);
  std::vector<float> resDefault(3);

  resDefault[0] = -1.0;
  resDefault[1] = -1.0;
  resDefault[2] =  0.0;


  std::vector<cv::KeyPoint> kp_query; // Keypoints of the query image
  cv::Mat des_query;
  cv::Mat query_img_gray;

  cv::cvtColor(query_img,
               query_img_gray,
               cv::COLOR_BGR2GRAY);


  detector->detectAndCompute(query_img_gray,
                             cv::noArray(),
                             kp_query,
                             des_query);


  std::vector<cv::DMatch> matches;

  if(des_query.rows > 0){
    matcher.match(des_query, des_ref,
                  matches);

    std::vector<cv::KeyPoint> matched_query, matched_ref, inliers_query, inliers_ref;
    std::vector<cv::DMatch> good_matches;

    //-- Localize the object
    std::vector<cv::Point2f> pts_query;
    std::vector<cv::Point2f> pts_ref;

    for(cv::DMatch currentMatch : matches) {
      cv::DMatch first = currentMatch;

      matched_query.push_back(kp_query[first.queryIdx]);
      matched_ref.push_back(kp_ref[first.trainIdx]);

      pts_query.push_back(kp_query[first.queryIdx].pt);
      pts_ref.push_back(kp_ref[first.trainIdx].pt);

    }

    cv::Mat mask;
    if(matched_query.size() > 0){

      // Homography
      cv::Mat homography;

      homography = cv::findHomography(pts_query,
                                      pts_ref,
                                      cv::RANSAC,
                                      5,
                                      mask);


      // Input Quadilateral or Image plane coordinates
      std::vector<cv::Point2f> centers(1), centers_transformed(1);

      cv::Point2f center(query_img_gray.cols / 2,
                         query_img_gray.rows / 2);

      cv::Point2f center_transformed(query_img.cols / 2,
                                     query_img.rows / 2);

      centers[0] = center; // Workaround for using perspective transform

      if (homography.cols > 0) {

        cv::perspectiveTransform(centers,
                                 centers_transformed,
                                 homography);

        center_transformed = centers_transformed[0];

        res[0] = center_transformed.x;
        res[1] = center_transformed.y;
        res[2] = matches.size();

        /* Good matches have to fulfill certain criteria */
        if (use_good_matches) {

           double max_dist = 0; double min_dist = 100;

           //-- Quick calculation of max and min distances between keypoints
           for( int i = 0; i < des_query.rows; i++ )
           { double dist = matches[i].distance;
              if( dist < min_dist ) min_dist = dist;
              if( dist > max_dist ) max_dist = dist;
           }


           //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
           std::vector< DMatch > good_matches;

           for( int i = 0; i < des_query.rows; i++ ){
              if( matches[i].distance < 3 * min_dist ){
                 good_matches.push_back( matches[i]);
              }
           }

           //-- Localize the object
           std::vector<Point2f> obj;
           std::vector<Point2f> scene;

           for( int i = 0; i < good_matches.size(); i++ )
           {
              //-- Get the keypoints from the good matches
              obj.push_back( kp_query[ good_matches[i].queryIdx ].pt );
              scene.push_back( kp_ref[ good_matches[i].trainIdx ].pt );
           }


           /* Find homography matrix for good matches */
           Mat H = findHomography( obj, scene, CV_RANSAC );

           /* Transformation of image center */
           std::vector<cv::Point2f> centers_transformed_good(1);
           cv::Point2f center_transformed_good(query_img.cols / 2,
                                          query_img.rows / 2);

           cv::perspectiveTransform(centers, centers_transformed_good, H);

           center_transformed_good = centers_transformed_good[0];

           /* Overwrite results */
           res[0] = center_transformed_good.x;
           res[1] = center_transformed_good.y;
           res[2] = good_matches.size();

           if (draw_matches) {


              Mat img_matches;

              Size sz1 = query_img.size();
              Size sz2 = ref_img.size();
              Mat img_matches_no_lines(sz1.height, sz1.width + sz2.width, CV_8UC3);

              if (draw_lines) {

                 drawMatches( query_img, kp_query, ref_img_c, kp_ref,
                              good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                              vector<char>(), DrawMatchesFlags::DEFAULT );

              } else {

                 Mat left(img_matches_no_lines, Rect(0, 0, sz1.width, sz1.height));
                 query_img.copyTo(left);

                 Mat right(img_matches_no_lines, Rect(sz1.width, 0, sz2.width, sz2.height));
                 ref_img_c.copyTo(right);
              }

              //-- Get the corners from the image_1 ( the object to be "detected" )
              std::vector<Point2f> obj_corners(4);
              obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( query_img_gray.cols, 0 );
              obj_corners[2] = cvPoint( query_img_gray.cols, query_img_gray.rows );
              obj_corners[3] = cvPoint( 0, query_img_gray.rows );
              std::vector<Point2f> scene_corners(4);

              /* Transformation of corners */
              perspectiveTransform( obj_corners, scene_corners, H);

              //-- Draw lines between the corners (the mapped object in the scene - image_2 )
              if (draw_lines) {
                 line( img_matches, scene_corners[0] + Point2f( query_img_gray.cols, 0), scene_corners[1] + Point2f( query_img_gray.cols, 0), Scalar(0, 255, 0), 8 );
                 line( img_matches, scene_corners[1] + Point2f( query_img_gray.cols, 0), scene_corners[2] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
                 line( img_matches, scene_corners[2] + Point2f( query_img_gray.cols, 0), scene_corners[3] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
                 line( img_matches, scene_corners[3] + Point2f( query_img_gray.cols, 0), scene_corners[0] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
              } else {
                 line( img_matches_no_lines, scene_corners[0] + Point2f( query_img_gray.cols, 0), scene_corners[1] + Point2f( query_img_gray.cols, 0), Scalar(0, 255, 0), 8 );
                 line( img_matches_no_lines, scene_corners[1] + Point2f( query_img_gray.cols, 0), scene_corners[2] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
                 line( img_matches_no_lines, scene_corners[2] + Point2f( query_img_gray.cols, 0), scene_corners[3] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
                 line( img_matches_no_lines, scene_corners[3] + Point2f( query_img_gray.cols, 0), scene_corners[0] + Point2f( query_img_gray.cols, 0), Scalar( 0, 255, 0), 8 );
              }

              // Draw centers
              int thickness = 8;
              int lineType = 8;

              if (draw_lines) {

            cv::circle( img_matches,
                      center,
                      12.0,
                      Scalar( 0, 0, 255 ),
                      thickness - 3,
                      lineType );

              cv::circle( img_matches,
                      center_transformed_good + Point2f( query_img_gray.cols, 0),
                      18.0,
                      Scalar( 0, 0, 255 ),
                      thickness,
                      lineType );

              } else {
            cv::circle( img_matches_no_lines,
                      center,
                      18.0,
                      Scalar( 0, 0, 255 ),
                      thickness,
                      lineType );

              cv::circle( img_matches_no_lines,
                      center_transformed_good + Point2f( query_img_gray.cols, 0),
                      12.0,
                      Scalar( 0, 0, 255 ),
                      thickness - 3,
                      lineType );
              }


              //-- Show detected matches
              if (draw_lines) {
                 if (show_img) {
                    imshow( "Good Matches & Object detection", img_matches );
                    waitKey(0);
                 }
                 if (save_img) {
                    char img_name[256];
                    sprintf(img_name, "recordings/img_%05d.png", num);
                    imwrite(img_name, img_matches);
                 }
              } else {
                 if (show_img) {
                    imshow( "Good Matches & Object detection", img_matches_no_lines );
                    waitKey(0);
                 }
                 if (save_img) {
                    char img_name[256];
                    sprintf(img_name, "recordings/img_%05d.png", num);
                    imwrite(img_name, img_matches_no_lines);
                 }
              }

           }
    }
        return res;
      }
      else {
        return resDefault;
      }
    }
    else {
      return resDefault;
    }
  }
  else {
    return resDefault;
  }
}
