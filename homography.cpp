#include "opencv2/opencv_modules.hpp"
 #include "opencv2/core/core.hpp"
 #include "opencv2/nonfree/nonfree.hpp"
#include <stdio.h>

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
// #include "opencv2/flann/flann_base.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

 
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
void readme();

/**
 * @function main
 * @brief Main function
 */
 Mat ransac(vector<Point2f> p1, vector<Point2f> p2, int error = 20) {
  srand(time(NULL));
  vector<int> idxs(p1.size(), 0);
  for (int i = 0; i < idxs.size(); i++) {
    idxs[i] = i;
  }
  while (1) {
    // get 4 random point pairs
    std::random_shuffle(idxs.begin(), idxs.end());
    vector<Point2f> src, dst;
    for (int i = 0; i < 4; i++) {
      src.push_back(p1[idxs[i]]);
      dst.push_back(p2[idxs[i]]);
    }
    // compute projective transform
    Mat H = getPerspectiveTransform(src, dst);
    vector<Point2f> p2_d;
    perspectiveTransform(p1, p2_d, H);
    // count num of inliers
    int count = 0;
    for (int i = 0; i < p2_d.size(); i++) {
      if (norm(p2_d[i] - p2[i]) <= error)
        count++;
    }
    // if count is high enough, return answer
    if (count >= p1.size()/5) {
      // should do least squaeres fitting using all the inliers...
      // just returning H for now.
      return H;
    }
  }
}
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { printf(" --(!) Error reading images \n"); return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

 std::vector< Point2f > obj;
 std::vector< Point2f > scene;

  for( int i = 0; i < matches.size(); i++ )
 {
 //-- Get the keypoints from the  matches
 obj.push_back( keypoints_1[ matches[i].queryIdx ].pt );
 scene.push_back( keypoints_2[ matches[i].trainIdx ].pt );
 }
 
  //-- Show detected matches
  imshow( " Matches", img_matches );

  for( int i = 0; i < (int)matches.size(); i++ )
  { printf( "--  Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); }
  
   // Mat H = findHomography( obj, scene, CV_RANSAC );
  Mat H = ransac(obj, scene);
 // Use the Homography Matrix to warp the images
 cv::Mat result;
 warpPerspective(img_1,result,H,cv::Size(img_1.cols+img_2.cols,img_1.rows));
 cv::Mat half(result,cv::Rect(0,0,img_2.cols,img_2.rows));
 img_2.copyTo(half);
 imshow( "Result", result );
 imwrite("result.png", result);
  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./SURF_FlannMatcher <img1> <img2>\n"); }

#endif