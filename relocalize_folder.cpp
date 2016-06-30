// No copyright - Volker

#include <stdio.h>

// Include standard libraries
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>

#include "relocalize.h"

using namespace std;

int main(int argc, char *argv[]) {

   int max_imgs;

  // Construct relocalizer with reference image path
  Relocalizer relocalizer(argv[1]);

  /* Set maximum amount of images */
  if (argc < 3) {
     max_imgs = 1000;
  } else {
     max_imgs = atoi(argv[3]);
  }

  FILE *fp_sift = NULL;
  fp_sift = fopen("sift_labels.csv", "a");

  int i = 0;
  for (i = 0; i < max_imgs; i++) {

     // Read in query image
     char impath[2048];
     sprintf(impath, "%s/img_%05d.png", argv[2], i);
     printf("%s\n", impath);

     cv::Mat query_img = cv::imread(impath);

     // Get estimation (x, y) in pixels from relocalizer
     std::vector<float> estimation;
     estimation = relocalizer.calcLocation(query_img, i);

     // Print estimations
     cout << estimation[0] << " " << estimation[1] << endl;
     int est_x = (int) estimation[0];
     int est_y = (int) estimation[1];
     int est_matches = (int) estimation[2];

     fprintf(fp_sift, "%d,%d,%d\n", est_x, est_y, est_matches);
  }

  fclose(fp_sift);
  return 0;
}
