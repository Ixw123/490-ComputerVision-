#ifndef EDGES_H
#define EDGES_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Filtering.hpp"

using namespace cv;
using namespace std;

void checkPairs(Mat lapImage, int r1, int c1, int r2, int c2, Mat &edges);
void applyMarrHildreth(Mat input, Mat &edges);


#endif
