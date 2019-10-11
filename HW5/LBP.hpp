#ifndef LBP_H
#define LBP_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const int NEIGHBOR_CNT = 8;
const int MAX_LABEL_CNT = NEIGHBOR_CNT + 2;

int getPixel(Mat inputImage, int row, int col);
void getLBPNeighbors(Mat inputImage, int centerRow, int centerCol, int *neighbors);
void thresholdArray(int threshold, int *data, int cnt);
int getUniformLabel(int *data, int cnt);
void getLBPImage(Mat inputImage, Mat &outputImageLBP);
void computeLBPHistogram(Mat imageLBP, double *histogram, int histBinCnt);
double computeHistDistance(double *first, double *second, int histBinCnt);

#endif
