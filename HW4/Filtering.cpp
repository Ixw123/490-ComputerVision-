#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

//#include <Filtering.hpp>
#include <memory.h>
#include <cstring>
#include <cstdint>
#include <stdio.h> 

using namespace cv;
using namespace std;

double convolveOnce(Mat input, Mat filter, int centerRow, int centerCol) {
	double sum = 0.0;
	int imr, imc;
	double filterval;
	double pixel;

	for (int i = 0 ; i < filter.rows; i++){
		for (int j = 0;j < filter.cols; j++) {
			imr = filter.rows/2 - i + centerRow;
			imc = filter.cols/2 - j + centerCol;
			filterval = filter.at<double>(i,j);
			if (imr < 0 || imr >= input.rows || imc < 0 || imc >= input.cols) {
				pixel = 0;
			}
			else {
				pixel = (double)input.at<double>(imr,imc);
			}
			sum += pixel*filterval;
		}
	}
	return sum;
}
void applyLinearFilter(Mat input, Mat filter, Mat &output, double scaleFactor ) {
	double sum = 0.0;
	//double *rfilter;

	output.create(input.rows, input.cols,CV_64FC1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			sum = convolveOnce(input, filter, i, j);
			output.at<double>(i,j) = sum;
		}
	}
	output *= scaleFactor;
}
void applyBoxFilter(Mat input, int filterWidth, int filterHeight, Mat &output) {
	Mat boxfilter;
	double size = filterWidth*filterHeight;
	boxfilter.create(filterHeight, filterWidth, CV_64FC1);
	boxfilter = Scalar(1.0);
	applyLinearFilter(input, boxfilter, output, (double)1.0/size);
}
void applyGaussian3x3(Mat input, Mat &output) {
	Mat gaussianfilter;
	gaussianfilter = (Mat_<double>(3,3)<<1,2,1,2,4,2,1,2,1);
	applyLinearFilter(input, gaussianfilter, output, 1.0/16.0);
}
void applySobel3x3(Mat input, bool isVertical, Mat &output) {
	Mat sobelfilter;
	if(isVertical) {
		sobelfilter = (Mat_<double>(3,3)<<1,2,1,0,0,0,-1,-2,-1);
	}
	else {
		sobelfilter = (Mat_<double>(3,3)<<1,0,-1,2,0,-2,1,0,-1);
	}
	applyLinearFilter(input, sobelfilter, output, 0.25);
}
void applyLaplacian(Mat input, Mat &output) {
	Mat laplacianfilter;
	laplacianfilter = (Mat_<double>(3,3)<<1,1,1,1,-8,1,1,1,1);
	applyLinearFilter(input, laplacianfilter, output, 1.0/8.0);
}
