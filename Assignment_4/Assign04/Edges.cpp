#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Filtering.hpp"

using namespace cv;
using namespace std;


void checkPairs(Mat lapImage, int r1, int c1, int r2, int c2, Mat &edges) {
	double val1, val2;

	val1 = (double)lapImage.at<double>(r1,c1);
	val2 = (double)lapImage.at<double>(r2,c2);
	double val;

	val = abs(val1 - val2);
	if (val < 0.625) {

	}
	else if (val1 < 0 && val2 < 0 || val1 >= 0 && val2 >= 0) {

	}
	else {
		if (val1 < val2) {
			edges.at<uchar>(r1,c1) = 255;
			//val1 = 255; 
		}
		else {
			edges.at<uchar>(r2,c2) = 255;
			//val2 = 255;
		}
	}
}
void applyMarrHildreth(Mat input, Mat &edges) {
	Mat output;
	input.convertTo(input, CV_64FC1);
	for(int i = 0; i < 9; i++) {
		applyGaussian3x3(input, output);
		output.copyTo(input);
	}
	applyLaplacian(input, output);
	edges.create(output.rows, output.cols,CV_8UC1);
	edges = Scalar(0);
	for (int i = 1; i < output.rows - 1; i++)
		for (int j = 1; j < output.cols - 1; j++) {
			checkPairs(output, i, j, i + 1, j, edges);
			checkPairs(output, i, j, i + 1, j + 1, edges);
			checkPairs(output, i, j, i, j + 1, edges);
			checkPairs(output, i, j, i + 1, j - 1, edges);
		}
}