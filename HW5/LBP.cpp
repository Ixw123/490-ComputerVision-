#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "LBP.hpp"

using namespace cv;
using namespace std;

int getPixel(Mat inputImage, int row, int col) {
	int value = 0;

	if (row <= inputImage.rows - 1 && row >= 0 && col <= inputImage.cols - 1 && col >= 0) {
		value = inputImage.at<uchar>(row,col);
	}

	return value;
}
void getLBPNeighbors(Mat inputImage, int centerRow, int centerCol, int *neighbors) {
	int value;

	for (int i = 0; i < NEIGHBOR_CNT; i++) {
		switch (i) {
			case 0:
				value = getPixel(inputImage, centerRow - 1, centerCol);
				break;
			case 1:
				value = getPixel(inputImage, centerRow - 1, centerCol + 1);
				break;
			case 2:
				value = getPixel(inputImage, centerRow, centerCol + 1);
				break;
			case 3:
				value = getPixel(inputImage, centerRow + 1, centerCol + 1);
				break;
			case 4:
				value = getPixel(inputImage, centerRow + 1, centerCol);
				break;
			case 5:
				value = getPixel(inputImage, centerRow + 1, centerCol - 1);
				break;
			case 6:
				value = getPixel(inputImage, centerRow, centerCol - 1);
				break;
			case 7:
				value = getPixel(inputImage, centerRow - 1, centerCol - 1);
				break;
		}
		*(neighbors + i) = value;
	}
}
void thresholdArray(int threshold, int *data, int cnt) {

	for (int i = 0; i < cnt; i++) {
		if (*(data + i) > threshold) {
			*(data + i) = 1;
		}
		else {
			*(data + i) = 0;
		}
	}
}
int getUniformLabel(int *data, int cnt) {
	int transitions = 0;
	int change, next;
	int num_one = 0;
	int label;

	for (int i = 0; i < cnt; i++) {
		change = *(data + i);
		next = *(data + ((i + 1) % cnt));
		if (change == 1) {
			num_one++;
		}
		if (change != next) {
			transitions++;
		}
	}
	if (transitions <= 2) {
		label = num_one;
	}
	else if (transitions > 2) {
		label = cnt + 1;
	}
	
	return label;
}
void getLBPImage(Mat inputImage, Mat &outputImageLBP) {
	int pixel;
	int label;
	int neighbors[NEIGHBOR_CNT];

	outputImageLBP.create(inputImage.rows, inputImage.cols,CV_8UC1);
	for (int i = 0; i < inputImage.rows; i++) {
		for (int j = 0; j < inputImage.cols; j++) {
			pixel = getPixel(inputImage, i, j);
			getLBPNeighbors(inputImage, i, j, neighbors);
			thresholdArray(pixel, neighbors, NEIGHBOR_CNT);
			outputImageLBP.at<uchar>(i,j) = getUniformLabel(neighbors, NEIGHBOR_CNT);
		}
	}
}
void computeLBPHistogram(Mat imageLBP, double *histogram, int histBinCnt) {
	int index;

	for (int i = 0; i < histBinCnt; i++) {
		*(histogram + i) = 0;
	}
	for (int i = 0; i < imageLBP.rows; i++) {
		for (int j = 0; j < imageLBP.cols; j++) {
			index = imageLBP.at<uchar>(i,j);
			*(histogram + index)+= 1;
		}
	}
	for(int i = 0; i < histBinCnt; i++) {
                histogram[i] /= imageLBP.rows*imageLBP.cols;
    }
}
double computeHistDistance(double *first, double *second, int histBinCnt) {
	double sum = 0;

	for (int i = 0; i < histBinCnt; i++) {
		sum += abs(*(first + i) - *(second + i));
	}

	return sum;
}