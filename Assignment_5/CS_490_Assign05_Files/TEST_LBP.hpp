#ifndef TEST_LBP_H
#define TEST_LBP_H

#include "LBP.hpp"

#define EPSILON		1e-05
#define GROUND_IMAGE_COUNT	7

struct ImageHist {
	string filename;
	double histogram[MAX_LABEL_CNT];
};

struct HistDist {
	string filename;
	double dist;
};

string getFilename(string path);
string getContainingFolder(string path);
double computeErrorSum(Mat processedImage, Mat groundImage);
bool checkForError(int testIndex, Mat computed, Mat ground);
bool checkForError(int testIndex, double *first, double *second, int histBinCnt);
void loadHistograms(string histogramFilename, vector<ImageHist> &allHists);
ImageHist extractLBP(string filepath);
void printPassed(int testIndex, bool passed);

bool TEST_getPixel();
bool TEST_ONE_getLBPNeighbors(
	Mat image, int centerRow,
	int centerCol, vector<int> ground);
bool TEST_getLBPNeighbors();
bool TEST_ONE_thresholdArray(
	vector<int> input,
	int threshold,
	vector<int> ground);
bool TEST_thresholdArray();

bool TEST_ONE_getUniformLabel(
	vector<int> input,	
	int ground);
bool TEST_getUniformLabel();
bool TEST_LBP(string inputPath);

#endif
