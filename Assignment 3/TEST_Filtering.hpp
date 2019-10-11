#ifndef TEST_FILTERING_H
#define TEST_FILTERING_H

#include "Filtering.hpp"

#define EPSILON		1e-06

string getFilename(string path);
double computeErrorSum(Mat processedImage, Mat groundImage);
bool checkForError(int testIndex, Mat computed, Mat ground);

bool TEST_ONE_applyLinearFilter(int testIndex, Mat input, Mat filter, double scaleFactor);
bool TEST_applyLinearFilter(string filepath);

bool TEST_ONE_applyBoxFilter(int testIndex, Mat input, int filterWidth, int filterHeight);
bool TEST_applyBoxFilter(string filepath);

bool TEST_ONE_applyGaussian3x3(int testIndex, Mat input);
bool TEST_applyGaussian3x3(string filepath);

bool TEST_ONE_applySobel3x3(int testIndex, Mat input, bool isVertical);
bool TEST_applySobel3x3(string filepath);

#endif
