#ifndef TEST_FILTERING_H
#define TEST_FILTERING_H

#include "Filtering.hpp"
#include "Edges.hpp"

#define EPSILON		1e-06

string getFilename(string path);
string getContainingFolder(string path);
double computeErrorSum(Mat processedImage, Mat groundImage);
bool checkForError(int testIndex, Mat computed, Mat ground);

bool TEST_ONE_applyLinearFilter(int testIndex, Mat input, Mat filter, double scaleFactor);
bool TEST_applyLinearFilter(string filepath);

bool TEST_ONE_applyGaussian3x3(int testIndex, Mat input);
bool TEST_applyGaussian3x3(string filepath);

bool TEST_ONE_applyLaplacian(int testIndex, Mat input);
bool TEST_applyLaplacian(string filepath);

bool TEST_ONE_checkPairs(int testIndex, Mat input, int r1, int c1, int r2, int c2, int outR, int outC, int groundValue);
bool TEST_checkPairs();

bool TEST_ONE_applyMarrHildreth(int testIndex, Mat input, Mat groundImage);
bool TEST_applyMarrHildreth(string filepath);

#endif
