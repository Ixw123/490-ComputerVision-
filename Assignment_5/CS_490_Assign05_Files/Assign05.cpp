#include <opencv2/core/core.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "LBP.hpp"
#include "TEST_LBP.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// If no command line arguments entered, exit
	if (argc < 3) {
		cout << "ERROR: Insufficient command line arguments!" << endl;
		cout << "[path to image] [path to directory for output] [testing? (0 or 1)]" << endl;
		return -1;
	}

	// Testing mode?
	int testingMode = atoi(argv[2]);

	// Get path to input directory
	string baseInputDirectory = string(argv[1]);

	if (!testingMode) {
		// REGULAR APPLICATION

		vector<ImageHist> allHist;

		// Load up all images and compute histograms
		for (int i = 0; i < GROUND_IMAGE_COUNT; i++) {
			// Get full path
			string filename = "Image" + to_string(i) + ".png";
			string filepath = baseInputDirectory + "/" + filename;
			
			// Store histogram		
			ImageHist ihist = extractLBP(filepath);
			allHist.push_back(ihist);
		}

		// Load target image
		string filename = "Target.png";
		string targetpath = baseInputDirectory + "/" + filename;
		ImageHist targetHist = extractLBP(targetpath);
		
		// Compare target image histogram to other images
		cout << "DISTANCES FROM TARGET IMAGE:" << endl;
		for (int i = 0; i < allHist.size(); i++) {
			double dist = computeHistDistance(allHist.at(i).histogram, targetHist.histogram, MAX_LABEL_CNT);
			cout << allHist.at(i).filename << ": " << dist << endl;
		}
	}
	else {
		// TESTING MODE

		bool allGood = true;

		if (!TEST_getPixel()) {
			cout << "TEST_getPixel: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getPixel: SUCCESS" << endl;
		}

		if (!TEST_getLBPNeighbors()) {
			cout << "TEST_getLBPNeighbors: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getLBPNeighbors: SUCCESS" << endl;
		}

		if (!TEST_thresholdArray()) {
			cout << "TEST_thresholdArray: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_thresholdArray: SUCCESS" << endl;
		}

		if (!TEST_getUniformLabel()) {
			cout << "TEST_getUniformLabel: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getUniformLabel: SUCCESS" << endl;
		}
				
		if (!TEST_LBP(baseInputDirectory)) {
			cout << "TEST_LBP: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_LBP: SUCCESS" << endl;
		}

		// All tests good?
		if (allGood) {
			cout << "ALL TESTS SUCCEED!" << endl;
		}
	}

	return 0;
}
