#include "TEST_LBP.hpp"

string getFilename(string path) {
	// Find right-most / or \\ character
	size_t found = path.rfind("/");
	if (found == string::npos) {
		found = path.rfind("\\");
	}

	// Didn't find anything; start at beginning
	if (found == string::npos) {
		found = 0;
	}

	size_t newLen = path.length() - found;
	string filename = path.substr(found + 1, newLen);

	return filename;
}

string getContainingFolder(string path) {
	// Find right-most / or \\ character
	size_t found = path.rfind("/");
	if (found == string::npos) {
		found = path.rfind("\\");
	}

	string containPath = "";

	// Didn't find anything; pass local directory
	if (found == string::npos) {
		containPath = "./";
	}
	else {
		containPath = path.substr(0, found + 1);
	}

	return containPath;
}

double computeErrorSum(Mat processedImage, Mat groundImage) {
	// Compare output image to ground truth
	// (Some code taken from: https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html)
	Mat diffImage;
	absdiff(processedImage, groundImage, diffImage);
	Mat floatDiffImage;
	diffImage.convertTo(floatDiffImage, CV_64F);
	floatDiffImage = floatDiffImage.mul(floatDiffImage);
	Scalar diff = sum(floatDiffImage);
	return diff[0];
}

bool checkForError(int testIndex, Mat computed, Mat ground) {

	// Compute error
	double error = computeErrorSum(computed, ground);

	cout << "\t\t" << "TEST " << testIndex << " total difference: " << error << endl;

	bool testPassed = true;
	if (error > EPSILON) {
		cout << "\t\t\t" << "TEST " << testIndex << ": FAILED" << endl;
		//cout << "\t" << "COMPUTED: " << computed << endl;
		//cout << "\t" << "GROUND: " << ground << endl;
		//exit(1);
		testPassed = false;
	}

	return testPassed;
}

bool checkForError(int testIndex, double *first, double *second, int histBinCnt) {
	// Convert to Mats
	Mat computed(Size(histBinCnt, 1), CV_64FC1, first);
	Mat ground(Size(histBinCnt, 1), CV_64FC1, second);

	// Compute error
	return checkForError(testIndex, computed, ground);
}

void loadHistograms(string histogramFilename, vector<ImageHist> &allHists) {
	std::ifstream file(histogramFilename);

	if (file.fail()) {
		cout << "ERROR: COULD NOT LOAD HISTOGRAM FILE!" << endl;
		exit(1);
	}

	while (!file.eof()) {
		ImageHist hist;
		if (file >> hist.filename) {

			for (int i = 0; i < MAX_LABEL_CNT; i++) {
				file >> hist.histogram[i];
			}

			allHists.push_back(hist);
		}
	}

	file.close();
}

void loadDistances(string distancesFilename, vector<HistDist> &allDist) {
	std::ifstream file(distancesFilename);

	if (file.fail()) {
		cout << "ERROR: COULD NOT LOAD DISTANCES FILE!" << endl;
		exit(1);
	}

	while (!file.eof()) {
		HistDist histDist;
		if (file >> histDist.filename) {
			file >> histDist.dist;

			allDist.push_back(histDist);
		}
	}

	file.close();
}

ImageHist extractLBP(string filepath) {
	cout << "Loading image: " << filepath << endl;

	// Load image
	Mat image = imread(filepath, IMREAD_GRAYSCALE);

	// Check if data is invalid
	if (!image.data) {
		cout << "ERROR: Could not open or find the image!" << endl;
		exit(1);
	}

	// Get LBP image
	Mat imageLBP;
	getLBPImage(image, imageLBP);

	// Compute LBP histogram
	ImageHist ihist;
	ihist.filename = getFilename(filepath);
	computeLBPHistogram(imageLBP, ihist.histogram, MAX_LABEL_CNT);

	return ihist;
}

bool TEST_LBP(string inputPath) {
	bool allPassed = true;

	// Get containing ground folder	
	string groundPath = inputPath + "/../ground/";

	// Load ground truth histograms
	string groundHistFilename = groundPath + "HISTOGRAMS.txt";
	vector<ImageHist> groundHist;
	loadHistograms(groundHistFilename, groundHist);

	// Load ground truth distances
	string groundDistFilename = groundPath + "DISTANCES.txt";
	vector<HistDist> allDist;
	loadDistances(groundDistFilename, allDist);

	// Load target image
	string filename = "Target.png";
	string targetpath = inputPath + "/" + filename;
	ImageHist targetHist = extractLBP(targetpath);

	for (int i = 0; i < GROUND_IMAGE_COUNT; i++) {
		// Load image
		string inputFilename = "Image" + std::to_string(i) + ".png";
		string fullInputPath = inputPath + "/" + inputFilename;
		Mat inputImage = imread(fullInputPath, IMREAD_GRAYSCALE);

		if (!inputImage.data) {
			cout << "ERROR: Could not open or find the image " << fullInputPath << endl;
			exit(1);
		}

		// Compute LBP
		Mat outputLBP;
		getLBPImage(inputImage, outputLBP);

		// Load ground truth LBP image
		string fullGroundPath = groundPath + "LBP_Image" + std::to_string(i) + ".png";
		Mat groundImage = imread(fullGroundPath, IMREAD_GRAYSCALE);

		if (!groundImage.data) {
			cout << "ERROR: Could not open or find the image " << fullGroundPath << endl;
			exit(1);
		}

		cout << "TEST " << i << ": " << inputFilename << endl;

		// Test LBP image
		cout << "\t" << "getLBPImage() test..." << endl;
		allPassed &= checkForError(i, outputLBP, groundImage);

		// Compute histogram
		double hist[MAX_LABEL_CNT];
		computeLBPHistogram(outputLBP, hist, MAX_LABEL_CNT);

		// Find matching ground histogram
		int matchingHist = -1;
		for (int j = 0; j < groundHist.size(); j++) {
			if (groundHist.at(j).filename == inputFilename) {
				matchingHist = j;
				break;
			}
		}

		cout << "\t" << "computeLBPHistogram() test..." << endl;
		allPassed &= checkForError(i, hist, groundHist.at(matchingHist).histogram, MAX_LABEL_CNT);

		// Get distance from target
		double dist = computeHistDistance(hist, targetHist.histogram, MAX_LABEL_CNT);

		int matchingDist = -1;
		for (int j = 0; j < allDist.size(); j++) {
			if (allDist.at(j).filename == inputFilename) {
				matchingDist = j;
				break;
			}
		}
		
		double error = fabs(allDist.at(matchingDist).dist - dist);

		cout << "\t" << "computeHistDistance() test..." << endl;
		cout << "\t\t" << "Total difference: " << error << endl;

		bool testPassed = true;
		if (error > EPSILON) {
			cout << "\t\t\t" << "TEST FAILED" << endl;			
			testPassed = false;
		}

		allPassed &= testPassed;

		if (!allPassed) {
			break;
		}
	}

	return allPassed;

}

void printPassed(int testIndex, bool passed) {
	if (passed) {
		cout << "\t" << "TEST " << testIndex << ": SUCCEEDED" << endl;
	}
	else {
		cout << "\t" << "TEST " << testIndex << ": FAILED" << endl;
	}
}

bool TEST_getPixel() {
	bool allPassed = true;

	Mat image = (Mat_<uchar>(2, 3) << 1, 2, 3, 4, 5, 6);
	int testIndex = 0;

	allPassed &= (getPixel(image, 0, 0) == 1);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 0, 1) == 2);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 0, 2) == 3);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, 1, 0) == 4);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 1, 1) == 5);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 1, 2) == 6);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, -1, 0) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 2, 0) == 0);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, -1, 1) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 2, 1) == 0);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, -1, 2) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 2, 2) == 0);
	printPassed(testIndex++, allPassed);
	
	allPassed &= (getPixel(image, 0, -1) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 0, 3) == 0);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, 1, -1) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 1, 3) == 0);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, -1, -1) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, -1, 3) == 0);
	printPassed(testIndex++, allPassed);

	allPassed &= (getPixel(image, 2, -1) == 0);
	printPassed(testIndex++, allPassed);
	allPassed &= (getPixel(image, 2, 3) == 0);
	printPassed(testIndex++, allPassed);
	
	return allPassed;
}

bool TEST_ONE_getLBPNeighbors(	Mat image, 
								int centerRow, 
								int centerCol, 
								vector<int> ground) {

	int neighbors[MAX_LABEL_CNT];
	getLBPNeighbors(image, centerRow, centerCol, neighbors);

	for (int i = 0; i < ground.size(); i++) {
		if (ground.at(i) != neighbors[i]) {
			return false;
		}
	}

	return true;
}

bool TEST_getLBPNeighbors() {
	bool allPassed = true;

	Mat image = (Mat_<uchar>(3, 3) << 
					1, 2, 3, 
					4, 5, 6,
					7, 8, 9);
	int testIndex = 0;
		
	allPassed &= TEST_ONE_getLBPNeighbors(image, 1, 1,
			{ 2, 3, 6, 9, 8, 7, 4, 1 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getLBPNeighbors(image, 0, 0,
			{ 0, 0, 2, 5, 4, 0, 0, 0 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getLBPNeighbors(image, 2, 1,
			{ 5, 6, 9, 0, 0, 0, 7, 4 });
	printPassed(testIndex++, allPassed);

	return allPassed;
}

bool TEST_ONE_thresholdArray(
	vector<int> input,
	int threshold,
	vector<int> ground) {

	thresholdArray(threshold, input.data(), input.size());

	for (int i = 0; i < ground.size(); i++) {
		if (ground.at(i) != input.at(i)) {
			return false;
		}
	}

	return true;
}

bool TEST_thresholdArray() {
	bool allPassed = true;
	int testIndex = 0;
	
	allPassed &= TEST_ONE_thresholdArray(
	{ 2, 3, 6, 9, 8, 7, 4, 1 }, 5,
	{ 0, 0, 1, 1, 1, 1, 0, 0 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_thresholdArray(
	{ 2, 3, 6, 9, 8, 7, 4, 1 }, 4,
	{ 0, 0, 1, 1, 1, 1, 0, 0 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_thresholdArray(
	{ 2, 3, 6, 9, 8, 7, 4, 1 }, 0,
	{ 1, 1, 1, 1, 1, 1, 1, 1 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_thresholdArray(
	{ 2, 3, 6, 9, 8, 7, 4, 1 }, 9,
	{ 0, 0, 0, 0, 0, 0, 0, 0 });
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_thresholdArray(
	{ 2, 3, 6, 9 }, 5,
	{ 0, 0, 1, 1 });
	printPassed(testIndex++, allPassed);

	return allPassed;
}

bool TEST_ONE_getUniformLabel(
	vector<int> input,
	int ground) {

	int label = getUniformLabel(input.data(), input.size());
	return (label == ground);
}

bool TEST_getUniformLabel() {

	bool allPassed = true;
	int testIndex = 0;

	allPassed &= TEST_ONE_getUniformLabel(
	{ 0, 1, 1, 1, 0, 0, 0, 0 }, 3);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 0, 0, 0, 0, 0, 0, 0, 0 }, 0);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 0, 0, 0, 0, 1, 1, 1, 0 }, 3);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 1, 0, 0, 0, 1, 1, 1, 1 }, 5);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 1, 0, 1, 0, 1, 1, 1, 1 }, 9);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 0, 1, 0, 1 }, 5);
	printPassed(testIndex++, allPassed);

	allPassed &= TEST_ONE_getUniformLabel(
	{ 1, 1, 1, 1, 1 }, 5);
	printPassed(testIndex++, allPassed);

	return allPassed;
}

