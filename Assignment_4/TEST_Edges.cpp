#include "TEST_Edges.hpp"

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
	string filename = path.substr(found+1, newLen);

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

	cout << "\t" << "TEST " << testIndex << " total difference: " << error << endl;

	bool testPassed = true;
	if (error > EPSILON) {
		cout << "\t" << "TEST " << testIndex << ": FAILED" << endl;
		//cout << "\t" << "COMPUTED: " << computed << endl;
		//cout << "\t" << "GROUND: " << ground << endl;
		//exit(1);
		testPassed = false;
	}

	return testPassed;
}

bool TEST_ONE_applyLinearFilter(int testIndex, Mat input, Mat filter, double scaleFactor) {
		
	// Compute filtered image
	Mat output;
	applyLinearFilter(input, filter, output, scaleFactor);

	// Compute ground image
	Mat groundOutput;
	cv::flip(filter, filter, -1);
	cv::filter2D(input, groundOutput, CV_64F, filter, cv::Point(-1,-1), 0.0, BORDER_CONSTANT);
	groundOutput *= scaleFactor;

	// Did we pass?
	return checkForError(testIndex, output, groundOutput);
}

bool TEST_applyLinearFilter(string filepath) {

	bool allPassed = true;

	// TEST 1 //////////////////////////////////////////
	
	allPassed &= TEST_ONE_applyLinearFilter(
		1,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1),
		(Mat_<double>(1, 3) << 1, 2, 3),
		1.0);
											
	// TEST 2 //////////////////////////////////////////
	
	allPassed &= TEST_ONE_applyLinearFilter(
		2,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1),
		(Mat_<double>(1, 3) << 1, 2, 3),
		1.0/6.0);

	// TEST 3 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLinearFilter(
		3,
		(Mat_<double>(3, 6) <<  0, 1, 1, 0, 2, 1,
								1,2, 1, 4, 5, 1,
								7, 3, 2, 4, 2, 1),
		(Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9),
		1);

	// TEST 4 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLinearFilter(
		4,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1),
		(Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9),
		1.0 / 45.0);

	// TEST 5 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLinearFilter(
		5,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1),
		(Mat_<double>(3, 3) << -3, -2, -1, 0, 1, 2, 3, 4, 5),
		1.0);

	// TEST 6 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLinearFilter(
		6,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1),
			(Mat_<double>(3, 3) << -3, -2, -1, 0, 1, 2, 3, 4, 5),
		1.0/21.0);

	// TEST 7 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLinearFilter(
		7,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1),
			(Mat_<double>(3, 1) << 1, 2, 3),
		1.0/6.0);

	// TEST 8 //////////////////////////////////////////
	Mat image = imread(filepath, IMREAD_GRAYSCALE);
	image.convertTo(image, CV_64FC1);
	
	allPassed &= TEST_ONE_applyLinearFilter(
		8,
		image,
		(Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9),
		1.0 / 45.0);

	return allPassed;
}

bool TEST_ONE_applyGaussian3x3(int testIndex, Mat input) {
	// Compute filtered image
	Mat output;
	applyGaussian3x3(input, output);

	// Compute ground image
	Mat groundOutput;
	Mat inputFloat;
	input.convertTo(inputFloat, CV_64F);
	cv::GaussianBlur(inputFloat, groundOutput, Size(3, 3), 0, 0, BORDER_CONSTANT);

	// Did we pass?
	return checkForError(testIndex, output, groundOutput);
}

bool TEST_applyGaussian3x3(string filepath) {

	bool allPassed = true;

	// TEST 1 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyGaussian3x3(
		1,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1));

	// TEST 2 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyGaussian3x3(
		2,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1));

	// TEST 3 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyGaussian3x3(
		3,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1));


	// TEST 4 //////////////////////////////////////////
	Mat image = imread(filepath, IMREAD_GRAYSCALE);
	image.convertTo(image, CV_64FC1);

	allPassed &= TEST_ONE_applyGaussian3x3(
		4,
		image);

	return allPassed;
}

bool TEST_ONE_applyLaplacian(int testIndex, Mat input) {
	// Compute filtered image
	Mat output;
	applyLaplacian(input, output);

	// Compute ground image
	Mat groundOutput;
	Mat filter = (Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	cv::filter2D(input, groundOutput, CV_64F, filter, cv::Point(-1, -1), 0.0, BORDER_CONSTANT);
	groundOutput *= 1.0/8.0;
		
	// Did we pass?
	return checkForError(testIndex, output, groundOutput);
}

bool TEST_applyLaplacian(string filepath) {

	bool allPassed = true;

	// TEST 1 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLaplacian(
		1,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1));

	// TEST 2 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLaplacian(
		2,
		(Mat_<double>(1, 6) << 0, 1, 1, 0, 2, 1));

	// TEST 3 //////////////////////////////////////////

	allPassed &= TEST_ONE_applyLaplacian(
		3,
		(Mat_<double>(3, 6) <<
			0, 1, 1, 0, 2, 1,
			1, 2, 1, 4, 5, 1,
			7, 3, 2, 4, 2, 1));

	
	// TEST 4 //////////////////////////////////////////
	Mat image = imread(filepath, IMREAD_GRAYSCALE);
	image.convertTo(image, CV_64FC1);

	allPassed &= TEST_ONE_applyLaplacian(
		4,
		image);	

	return allPassed;
}

bool TEST_ONE_checkPairs(int testIndex, Mat input, int r1, int c1, int r2, int c2, int outR, int outC, int groundValue) {
	Mat edges = Mat::zeros(input.rows, input.cols, CV_8UC1);
	checkPairs(input, r1, c1, r2, c2, edges);	
	bool didPass = edges.at<uchar>(outR, outC) == groundValue;

	cout << "\t" << "TEST " << testIndex << " values: " << (int)(edges.at<uchar>(outR, outC)) << " " << groundValue << endl;

	if (!didPass) {
		cout << "\t" << "TEST " << testIndex << ": FAILED" << endl;
	}

	return didPass;
}

bool TEST_checkPairs() {

	bool allPassed = true;

	Mat testImage = (Mat_<double>(3, 3) <<
		5, 8, -1,
		0.1, -1, -7,
		-0.1, 1, 2);

	// TEST 1 //////////////////////////////////////////
	
	allPassed &= TEST_ONE_checkPairs(1, testImage,
		0,0,
		1,1,
		1,1, 255);
	
	// TEST 2 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(2, testImage,
		1, 1,
		2, 2,
		1, 1, 255);

	// TEST 3 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(3, testImage,
		0, 0,
		0, 1,
		0, 0, 0);

	// TEST 4 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(4, testImage,
		0, 0,
		0, 1,
		0, 1, 0);

	// TEST 5 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(5, testImage,
		1, 0,
		2, 0,
		1, 0, 0);

	// TEST 6 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(6, testImage,
		1, 0,
		2, 0,
		2, 0, 0);

	// TEST 7 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(7, testImage,
		0, 0,
		1, 1,
		0, 0, 0);

	// TEST 8 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(8, testImage,
		1, 2,
		2, 2,
		1, 2, 255);

	// TEST 9 //////////////////////////////////////////

	allPassed &= TEST_ONE_checkPairs(9, testImage,
		1, 2,
		2, 2,
		2, 2, 0);

	
	return allPassed;
}

bool TEST_ONE_applyMarrHildreth(int testIndex, Mat input, Mat groundImage) {
	// Call Marr-Hildreth
	Mat output;
	applyMarrHildreth(input, output);

	return checkForError(testIndex, output, groundImage);
}

bool TEST_applyMarrHildreth(string filepath) {

	bool allPassed = true;

	// Get containing ground folder
	string inputPath = getContainingFolder(filepath);
	string groundPath = inputPath + "../ground/";

	for (int i = 0; i < 6; i++) {
		string inputFilename = inputPath + "Image0" + std::to_string(i) + ".png";
		string groundFilename = groundPath + "EDGES_Image0" + std::to_string(i) + ".png";

		Mat inputImage = imread(inputFilename, IMREAD_GRAYSCALE);
		Mat groundImage = imread(groundFilename, IMREAD_GRAYSCALE);
		
		allPassed &= TEST_ONE_applyMarrHildreth(i, inputImage, groundImage);			
	}

	return allPassed;
}