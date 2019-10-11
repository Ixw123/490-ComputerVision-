#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Filtering.hpp"
#include "Edges.hpp"
#include "TEST_Edges.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// If no command line arguments entered, exit
	if (argc < 4) {
		cout << "ERROR: Insufficient command line arguments!" << endl;
		cout << "[path to image] [path to directory for output] [testing? (0 or 1)]" << endl;
		return -1;
	}

	// Testing mode?
	int testingMode = atoi(argv[3]);

	// Get path to file
	string filepath = string(argv[1]);

	if (!testingMode) {
		// REGULAR APPLICATION

		// Get filename
		string filename = getFilename(filepath);

		// Load image
		cout << "Loading image: " << filepath << endl;
		Mat image = imread(filepath, IMREAD_GRAYSCALE);

		// Check if data is invalid
		if (!image.data) {
			cout << "ERROR: Could not open or find the image!" << endl;
			return -1;
		}
			
		// Perform edge detection
		Mat edges;
		applyMarrHildreth(image, edges);
				
		// Get path to output
		string outputDir = string(argv[2]);
				
		// Save output images			
		imwrite(outputDir + "/EDGES_" + filename, edges);		
		cout << "Images saved." << endl;

		// Show our image (with the filename as the window title)
		imshow(filename, image);
				
		// Show output images		
		imshow("Edge Image", edges);
		
		// Wait for a keystroke to close the window
		waitKey(-1);

		// Cleanup all windows
		destroyAllWindows();
	}
	else {
		// TESTING MODE

		bool allGood = true;

		// Testing applyLinearFilter
		if (!TEST_applyLinearFilter(filepath)) {
			cout << "TEST_applyLinearFilter: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_applyLinearFilter: SUCCESS" << endl;
		}
		
		// Testing applyGaussian3x3
		if (!TEST_applyGaussian3x3(filepath)) {
			cout << "TEST_applyGaussian3x3: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_applyGaussian3x3: SUCCESS" << endl;
		}

		// Testing applyLaplacian
		if (!TEST_applyLaplacian(filepath)) {
			cout << "TEST_applyLaplacian: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_applyLaplacian: SUCCESS" << endl;
		}

		// Testing checkPairs
		if (!TEST_checkPairs()) {
			cout << "TEST_checkPairs: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_checkPairs: SUCCESS" << endl;
		}

		// Testing applyMarrHildreth
		if (!TEST_applyMarrHildreth(filepath)) {
			cout << "TEST_applyMarrHildreth: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_applyMarrHildreth: SUCCESS" << endl;
		}
						
		// All tests good?
		if (allGood) {
			cout << "ALL TESTS SUCCEED!" << endl;
		}
	}

	return 0;
}

