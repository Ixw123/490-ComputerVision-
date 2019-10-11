#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// If no command line arguments entered, use webcam
	if (argc <= 1) {
		// Webcam

		cout << "Opening webcam..." << endl;

		// Grab the default camera
		VideoCapture camera(0);	

		// Did we get it?
		if (!camera.isOpened()) {
			cout << "ERROR: Cannot open camera!" << endl;
			return -1;
		}
			
		// Create window ahead of time
		string windowName = "Webcam";
		namedWindow(windowName);

		int key = -1;

		while (key == -1) {
			Mat frame;

			// Get next frame from camera
			camera >> frame; 

			// Show the image
			imshow(windowName, frame);

			// Wait 30 milliseconds, and grab any key presses
			key = waitKey(30);			
		}

		// Camera's destructor will close the camera

		cout << "Closing application..." << endl;
	}
	else {
		// Try to load image from argument

		// Get filename
		string filename = string(argv[1]);

		// Load image
		cout << "Loading image: " << filename << endl;
		Mat image = imread(filename); // For grayscale: imread(filename, IMREAD_GRAYSCALE);

		// Check if data is invalid
		if (!image.data) {
			cout << "ERROR: Could not open or find the image!" << endl;
			return -1;
		}

		// Show our image (with the filename as the window title)
		imshow(filename, image);

		// Wait for a keystroke to close the window
		waitKey(-1);

		// Cleanup this window
		destroyWindow(filename);
		// If we wanted to get rid of ALL windows: destroyAllWindows();			
	}

	return 0;
}


