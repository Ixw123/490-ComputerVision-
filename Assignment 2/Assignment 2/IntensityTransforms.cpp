/*****************************************************************************************
        Written by Micah Church
        Title: IntensityTransforms.cpp
        Target: G++ compiler
        Date: Febuary 2, 2018
*****************************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*****************************************************************************************
1) Calculate the the normalized histogram
        1.1) Zeros out all the values of hist
        1.2) Get the unormalized histogram
        1.3) Get the normalized histogram by using the unormalized histogram
2) Calculate the cumulative distribution fucntion (CDF) of the histogram
3) Perform histogram stretching
4) Use the functions to get and equalized image
        4.1) Compute the normalized histogram using calculateHistogram()
        4.2) Compute the cumulative distribution function (CDF) using calculateCumulative()
        4.3) Stretch the CDF using stretchCumulative()
        4.4) Multipy the stretched CDF by 255.0 (Not 256)
        4.5) Call create on the equalizedImage to make it the same size and type as origImage
        4.6) Transform each pizel intensity in origImage using the stretched CDF to get the
             pixels for the output image (equalizedImage)
*****************************************************************************************/
/*****************************************************************************************
    1) This function calculates the the normalized histogram
*****************************************************************************************/
void calculateHistogram(Mat image, double hist[], int length) {
        //1.1) Zero out all values of the histogram
        for(int i = 0; i < length; i++) {
                hist[i] = 0.0;
        }
        /*-------------------------------------------------------
         * 1.2) First get the unormalized histogram
         * This just means to determine the quantity
         * of every possible pizel
        -------------------------------------------------------*/
        for(int i = 0; i < image.rows; i++) {
                for(int j = 0; j < image.cols; j++) {
                        uchar pixel = image.at<uchar>(i,j);
                        hist[pixel]++;
                }
        }
        /*-------------------------------------------------------
         * 1.3) Now use the unormalized histogram to find the
         * normalized histogram which just means determine the
         * probability of finding each so you divide it by the
         * total number of pixels
        -------------------------------------------------------*/
        for(int i = 0; i < length; i++) {
                hist[i] /= image.rows*image.cols;
        }
}
/*****************************************************************************************
    2) Caluculate the cumulative distribution function(CDF) of the histogram
*****************************************************************************************/
void calculateCumulative(double hist[], double cdfHist[], int length) {
        //This just means to get the cumulative amount for the histogram
        for(int i = 0; i < length; i++) {
            if(i == 0)
                cdfHist[i] = hist[i];
            else
                cdfHist[i] = hist[i] + cdfHist[i - 1];
        }
}
/*****************************************************************************************
    3) Perform histogram stretching to even out all the values
*****************************************************************************************/
void stretchCumulative(double cdfHist[], double stretchCDFHist[], int length) {
        //First determine the first value of the histogram
        double cdfirst = cdfHist[0];
        //Take the first value and subtract that from all the values in the histogram
        for (int i = 0; i < length; i++) {
                stretchCDFHist[i] = cdfHist[i]-cdfirst;
        }
        //Get the last value of the new histogram
        double cdflast = stretchCDFHist[length - 1];
        //Divide all the values of the histogram by the last value
        for (int i = 0; i < length; i++) {
                stretchCDFHist[i] = stretchCDFHist[i]/cdflast;
        }

}
/*****************************************************************************************
    4) Equalize an image using all the functions and some extra steps
*****************************************************************************************/
void getEqualizedImage(Mat origImage, Mat &equalizedImage){
        //Declare all the things needed for the above functions
        const int length = 256;
        double cdfHist[length];
        double hist[length];
        double stretchCDFHist[length];
        //4.1) Call calculateHistogram to get the normalized histogram
        calculateHistogram(origImage, hist, length);
        //4.2) Compute the cumulative distribution function (CDF) using calculateCumulative()
        calculateCumulative(hist, cdfHist, length);
        //4.3) Stretch the CDF using stretchCumulative()
        stretchCumulative(cdfHist, stretchCDFHist, length);
        //4.4) Multipy the stretched CDF by 255.0 (Not 256)
        for(int i = 0; i < length; i++) {
                stretchCDFHist[i] *= 255.0;
        }
        //4.5) Call create on the equalizedImage to make it the same size and type as origImage
        equalizedImage.create(origImage.rows, origImage.cols, origImage.type());
        //4.6) Transform each pizel intensity in origImage using the stretched CDF to get the
        //     pixels for the output image (equalizedImage)
        for(int i = 0; i < origImage.rows; i++) {
                for(int j = 0; j < origImage.cols; j++) {
                        int pixel = origImage.at<uchar>(i,j);
                        equalizedImage.at<uchar>(i,j) = saturate_cast<uchar>(stretchCDFHist[pixel]);
                }
        }
}
