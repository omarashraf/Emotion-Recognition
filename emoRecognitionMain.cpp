#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "ml.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>	

#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

using namespace std;
using namespace cv;

// function headers
std::vector<double> detectFaceFeatures(Mat &frame);
void generateSampleData(std::vector<double> samples_data, string label);

// global variables
Mat frame; 

String face_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String mouth_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml";
String eyes_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String nose_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml";

CascadeClassifier face_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;

string window_name = "Capture - Face detection";
int thresh = 110;
int max_thresh = 255;
float canny_ratio = 1.55;
RNG rng(12345);

// for the 4 corners of the mouth (the x and y coordinates within the mouth image (region of interest), not the whole frame)
int minMouthX = 1000, maxMouthX = -1, minMouthY = 1000, maxMouthY = -1;

// 4 points representing the corners of the mouth, wrt the mouth's region of interest.
Point leftCornerMouth(0, 0);
Point rightCornerMouth(0, 0);
Point topCornerMouth(0, 0);
Point bottomCornerMouth(0, 0);

// for the 4 corners of the left and right eye, wrt eye's region of interest (same variables are used for both eyes to get max. and min.).
int minLeftEyeX, maxLeftEyeX, minLeftEyeY, maxLeftEyeY;

// 4 points representing the corners of the left and right eyes, the coordinates are wrt the eye's region of interest.
Point leftCornerLeftEye(0, 0);
Point rightCornerLeftEye(0, 0);
Point topCornerLeftEye(0, 0);
Point bottomCornerLeftEye(0, 0);

// for the 4 corners of the eyebrows, wrt eyebrows' region of interest.
int minEyebrowX = 1000, maxEyebrowX = -1, minEyebrowY = 1000, maxEyebrowY = -1;
int maxYminEyebrowX = -1, maxYmaxEyebrowX = -1;

// 4 points representing the corners of the eyebrow, wrt eyebrows' region of interest.
Point leftCornerEyebrow(0, 0), rightCornerEyebrow(0, 0), topCornerEyebrow(0, 0), bottomCornerEyebrow(0, 0);

// flag to eleminate contours that don't belong to the mouth (contours surrounding the chin).
bool eleminateChin;

// flags to eleminate contours that don't belong to the eye.
bool eleminateEyebrows;
bool eleminateBelowEyes;

// data holders that are used in the generateSampleData().
string newLine;
stringstream outLine;

// points' coordinates relative to the whole frame (mouth).
Point left_corner_mouth_whole(0, 0);
Point top_corner_mouth_whole(0, 0);
Point right_corner_mouth_whole(0, 0);
Point bottom_corner_mouth_whole(0, 0);

// points' coordinates relative to the whole frame (left eye).
Point left_corner_left_eye(0, 0);
Point top_corner_left_eye(0, 0);
Point right_corner_left_eye(0, 0);
Point bottom_corner_left_eye(0, 0);

// points' coordinates relative to the whole frame (right eye).
Point left_corner_right_eye(0, 0);
Point top_corner_right_eye(0, 0);
Point right_corner_right_eye(0, 0);
Point bottom_corner_right_eye(0, 0);

// points' coordinates relative to the whole frame (left eyebrow).
Point left_corner_left_eyebrow_whole(0, 0);
Point top_corner_left_eyebrow_whole(0, 0);
Point right_corner_left_eyebrow_whole(0, 0);
Point bottom_corner_left_eyebrow_whole(0, 0);

// points' coordinates relative to the whole frame (right eyebrow).
Point left_corner_right_eyebrow_whole(0, 0);
Point top_corner_right_eyebrow_whole(0, 0);
Point right_corner_right_eyebrow_whole(0, 0);
Point bottom_corner_right_eyebrow_whole(0, 0);

// point's coordinate of the center of the nose wrt whole frame.
Point nose_center_whole(0, 0);

// points' coordinates of the left and right eyes' centers wrt whole frame.
Point eye_center_left;
Point eye_center_right;

// counter for renaming each of the frames that are generated from the video stream, 
// in order to test all of the different frames being generated.
int counter_recording = 1;


// the main method caputers a frame/video stream, calls the main method detectFaceFeatures()
// that gets all the points of interest on the face, trains the machine learning algorithm, if needed,
// with the obtained data, and predicts the emotion for new samples captured. 
int main(int argc, const char* argv[]) {

	// load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!mouth_cascade.load(mouth_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!nose_cascade.load(nose_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// preparing and configuring the default camera to work with opencv.
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "The camera is not opened." << endl;
		while (1);
		return -1;
	}

	// to train the classfier of the mouth/eyebrows, uncomment the corresponding 
	// method to train the data accordingly.

	//trainClassifierMouth();
	//trainClassifierEyebrows();
	
	// for using the video stream, make the for loop run infintely,
	// however for testing just one frame, one iteration is enough.
	for (;;) {		

		cap >> frame;   // uncomment for video testing.
		//frame = imread("C:/Users/Omar/Downloads/samples/omar_happy.jpg"); // uncomment and add the path of your image for image testing.


		if (!frame.empty()) {
			cout << "----------------------------------------------" << endl;
			// applying the main method which gets the points of interest on the face and returns
			// the coordinates of all of the aforementioned points.
			std::vector<double> face_data = detectFaceFeatures(frame);

			// data holders for mouth and eyebrows' data separately (2 parameters for the mouth, 
			// 2 parameters for the eyebrows).
			std::vector<double> eyebrows_data;
			std::vector<double> mouth_data;

			if (face_data.size() != 0) {
				for (int q = 0; q < face_data.size() / 2; q++) {
					eyebrows_data.push_back(face_data[q]);
				}
				for (int q = 2; q < face_data.size(); q++) {
					mouth_data.push_back(face_data[q]);
				}
				cout << "******Eyebrows' parameters******" << endl;
				for (int j = 0; j < eyebrows_data.size(); j++) {
					if (j == 0) {
						cout << "Eyebrows' height parameter --> " << eyebrows_data[j] << endl;
					}
					else {
						cout << "Eyebrows' angles parameter --> " << eyebrows_data[j] << endl;
					}
				}
				cout << "******Mouth's parameters******" << endl;
				for (int j = 0; j < mouth_data.size(); j++) {
					if (j == 0) {
						cout << "Mouth's height ratio --> " << mouth_data[j] << endl;
					}
					else {
						cout << "Mouth's area --> " << mouth_data[j] << endl;
					}
				}

				// loading the machine learning algorithm and passing the eyebrows data, and 
				// getting the prediction back as a number.
				// +ve number --> surprise emotion.
				// -ve number --> non-surprise emotion.
				double eyebrow_predicition = loadClassifierAndPredictEyebrows(eyebrows_data);
				if (eyebrow_predicition >= 0) {
					cout << "eyebrow_prediction " << eyebrow_predicition << endl;
					cout << "Prediction of emotion --> Surprise!" << endl;
					cv::putText(frame, "Surprise!", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
						1.0, cv::Scalar(255, 0, 0), 1);
				}
				else {
					// mouth points are only detected in non-surprise emotion.
					circle(frame, left_corner_mouth_whole, 1, Scalar(0, 0, 255), 3);
					circle(frame, right_corner_mouth_whole, 1, Scalar(0, 0, 255), 3);
					circle(frame, top_corner_mouth_whole, 1, Scalar(0, 0, 255), 3);
					circle(frame, bottom_corner_mouth_whole, 1, Scalar(0, 0, 255), 3);
					// loading the machine learning algorithm with the data of the mouth and getting
					// back the prediction as a number.
					// +ve number --> happy emotion.
					// -ve number --> sad emotion.
					double mouth_prediction = loadClassifierAndPredictMouth(mouth_data);
					cout << "mouth_prediction " << mouth_prediction << endl;
					if (mouth_prediction >= 0) {
						cout << "Predtiction of emotion --> Happy =)" << endl;
						cv::putText(frame, "Happy =)", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
							1.0, cv::Scalar(255, 0, 0), 1);
					}
					else {
						cout << "Predtiction of emotion --> Sad =(" << endl;
						cv::putText(frame, "Sad =(", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
							1.0, cv::Scalar(255, 0, 0), 1);
					}
				}
			}

			// to save the frames being recorded during a video stream, uncomment the following snippet.
			string tmp = "C:/Users/Omar/Downloads/video_samples/test/frame_" + std::to_string(static_cast<long long> (counter_recording));
			string filename = tmp + ".png";
			cv::imwrite(filename, frame);
			counter_recording++; 

			// showing the frame captured with the pointss of interest on it.
			cv::imshow("frame ", frame);
			if(waitKey(20) >= 0) {}
		}
		else {
			printf("No captured frame!");
		}
	}
	int c = waitKey(0);
	while (1);
	return 0;
}

// detectFaceFeatures() method detects points of interest of mouth, eyes(needs to be improved) and eyebrows
// and calculates the parameters, later on passed to the machine learning algorithm. 
std::vector<double> detectFaceFeatures(Mat &frame) {
	
	// to return the eyebrows and mouth data at the end of the method.
	std::vector<double> ml_parameters;

	double y_coordinate = -1, left_top_height_ratio = -1, right_top_height_ratio = -1, left_right_height_ratio = -1;

	// three vectors that holds the to-be-detected features accordingly.
	std::vector<Rect> faces;
	std::vector<Rect> mouth;
	std::vector<Rect> eyes;

	// gray-scale version of the captured frame.
	Mat frame_gray;

	// indices indicating which of the detected features is to be used.
	int face_index = 0;
	int mouth_index = 0;

	// boolean variables to state which eye has been detected.
	bool left_eye = false, right_eye = false;

	// converting the input frame to gray-scale and equalizing histogram.
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	
	// variables to hold the eyes' and eyebrows' region of interest.
	Mat eye_image;
	Rect eye_rectangle;
	Mat eyebrow_image;
	Rect eyebrow_rectangle;

	// variables to hold the coordinates of the mouth corners wrt mouth's region of interes.
	minMouthX = 1000, maxMouthX = -1, minMouthY = 1000, maxMouthY = -1;
	leftCornerMouth.x = 0, leftCornerMouth.y = 0;
	rightCornerMouth.x = 0, rightCornerMouth.y = 0;
	topCornerMouth.x = 0, topCornerMouth.y = 0;
	bottomCornerMouth.x = 0, bottomCornerMouth.y = 0;
	eleminateChin = false;

	// variables to get the eyebrows' region of interest using that of the eyes later on.
	int eyebrowX, eyebrowY, eyebrowWidth, eyebrowHeight;
	int upper_limit_eyebrows_index;

	// detect faces using the pre-defined classifier.
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//cout << "Face size: " << faces.size() << endl;		// uncomment to get the no. faces detected.
	// getting which face is the main face in the frame captured.
	if (faces.size() > 0) {
		for (int i = 0; i < faces.size(); i++) {
			if (!(faces[i].width < 175 && faces[i].height < 175)) {
				face_index = i;
				break;
			}	
		}

		// faceROI is an image of only the rectangle surrounding the face detected in the image.
		Mat faceROI = frame_gray(faces[face_index]);
		//rectangle(frame, faces[face_index], Scalar(0, 255, 0), 2, 8);		\\ uncomment to draw a rectangle surrounding only the face.

		// detect the eyes, using the pre-defined classifiers.
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// variables to hold the images and points of edges detected within the eyes region.
		Mat canny_output_eye;
		std::vector<std::vector<Point>> contours_eye;
		std::vector<Vec4i> hierarchy_eye;

		//cout << "Eyes: " << eyes.size() << endl;		// uncomment to get the no. of eyes detected.
		if (eyes.size() == 0) {
			cout << "No eyes are detected." << endl;
		}
		else {
			if (eyes.size() == 1 || eyes.size() == 2) {
				for (int j = 0; j < eyes.size(); j++) {
					// resetting the variables of the eye again for the next iteration (i.e. next eye).
					minLeftEyeX = 1000, maxLeftEyeX = -1, minLeftEyeY = 1000, maxLeftEyeY = -1;
					leftCornerLeftEye.x = 0, leftCornerLeftEye.y = 0, rightCornerLeftEye.x = 0,	rightCornerLeftEye.y = 0;
					topCornerLeftEye.x = 0,	topCornerLeftEye.y = 0,	bottomCornerLeftEye.x = 0, topCornerLeftEye.y = 0;
					y_coordinate = -1, left_top_height_ratio = -1, right_top_height_ratio = -1, left_right_height_ratio = -1;

					// getting the exact rectangle surrounding the eyes.
					eye_rectangle = Rect(faces[face_index].x + eyes[j].x, faces[face_index].y + eyes[j].y, eyes[j].width, eyes[j].height);
					eye_image = frame(eye_rectangle);
					
					cvtColor(eye_image, eye_image, CV_RGB2GRAY);
					equalizeHist(eye_image, eye_image);

					// applying the canny edge detection method, then getting the contours and storing them in contours_eye.
					Canny(eye_image, canny_output_eye, thresh, thresh * 3.5, 3);
					findContours(canny_output_eye, contours_eye, hierarchy_eye, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
					
					// eliminating the contours surrounding the eyes is done using basic mathematical manipulation.
					// to simplify the idea, an average of each of the arrays in the 2D array (contours_eye) and according to
					// a constant threshold, this array of contours is eliminated/kept.
					// array of averages for contours.
					std::vector<double> averageContours;

					int lowerLimit = 0;
					int upperLimit = contours_eye.size() - 1;

					// Calc. average of each row of contours.
					for (int s = 0; s < contours_eye.size(); s++) {
						double av = 0;
						for (int h = 0; h < contours_eye[s].size(); h++) {
							av += contours_eye[s][h].y;
						}
						av /= contours_eye[s].size();
						averageContours.push_back(av);
					}

					if (contours_eye.size() != 0 && contours_eye.size() != 1) {
						for (int b = 0; b < (contours_eye.size() / 2) - 1; b++) {
							if (abs(averageContours[b] - averageContours[b + 1]) > 8) {
								eleminateBelowEyes = true;
								lowerLimit = b + 1;
							}
						}

						for (int b = (contours_eye.size() / 2) - 1; b < contours_eye.size() - 1; b++) {
							if (abs(averageContours[b] - averageContours[b + 1]) >= 6) {
								eleminateEyebrows = true;
								upperLimit = b;
							}
						}
					}
					else {
						cout << "No contours are deteced!!" << endl;
					}

					// determine the start and end of conrours included accoring to the elimination process done previously.
					int start = 0;
					int end = contours_eye.size();

					if (eleminateEyebrows && eleminateBelowEyes) {
						start = lowerLimit;
						end = upperLimit;
					}
					else {
						if (eleminateBelowEyes) {
							start = lowerLimit;
						}
						else {
							if (eleminateEyebrows) {
								end = upperLimit;
							}
						}
					}

					// drawing the contours for debugging.
					/*for (int k = start; k < end; k++) {
						drawContours(eye_image, contours_eye, k, Scalar(255, 255, 255), 2, 8, hierarchy_eye, 0, Point());
					}*/

					// selecting the 4 points that determine the eye.
					for (int m = start; m < end; m++) {
						for (int n = 0; n < contours_eye[m].size(); n++) {
							minLeftEyeX = contours_eye[m][n].x < minLeftEyeX ? contours_eye[m][n].x : minLeftEyeX;
							leftCornerLeftEye = contours_eye[m][n].x <= minLeftEyeX ? contours_eye[m][n] : leftCornerLeftEye;
							maxLeftEyeX = contours_eye[m][n].x > maxLeftEyeX ? contours_eye[m][n].x : maxLeftEyeX;
							rightCornerLeftEye = contours_eye[m][n].x >= maxLeftEyeX ? contours_eye[m][n] : rightCornerLeftEye;
							minLeftEyeY = contours_eye[m][n].y < minLeftEyeY ? contours_eye[m][n].y : minLeftEyeY;
							topCornerLeftEye = contours_eye[m][n].y <= minLeftEyeY ? contours_eye[m][n] : topCornerLeftEye;
							maxLeftEyeY = contours_eye[m][n].y > maxLeftEyeY ? contours_eye[m][n].y : maxLeftEyeY;
							bottomCornerLeftEye = contours_eye[m][n].y >= maxLeftEyeY ? contours_eye[m][n] : bottomCornerLeftEye;
						}
					}

					// centering the top and bottom points of the eyes.
					topCornerLeftEye.x = eyes[j].width / 2;
					bottomCornerLeftEye.x = eyes[j].width / 2;

					// determining if the eye being dealt with in this iteration is the left/right eye.
					if (eyes[j].x < (double) faces[face_index].width / 2) {
						eye_center_left.x = eyes[j].x + faces[face_index].x + (double) eyes[j].width / 2;
						eye_center_left.y = eyes[j].y + faces[face_index].y + (double) eyes[j].height / 2;
					}
					else {
						eye_center_right.x = eyes[j].x + faces[face_index].x + (double) eyes[j].width / 2;
						eye_center_right.y = eyes[j].y + faces[face_index].y + (double) eyes[j].height / 2;
					}

					// drawing the corners in the images representing the eyes only (debugging).
					/*circle(eye_image, Point(leftCornerLeftEye.x, leftCornerLeftEye.y), 1, Scalar(0, 0, 255), 3);
					circle(eye_image, Point(rightCornerLeftEye.x, rightCornerLeftEye.y), 1, Scalar(0, 0, 255), 3);
					circle(eye_image, Point(topCornerLeftEye.x, topCornerLeftEye.y), 1, Scalar(0, 0, 255), 3);
					circle(eye_image, Point(bottomCornerLeftEye.x, bottomCornerLeftEye.y), 1, Scalar(0, 0, 255), 3);*/

					// showing the image of the eye region only.
					//imshow("Eye" + (j + 1), eye_image);
					
					// drawing the corners in the images representing the whole face.
					/*circle(frame, Point(leftCornerLeftEye.x + faces[face_index].x + eyes[j].x, leftCornerLeftEye.y + faces[face_index].y + eyes[j].y), 2, Scalar(255, 0, 0), 3);
					circle(frame, Point(rightCornerLeftEye.x + (faces)[face_index].x + eyes[j].x, rightCornerLeftEye.y + faces[face_index].y + eyes[j].y), 2, Scalar(255, 0, 0), 3);
					circle(frame, Point(topCornerLeftEye.x + faces[face_index].x + eyes[j].x, topCornerLeftEye.y + faces[face_index].y + eyes[j].y), 2, Scalar(255, 0, 0), 3);
					circle(frame, Point(bottomCornerLeftEye.x + faces[face_index].x + eyes[j].x, bottomCornerLeftEye.y + faces[face_index].y + eyes[j].y), 2, Scalar(255, 0, 0), 3);*/

					/// ******************************************************************************************************************** ///
					/// ----------------------------------------------eyebrows detection---------------------------------------------------- ///
					/// ******************************************************************************************************************** ///
						
					// resetting the variables again for the next eyebrow
					minEyebrowX = 1000; maxEyebrowX = -1; minEyebrowY = 1000; maxEyebrowY = -1;
					leftCornerEyebrow.x = 0; leftCornerEyebrow.y = 0;
					rightCornerEyebrow.x = 0; rightCornerEyebrow.y = 0;
					topCornerEyebrow.x = 0; topCornerEyebrow.y = 0;
					bottomCornerEyebrow.x = 0; bottomCornerEyebrow.y = 0;
					
					// getting the rectangles surrounding the eyebrows manually from that of the eyes,
					//  since there is no such a classifier in OpenCV that detects the eyebrows directly.
					if (faces.size() > 0) {
						// the rectangle coordinates depend on whether the eye is the left\right one.
						if (faces[face_index].x + eyes[j].x < frame.cols / 2) {
							eyebrowX = faces[face_index].x + eyes[j].x - 20;
							eyebrowY = faces[face_index].y + (eyes[j].y - (eyes[j].height) / 2) - 10;
							eyebrowWidth = floor(((double)eyes[j].width * 1.75) + 0.5);
							eyebrowHeight = (eyes[j].height * 4) / 5;
						}
						else {
							eyebrowX = faces[face_index].x + eyes[j].x - 10;
							eyebrowY = faces[face_index].y + (eyes[j].y - (eyes[j].height) / 2) - 10;
							eyebrowWidth = floor(((double)eyes[j].width * 1.75) + 0.5);
							eyebrowHeight = (eyes[j].height * 4) / 5;
						}
					}

					eyebrow_rectangle = Rect(eyebrowX, eyebrowY, eyebrowWidth, eyebrowHeight);
					Mat clone_frame = frame.clone();
					eyebrow_image = clone_frame(eyebrow_rectangle);
					cvtColor(eyebrow_image, eyebrow_image, CV_BGR2GRAY);

					//rectangle(frame, eyebrow_rectangle, Scalar(0, 0, 255), 2);		// uncomment to draw the rectangles surrounding the eyebrows.

					// eyebrows are detected by:
					// first --> getting the number of all the intensities (shades of gray) present in an eyeborw image.
					// second --> choosing a different percentage according to the no. of different shades of gray.
					// third --> using this percentage to get the lowest intensities accordingly.
					std::vector<int> intensity;
					intensity.resize(256);

					std::vector<Point> eyebrows_points;
					for (int e = 0; e < eyebrow_image.rows; e++) {
						for (int f = 0; f < eyebrow_image.cols; f++) {
							Scalar colour = eyebrow_image.at<uchar>(Point(f, e));
							intensity[colour[0]]++;
						}
					}

					int unique_intensities = 0;
					std::vector<int> intensity_clone;
					intensity_clone = intensity;
					sort(intensity_clone.begin(), intensity_clone.end());
					for (int p = 0; p < intensity_clone.size() - 1; p++) {
						if (intensity_clone[p] != 0) {
							unique_intensities++;
						}
					}
					
					int eyebrow_intensity;
					if (unique_intensities < 200) {
						eyebrow_intensity = (int) ((double) unique_intensities * 0.3);
					}
					else {
						eyebrow_intensity = (int) ((double) unique_intensities * 0.5);
					}

					int threshold_intensity = 0;
					for (int o = 0; o < intensity.size(); o++) {
						if (intensity[o] != 0) {
							eyebrow_intensity--;
						}
						if (eyebrow_intensity == 0) {
							threshold_intensity = o;
							break;
						}
					}

					for (int e = 0; e < eyebrow_image.rows; e++) {
						for (int f = 0; f < eyebrow_image.cols; f++) {
							Scalar colour = eyebrow_image.at<uchar>(Point(f, e));
							if (colour[0] <= threshold_intensity) {
								eyebrows_points.push_back(Point(f, e));
							}
						}
					}

					// getting, initially, four points representing the eyebrow.
					// note --> the top and bottom points will be combined into one point representing the center
					// of the eyebrow.
					for (int o = 0; o < eyebrows_points.size(); o++) {
						if (eyebrows_points[o].x <= minEyebrowX && eyebrows_points[o].x > 10) {
							if (eyebrows_points[o].x < minEyebrowX || (eyebrows_points[o].x == minEyebrowX && eyebrows_points[o].y > leftCornerEyebrow.y)) {
								minEyebrowX = eyebrows_points[o].x;
								leftCornerEyebrow = eyebrows_points[o];
							}
						}

						if (eyebrows_points[o].x >= maxEyebrowX && eyebrows_points[o].x < eyebrow_image.cols - 25) {
							if ((eyebrows_points[o].y > leftCornerEyebrow.y && eyebrows_points[o].x == maxEyebrowX) 
								|| eyebrows_points[o].x > maxEyebrowX) {
								maxEyebrowX = eyebrows_points[o].x;
								rightCornerEyebrow = eyebrows_points[o];
							}
						}
							
						if (eyebrows_points[o].y <= minEyebrowY && eyebrows_points[o].x > ((double)eyebrow_image.cols / 2) - 5
							&& eyebrows_points[o].x < ((double)eyebrow_image.cols / 2) + 5 && eyebrows_points[o].y > 10) {
								minEyebrowY = eyebrows_points[o].y;
								topCornerEyebrow = eyebrows_points[o];
						}

						if (eyebrows_points[o].y >= maxEyebrowY && eyebrows_points[o].x > ((double)eyebrow_image.cols / 2) - 5
							&& eyebrows_points[o].x < ((double)eyebrow_image.cols / 2) + 5 && eyebrows_points[o].y > 10) {
								maxEyebrowY = eyebrows_points[o].y;
								bottomCornerEyebrow = eyebrows_points[o];
						}
					}

					// adjusting the coordinates of the detected points.
					if (rightCornerEyebrow.y < topCornerEyebrow.y) {
						rightCornerEyebrow.y = leftCornerEyebrow.y;
					}

					if (leftCornerEyebrow.y < topCornerEyebrow.y) {
						leftCornerEyebrow.y = rightCornerEyebrow.y;
					}

					if (topCornerEyebrow.y == 0) {
						topCornerEyebrow.y = leftCornerEyebrow.y;
					}


					topCornerEyebrow.x = (double) ((rightCornerEyebrow.x - leftCornerEyebrow.x) / 2) + leftCornerEyebrow.x;
					bottomCornerEyebrow.x = (double) ((rightCornerEyebrow.x - leftCornerEyebrow.x) / 2) + leftCornerEyebrow.x;
					if (bottomCornerEyebrow.y - topCornerEyebrow.y < 17) {
						topCornerEyebrow.y = topCornerEyebrow.y + (double) (bottomCornerEyebrow.y - topCornerEyebrow.y) / 2;
					}
					else {
						topCornerEyebrow.y += 3;
					}

					// drawing the detected points wrt eyebrows' rectangle.
					circle(eyebrow_image, leftCornerEyebrow, 2, Scalar(0, 0, 0), 2);
					circle(eyebrow_image, rightCornerEyebrow, 2, Scalar(0, 0, 0), 2);
					circle(eyebrow_image, topCornerEyebrow, 2, Scalar(0, 0, 0), 2);
					imshow("E " + j, eyebrow_image);


					// checking if the first detected eye is the left one or the right one and assigning t
					// he correct values to the corresponding variables.
					if (eyes[j].x < + faces[face_index].width / 2) {

						// left eye
						left_eye = true;

						left_corner_left_eye.x = leftCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						left_corner_left_eye.y = leftCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						top_corner_left_eye.x = topCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						top_corner_left_eye.y = topCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						right_corner_left_eye.x = rightCornerLeftEye.x + (faces)[face_index].x + eyes[j].x;
						right_corner_left_eye.y = rightCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						bottom_corner_left_eye.x = bottomCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						bottom_corner_left_eye.y = bottomCornerLeftEye.y + faces[face_index].y + eyes[j].y;

						// left eyebrow
						left_corner_left_eyebrow_whole.x = leftCornerEyebrow.x + eyebrowX;
						left_corner_left_eyebrow_whole.y = leftCornerEyebrow.y + eyebrowY;
						top_corner_left_eyebrow_whole.x = topCornerEyebrow.x + eyebrowX;
						top_corner_left_eyebrow_whole.y = topCornerEyebrow.y + eyebrowY;
						bottom_corner_left_eyebrow_whole.x = bottomCornerEyebrow.x + eyebrowX;
						bottom_corner_left_eyebrow_whole.y = bottomCornerEyebrow.y + eyebrowY;
						right_corner_left_eyebrow_whole.x = rightCornerEyebrow.x + eyebrowX;
						right_corner_left_eyebrow_whole.y = rightCornerEyebrow.y + eyebrowY;
					}
					else {
						// right eye
						right_eye = true;

						left_corner_right_eye.x = leftCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						left_corner_right_eye.y = leftCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						top_corner_right_eye.x = topCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						top_corner_right_eye.y = topCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						right_corner_right_eye.x = rightCornerLeftEye.x + (faces)[face_index].x + eyes[j].x;
						right_corner_right_eye.y = rightCornerLeftEye.y + faces[face_index].y + eyes[j].y;
						bottom_corner_right_eye.x = bottomCornerLeftEye.x + faces[face_index].x + eyes[j].x;
						bottom_corner_right_eye.y = bottomCornerLeftEye.y + faces[face_index].y + eyes[j].y;

						// right eyebrow
						left_corner_right_eyebrow_whole.x = leftCornerEyebrow.x + eyebrowX;
						left_corner_right_eyebrow_whole.y = leftCornerEyebrow.y + eyebrowY;
						top_corner_right_eyebrow_whole.x = topCornerEyebrow.x + eyebrowX;
						top_corner_right_eyebrow_whole.y = topCornerEyebrow.y + eyebrowY;
						bottom_corner_right_eyebrow_whole.x = bottomCornerEyebrow.x + eyebrowX;
						bottom_corner_right_eyebrow_whole.y = bottomCornerEyebrow.y + eyebrowY;
						right_corner_right_eyebrow_whole.x = rightCornerEyebrow.x + eyebrowX;
						right_corner_right_eyebrow_whole.y = rightCornerEyebrow.y + eyebrowY;
					}
					//rectangle(frame, eye_rectangle, Scalar(255, 0, 0), 2, 8);		// uncomment to draw rectangles surrounding the eyes.
				}
					
			}
			else {
				cout << "More than two eyes are detected." << endl;
			}
		}
		
		// adjusting the coordinates of one of the corners if it is misplaced.
		if (left_eye && right_eye) {
			if (top_corner_left_eyebrow_whole.y - top_corner_right_eyebrow_whole.y >= 20) {
				double average_eyebrows_y = (double) (top_corner_left_eyebrow_whole.y + left_corner_left_eyebrow_whole.y + right_corner_left_eyebrow_whole.y 
					+ left_corner_right_eyebrow_whole.y + right_corner_right_eyebrow_whole.y) / 5;
				top_corner_right_eyebrow_whole.y = average_eyebrows_y;
				cout << "Top left modified" << endl;
			}
			else {
				if (top_corner_right_eyebrow_whole.y - top_corner_left_eyebrow_whole.y >= 20) {
					double average_eyebrows_y = (double) (top_corner_right_eyebrow_whole.y + left_corner_left_eyebrow_whole.y + right_corner_left_eyebrow_whole.y 
						+ left_corner_right_eyebrow_whole.y + right_corner_right_eyebrow_whole.y) / 5;
					top_corner_left_eyebrow_whole.y = average_eyebrows_y;
					cout << "Top right modified" << endl;
				}
			}
		}
		
		// drawing the points of interest wrt whole frame.
		if (right_eye) {
			circle(frame, Point(left_corner_right_eyebrow_whole.x, left_corner_right_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
			circle(frame, Point(right_corner_right_eyebrow_whole.x, right_corner_right_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
			circle(frame, Point(top_corner_right_eyebrow_whole.x, top_corner_right_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
		}

		if (left_eye) {
			circle(frame, Point(left_corner_left_eyebrow_whole.x, left_corner_left_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
			circle(frame, Point(right_corner_left_eyebrow_whole.x, right_corner_left_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
			circle(frame, Point(top_corner_left_eyebrow_whole.x, top_corner_left_eyebrow_whole.y), 1, Scalar(0, 0, 255), 2);
		}
		
		
		/// ***************************************************************************************************************************** ///
		/// -----------------------------------------------mouth detection--------------------------------------------------------------- ///
		/// ***************************************************************************************************************************** ///
	
		// variables that hold the rectangle and the image containing only the mouth.
		Rect mouth_rectangle;
		Mat mouth_image;

		// variables holding the imfo about the mouth's contours.
		Mat canny_output_mouth;
		std::vector<std::vector<Point>> contours_mouth;
		std::vector<Vec4i> hierarchy_mouth;
		
		// getting the lower half of the face to detect the mouth.
		Rect lower_half_face = Rect(0, faceROI.rows - (faceROI.rows / 3), faceROI.cols, faceROI.rows / 3);
		Mat lower_half_face_image(faceROI, lower_half_face);

		// detecting the mouths present within the detecetd face.
		mouth_cascade.detectMultiScale(lower_half_face_image, mouth, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		if (mouth.size() == 0) {
			cout << "No mouth is detected." << endl;
		}
		else {
			if (mouth.size() != 1) {
				// to get the largest mouth detected.
				int max_width_mouth = -1;
				for (int i = 0; i < mouth.size(); i++) {
					if (mouth[i].width > max_width_mouth) {
						max_width_mouth = mouth[i].width;
						mouth_index = i;
					}
				}
			}

			//cout << "No. of mouths detected: " << mouth.size() << endl;		// uncomment to get the no. of mouths being deteceted.	

			// getting the rectangle surrounding the mouth.
			mouth_rectangle = Rect(faces[face_index].x + mouth[mouth_index].x, faces[face_index].y + mouth[mouth_index].y 
				+ faceROI.rows - (faceROI.rows / 3), mouth[mouth_index].width, mouth[mouth_index].height);
			mouth_image = frame(mouth_rectangle);
			//rectangle(frame, mouth_rectangle, Scalar(255, 0, 0), 2, 8);		// uncomment to draw the rectangle surrounding the mouth

			// convert the colored frame into gray-scale and equalizing the histogram of the mouth region.
			cvtColor(mouth_image, mouth_image, CV_RGB2GRAY);
			equalizeHist(mouth_image, mouth_image);

			// finding edges and contours oof the mouth region.
			Canny(mouth_image, canny_output_mouth, thresh, thresh * 3, 3);
			findContours(canny_output_mouth, contours_mouth, hierarchy_mouth, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


			// same algorithm as that of the eyes detection is being used.
			// after getting the contours, average arrays are calculated and according to 
			// a certain threshold, contours that donot belong to the mouth are eliminated.

			// array of averages for contours.
			std::vector<double> average_contours_mouth;

			// Calc. average of each row of contours.
			for (int s = 0; s < contours_mouth.size(); s++) {
				double av = 0;
				for (int h = 0; h < contours_mouth[s].size(); h++) {
					av += contours_mouth[s][h].y;
				}
				av /= contours_mouth[s].size();
 				average_contours_mouth.push_back(av);
			}

			
			int lowerLimit = 0;
			if (contours_mouth.size() > 4) {
				int upper_limit = contours_mouth.size() / 2;
				for (int b = 0; b <= upper_limit; b++) {
					if (abs(average_contours_mouth[b] - average_contours_mouth[b + 1]) >= 5 && average_contours_mouth[b] > 36) {
						eleminateChin = true;
						lowerLimit = b + 1;
						break;
					}
				}
			}
			
			int max_diff = -1;
			int lower_limit_tmp = 0;
			for (int j = 0; j < contours_mouth.size() - 1; j++) {
				if (abs(average_contours_mouth[j] - average_contours_mouth[j + 1]) >= max_diff && average_contours_mouth[j] < 47
				&& average_contours_mouth[j] > 35) {
					max_diff = abs(average_contours_mouth[j] - average_contours_mouth[j + 1]);
					lower_limit_tmp = j + 1;
				}
			}
			
			if (lowerLimit != 0) {
				if (max_diff >= 12 && abs(average_contours_mouth[lowerLimit] - average_contours_mouth[lowerLimit - 1]) < 6) {
					eleminateChin = true;
					lowerLimit = lower_limit_tmp;
				}			
			}

			int start_mouth = 0;
			if (eleminateChin) {
				start_mouth = lowerLimit;
			}

			// draw the contours in the mouth region (for debugging).
			for (int k = start_mouth; k < contours_mouth.size(); k++) {
				drawContours(mouth_image, contours_mouth, k, Scalar(255, 255, 255), 2, 8, hierarchy_mouth, 0, Point());
			}

			// getting the coordinates of the 4 corners of the mouth.
			for (int m = start_mouth; m < contours_mouth.size(); m++) {
				for (int n = 0; n < contours_mouth[m].size(); n++) {
					minMouthX = contours_mouth[m][n].x < minMouthX ? contours_mouth[m][n].x : minMouthX;
					leftCornerMouth = contours_mouth[m][n].x <= minMouthX ? contours_mouth[m][n] : leftCornerMouth;
					
					maxMouthX = contours_mouth[m][n].x > maxMouthX ? contours_mouth[m][n].x : maxMouthX;
					rightCornerMouth = contours_mouth[m][n].x >= maxMouthX ? contours_mouth[m][n] : rightCornerMouth;
					
					if (contours_mouth[m][n].x > (mouth_image.cols / 2) - (mouth_image.cols / 6) && contours_mouth[m][n].x < (mouth_image.cols / 2) + (mouth_image.cols / 6)) {
						minMouthY = contours_mouth[m][n].y < minMouthY ? contours_mouth[m][n].y : minMouthY;
						topCornerMouth = contours_mouth[m][n].y <= minMouthY ? contours_mouth[m][n] : topCornerMouth;
						maxMouthY = contours_mouth[m][n].y >= maxMouthY ? contours_mouth[m][n].y : maxMouthY;
						bottomCornerMouth = contours_mouth[m][n].y >= maxMouthY ? contours_mouth[m][n] : bottomCornerMouth;
					}
				}
			}
			
			// adjusting the coordinates of the obtained 4 corners of the mouth.
			topCornerMouth.x = minMouthX + (maxMouthX - minMouthX) / 2;
			bottomCornerMouth.x = minMouthX + ((maxMouthX - minMouthX) / 2);
			rightCornerMouth.y = ((bottomCornerMouth.y - topCornerMouth.y) / 2) + topCornerMouth.y;
			leftCornerMouth.y = ((bottomCornerMouth.y - topCornerMouth.y) / 2) + topCornerMouth.y;
			
			// getting data for the coordinates with respect to the whole face.
			left_corner_mouth_whole.x = leftCornerMouth.x + faces[face_index].x + mouth[mouth_index].x;
			left_corner_mouth_whole.y = leftCornerMouth.y + faces[face_index].y + mouth[mouth_index].y + faceROI.rows - (faceROI.rows / 3);
			top_corner_mouth_whole.x = topCornerMouth.x + faces[face_index].x + mouth[mouth_index].x;
			top_corner_mouth_whole.y = topCornerMouth.y + faces[face_index].y + mouth[mouth_index].y + faceROI.rows - (faceROI.rows / 3);
			right_corner_mouth_whole.x = rightCornerMouth.x + faces[face_index].x + mouth[mouth_index].x;
			right_corner_mouth_whole.y = rightCornerMouth.y + faces[face_index].y + mouth[mouth_index].y + faceROI.rows - (faceROI.rows / 3);
			bottom_corner_mouth_whole.x = bottomCornerMouth.x + faces[face_index].x + mouth[mouth_index].x;
			bottom_corner_mouth_whole.y =  bottomCornerMouth.y + faces[face_index].y + mouth[mouth_index].y + faceROI.rows - (faceROI.rows / 3);

			// drawing the corners on the mouth region only and showing the mouth region of interest (for debugging).
			circle(mouth_image, leftCornerMouth, 1, Scalar(0, 255, 255), 3);
			circle(mouth_image, rightCornerMouth, 1, Scalar(0, 255, 255), 3);
			circle(mouth_image, topCornerMouth, 1, Scalar(0, 255, 255), 3);
			circle(mouth_image, bottomCornerMouth, 1, Scalar(0, 255, 255), 3);
			imshow("Mouth", mouth_image);
		}


		/// ******************************************************************************************************************** ///
		/// -----------------------------------------------nose detection------------------------------------------------------- ///
		/// ******************************************************************************************************************** ///

		// varibles to hold the noses detected and the index of the used nose.
		std::vector<Rect> nose;
		int nose_selected = 0;

		// detecting the nose using the pre-defined classifier.
		nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		if (nose.size() == 0) {
			cout << "No noses are detected" << endl;
		}
		else {
			//cout << "Noses detected: " << nose.size() << endl;		// uncomment to get the no.  of noses detected.
			if (nose.size() != 1) {
				// selecting the nose to be used.
				for (int j = 0; j < nose.size(); j++) {
					if (abs(((nose[j].width / 2) + nose[j].x) - (faces[face_index].width / 2)) < 15
						&& abs(((nose[j].height / 2) + nose[j].y) - (faces[face_index].height / 2)) < 70) {

						nose_selected = j;
						break;
					}
				}
			}

			// uncomment to draw the nose center wrt whole frame.
			/*circle(frame, Point(faces[face_index].x + nose[nose_selected].x + nose[nose_selected].width / 2,
				faces[face_index].y + nose[nose_selected].y + nose[nose_selected].height / 2), 1, Scalar(0, 0, 255), 2,  8);*/

			// getting the coordinates of the center of the nose being used.
			nose_center_whole.x = faces[face_index].x + nose[nose_selected].x + nose[nose_selected].width / 2;
			nose_center_whole.y = faces[face_index].y + nose[nose_selected].y + nose[nose_selected].height / 2;
			//imshow("Nose", nose_image);		// uncomment for the nose image to be shown.
		}
	}
	else {
		cout << "No faces are detected." << endl;
		// retruning an empty vector since there are no parameter to be passed to
		// the machine learning algorithm.
		std::vector<double> tmp;
		return tmp;
	}

	/// ******************************************************************************************************************************* ///
	/// ---------------------------------------------machine learning algoruthm parameters--------------------------------------------- ///
	/// ******************************************************************************************************************************* ///

	// the two parameters used for the training of the eyebrows classifier.
	// 1. the height formed by the top corner of the eyebrow and center of eye of one eye multiplied by that
	//	  of that of the other eye and divided by the face's height.
	// 2. the angles incribed at the top corner of the eyebrow of the left one multiplied by
	//    that of the right eye.
	double top_center_eye_left = (double) abs(eye_center_left.y - top_corner_left_eyebrow_whole.y);
	double top_center_eye_right = (double) abs(eye_center_right.y - top_corner_right_eyebrow_whole.y);
	if (left_eye && !right_eye) {
		ml_parameters.push_back((top_center_eye_left * top_center_eye_left) / faces[face_index].height);	
	}
	else {
		if (right_eye && !left_eye) {
			ml_parameters.push_back((top_center_eye_right * top_center_eye_right) / faces[face_index].height);
		}
		else {
			ml_parameters.push_back((top_center_eye_left * top_center_eye_right) / faces[face_index].height);
		}
	}
	
	double p12 = sqrt(pow((top_corner_left_eyebrow_whole.x - left_corner_left_eyebrow_whole.x), 2.0f) + 
		pow((top_corner_left_eyebrow_whole.y - left_corner_left_eyebrow_whole.y), 2.0f));
	double p13 = sqrt(pow((top_corner_left_eyebrow_whole.x - right_corner_left_eyebrow_whole.x), 2.0f) + 
		pow((top_corner_left_eyebrow_whole.y - right_corner_left_eyebrow_whole.y), 2.0f));
	double p23 = sqrt(pow((left_corner_left_eyebrow_whole.x - right_corner_left_eyebrow_whole.x), 2.0f) + 
		pow((left_corner_left_eyebrow_whole.y - right_corner_left_eyebrow_whole.y), 2.0f));

	double angle_rad = acos((pow(p12, 2) + pow(p13, 2) - pow(p23, 2)) / (2 * p12 * p13));
	double angle_deg_left = (angle_rad / 3.14159) * 180;

	p12 = sqrt(pow((top_corner_right_eyebrow_whole.x - left_corner_right_eyebrow_whole.x), 2.0f) + 
		pow((top_corner_right_eyebrow_whole.y - left_corner_right_eyebrow_whole.y), 2.0f));
	p13 = sqrt(pow((top_corner_right_eyebrow_whole.x - right_corner_right_eyebrow_whole.x), 2.0f) + 
		pow((top_corner_right_eyebrow_whole.y - right_corner_right_eyebrow_whole.y), 2.0f));
	p23 = sqrt(pow((left_corner_right_eyebrow_whole.x - right_corner_right_eyebrow_whole.x), 2.0f) + 
		pow((left_corner_right_eyebrow_whole.y - right_corner_right_eyebrow_whole.y), 2.0f));

	angle_rad = acos((pow(p12, 2) + pow(p13, 2) - pow(p23, 2)) / (2 * p12 * p13));
	double angle_deg_right = (angle_rad / 3.14159) * 180;

	if (angle_deg_left == 0) {
		ml_parameters.push_back(angle_deg_right * angle_deg_right);
	}
	else {
		if (angle_deg_right == 0) {
			ml_parameters.push_back(angle_deg_left * angle_deg_left);
		}
		else {
			ml_parameters.push_back(angle_deg_left * angle_deg_right);
		}
	}
	
	// the two parameters used for the training of the mouth classifier.
	// 1. the area of the triangle formed by the top, bottom and right corners of the mouth.
	// 2. the height formed by the y-coordinates of the top and bottom points wrt face's height.
	double area;
	double height_ratio;

	if (faces.size() != 0) {
		area = (0.5 * (double) abs(((top_corner_mouth_whole.x - bottom_corner_mouth_whole.x) * (right_corner_mouth_whole.y - 
			top_corner_mouth_whole.y)) - ((top_corner_mouth_whole.x - right_corner_mouth_whole.x) * 
			(bottom_corner_mouth_whole.y - top_corner_mouth_whole.y)))) / (faces[face_index].area());
		height_ratio = (double)(bottom_corner_mouth_whole.y - top_corner_mouth_whole.y) / faces[face_index].height;
	}

	ml_parameters.push_back(height_ratio);
	ml_parameters.push_back(area);
	
	return ml_parameters;
}


// generateSampleData() method adds the new data to the csv file that contains data of 
// previous samples and labels each newly entered sample, before passing the data to 
// the machine learning algorithm to train.

// note: one cannot directly add new data to to csv files in c++, so the only way is to
// save old data in a data structure and then adding the new data before writing back to the csv file.
void generateSampleData(std::vector<double> mouth_samples, string label) {

	// the name of the file should be changed accordingly.
	ifstream my_file("C:/Users/Omar/Desktop/eyebrows.csv");
	string line;
	std::vector<string> data;

	// saving the already present data in the csv file, to avoid overwriting the old data.
	if (my_file.is_open()) {
		while (getline(my_file, line)) {
			data.push_back(line);
		}
	}
	else {
		cout << "*************Unable to open the file************" << endl;
		return;
	}

	// adding the newly passed data to the old ones in a stringstream before writing it back
	// to the csv file.
	outLine.str(std::string());
	for (int i = 0; i < mouth_samples.size(); i++) {
		outLine << mouth_samples[i] << ", ";
		if (i == mouth_samples.size() - 1) {
			outLine << label;
		}
	}

	newLine = outLine.str();
	data.push_back(newLine);

	// the name of the file should be changed according to the input file.
	ofstream out_file("C:/Users/Omar/Desktop/eyebrows.csv");

	// writing back the data to the file.
	for (int i = 0; i < data.size(); i++) {
		out_file << data[i] << endl;
	}
	out_file.close();
} 
