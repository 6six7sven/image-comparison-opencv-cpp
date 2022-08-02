#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//window to show images
const char* image_window = "Source Image";
const char* result_window = "Result Window";

int match_method = 1; //value 1 for matching method

//matrix = n dim array 4 multi channel array
Mat src_base, src_test1, src_test2;
Mat hsv_base, hsv_test1, hsv_test2;
Mat result;

void match_template(Mat img, Mat templ)
{
	img = src_base;
	templ = src_test1;
	Mat img_display;
	img.copyTo(img_display);

	if (src_base.empty() || src_test1.empty() || src_test2.empty())
	{
		cout << "Could not open images..." << endl;
		return;
	}

	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	matchTemplate(img, templ, result, match_method);
	cout << "Matching..." << endl;

	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	cout << "Normalizing..." << endl;

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	//defining method of matching:
	//For the first two methods ( TM_SQDIFF and MT_SQDIFF_NORMED ) the best match are the lowest values. 
	//For all the others, higher values represent better matches. 
	//So, we save the corresponding value in the matchLoc variable:
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		cout << "Using minloc" << endl;
	}
	else
	{
		matchLoc = maxLoc;
		cout << "Using maxloc" << endl;
	}
	
	//draws rectangle over the match
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);
	//waitKey(0);
}

void histogram_comparison(Mat src, Mat test1, Mat test2)
{
	src = src_base;
	test1 = src_test1;
	test2 = src_test2;

	//converting to hsv format (hue saturation value)
	cvtColor(src, hsv_base, COLOR_BGR2HSV);
	cvtColor(test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(test2, hsv_test2, COLOR_BGR2HSV);

	// image of half the base image (in HSV format):
	Mat hsv_half_down = hsv_base(Range(hsv_base.rows / 2, hsv_base.rows), Range(0, hsv_base.cols));

	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// Use the 0-th and 1-st channels
	int channels[] = { 0, 1 };

	//Calculate the Histograms for the base image, the 2 test images and the half-down base image:
	Mat hist_base, hist_half_down, hist_test1, hist_test2;
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false);
	normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());

	//Apply sequentially the 4 comparison methods between the histogram of the base image (hist_base) 
	//and the other histograms:
	for (int compare_method = 0; compare_method < 4; compare_method++)
	{
		double base_base = compareHist(hist_base, hist_base, compare_method);
		double base_half = compareHist(hist_base, hist_half_down, compare_method);
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
		double base_test2 = compareHist(hist_base, hist_test2, compare_method);
		cout << "Method " << compare_method << " Perfect, Base-Half, Base-Test(1), Base-Test(2) : "
			<< base_base << " / " << base_half << " / " << base_test1 << " / " << base_test2 << endl;
	}
}

void image_subtraction(Mat src1, Mat src2, Mat res)
{
	src1 = src_base;
	src2 = src_test1;
	res = result;

	cvtColor(src1, hsv_base, COLOR_BGR2HSV);
	cvtColor(src2, hsv_test1, COLOR_BGR2HSV);

	subtract(hsv_base, hsv_test1, res);

	imshow("Result", res);

	//waitKey(0);
}

void histogram_calculation(Mat src, Mat test1)
{
	src = src_base;
	test1 = src_test1;
	if (src.empty()&& src_test1.empty())
	{
		return;
	}
	vector<Mat> bgr_planes1;
	vector<Mat> bgr_planes2;
	split(src, bgr_planes1);
	split(test1, bgr_planes2);

	int histSize = 256;

	float range[] = { 0,256 }; 
	const float* histRange[] = { range };

	bool uniform = true, accumulate = false;

	Mat b_hist, g_hist, r_hist, b_hist1, g_hist1, r_hist1;
	calcHist(&bgr_planes1[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes2[0], 1, 0, Mat(), b_hist1, 1, & histSize, histRange, uniform, accumulate);

	calcHist(&bgr_planes1[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes2[1], 1, 0, Mat(), g_hist1, 1, &histSize, histRange, uniform, accumulate);

	calcHist(&bgr_planes1[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes2[2], 1, 0, Mat(), r_hist1, 1, &histSize, histRange, uniform, accumulate);

	

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	int hist_w1 = 512, hist_h1 = 400;
	int bin_w1 = cvRound((double)hist_w1 / histSize);
	Mat histImage1(hist_h1, hist_w1, CV_8UC3, Scalar(0, 0, 0));


	//normalizing so the values fall into the (0-255 color range)
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist1, b_hist1, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist1, g_hist1, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist1, r_hist1, 0, histImage.rows, NORM_MINMAX, -1, Mat());


	//loops to create graphs for each color channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage1, Point(bin_w1 * (i - 1), hist_h1 - cvRound(b_hist1.at<float>(i - 1))),
			Point(bin_w1 * (i), hist_h1 - cvRound(b_hist1.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);

		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage1, Point(bin_w1 * (i - 1), hist_h1 - cvRound(g_hist1.at<float>(i - 1))),
			Point(bin_w1 * (i), hist_h1 - cvRound(g_hist1.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);

		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage1, Point(bin_w1 * (i - 1), hist_h1 - cvRound(r_hist1.at<float>(i - 1))),
			Point(bin_w1 * (i), hist_h1 - cvRound(r_hist1.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("Source image", src);
	imshow("Test image", test1);
	imshow("calcHist Source", histImage);
	imshow("calcHist Test1", histImage1);
	waitKey(0);
}

int main()
{
	char choice;
	string path1 = "Resources/img1.jpg";
	string path2 = "Resources/img2.jpg";
	string path3 = "Resources/img-t.jpg";

	src_base = imread(path1, IMREAD_COLOR);
	src_test1 = imread(path2, IMREAD_COLOR);
	src_test2 = imread(path3, IMREAD_COLOR);

	if (src_base.empty() || src_test1.empty() || src_test2.empty())
	{
		cout << "Could not open images..." << endl;
		return -1;
	}

	cout << "Enter the comparison method: " << endl;
	cout << "(a) -------------Histogram Comparison-------" << endl;
	cout << "(b) -------------Match Template-------------" << endl;
	cout << "(c) -------------Image Subtraction----------" << endl; //size must be same
	cout << "(d) -------------Histogram Calculation------" << endl; 
	cout << "Enter the option: ";
	cin >> choice;
	switch (choice)
	{
	case 'a':
		cout << "\nPerforming Histogram Comparison..." << endl;
		histogram_comparison(src_base, src_test1, src_test2);
		break;
	case 'b':
		cout << "Performing Match Template..." << endl;
		imshow("Source 2", src_test1);
		match_template(src_base, src_test1);
		waitKey(0);
		break;
	case 'c':
		cout << "Performing Image Subtraction..." << endl;
		imshow("Source 2", src_test1);
		image_subtraction(src_base, src_test1, result);
		waitKey(0);
		break;
	case 'd':
		cout << "Performing Histogram Calculation..." << endl;
		histogram_calculation(src_base, src_test1); 
		waitKey(0);
	default:
		cout << "Invalid input..." << endl;
		break;
	}
	return 0;
}
