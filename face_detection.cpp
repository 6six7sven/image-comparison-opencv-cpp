#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp> 
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	/*
	* 
	* for video capture webcam(doesnt work)
	* 
	VideoCapture cap(0);
	Mat img;
	while (true)
	{
		CascadeClassifier faceCascade;
		faceCascade.load("Resources/models/haarcascade_frontalface_default.xml");

		if (faceCascade.empty())
		{
			cout << "XML file not loaded" << endl;
		}

		//storing bounding boxes
		vector<Rect> faces;
		faceCascade.detectMultiScale(img, faces, 1.1, 10);
		//iterate thru detected faces
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 2);
		}
		cap.read(img);
		imshow("Image", img);
		waitKey(200);
	}
	*/
	
	string path = "Resources/models/test.png";
	Mat img = imread(path);
	CascadeClassifier faceCascade;
	faceCascade.load("Resources/models/haarcascade_frontalface_default.xml");

	if (faceCascade.empty())
	{
		cout << "XML file not loaded" << endl;
	}

	//storing bounding boxes
	vector<Rect> faces;
	faceCascade.detectMultiScale(img, faces, 1.1, 10);
	//iterate thru detected faces
	for (int i = 0; i<faces.size(); i++)
	{
		rectangle(img, faces[i].tl(),faces[i].br(), Scalar(255, 255, 255), 2);
	}

	imshow("Image", img);
	waitKey(0);
	
	return 0;
}