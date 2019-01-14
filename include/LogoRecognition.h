#ifndef LOGORECOGNITION_H
#define LOGORECOGNITION_H

// #include "opencv2/core/core.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define sat(res) (std::min(std::max((int)round(res),0),255))

struct Moments
{
	double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
	Moments():
		m00(0.0), 
		m10(0.0), 
		m01(0.0), 
		m20(0.0), 
		m11(0.0), 
		m02(0.0), 
		m30(0.0), 
		m21(0.0), 
		m12(0.0), 
		m03(0.0)
	{}
};

// Moments* moments(const cv::Mat& I, cv::Vec3b color);
Moments* moments(const cv::Mat& I);

int pole(const cv::Mat& I);
int obwod(const cv::Mat& I);
float ksztaltMalinowskiej(float s, float l);
void center(double m00, double m10, double m01,	double& i, double& j);
// double niezmiennikM3(const double& i, const double& j, Moments* m);
double niezmiennikM7(const double& i, const double& j, Moments* m);
double niezmiennikM1(const double& i, const double& j, Moments* m);
void draw_rect(cv::Mat& I, int min_x, int min_y, int max_x, int max_y, 
	cv::Vec3b rectCol = {128,128,128});
int* get_surrounding_box(const cv::Mat& I);
cv::Mat rankFilter(const cv::Mat& src, int ksize, int rank);
void adjustConstrast(cv::Mat& image, float value);

#endif