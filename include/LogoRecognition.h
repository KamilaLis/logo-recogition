#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
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

Moments* moments(cv::Mat& I, cv::Vec3b color = { 0, 0, 0 });

int pole(cv::Mat& I, cv::Vec3b color = {0,0,0});
int obwod(cv::Mat& I, cv::Mat& res, cv::Vec3b color = { 0, 0, 0 });
float ksztaltMalinowskiej(float s, float l);
void center(double m00, double m10, double m01,	double& i, double& j);
double niezmiennikM3(const double& i, const double& j, Moments* m);
double niezmiennikM7(const double& i, const double& j, Moments* m);

void draw_rect(cv::Mat& I, int min_x, int min_y, int max_x, int max_y, 
	cv::Vec3b rectCol = {128,128,128});
int* get_surrounding_box(cv::Mat& I, cv::Vec3b color = { 0, 0, 0 });