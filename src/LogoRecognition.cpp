#include "LogoRecognition.h"

int pole(cv::Mat& I, cv::Vec3b color){
	int counter = 0;
	cv::Mat res(I.rows, I.cols, CV_8UC3);
	cv::Vec3b gray = { 128, 128, 128 };
	for (int i = 0; i < I.rows; ++i)
		for (int j = 0; j < I.cols; ++j)
			if (I.at<cv::Vec3b>(i, j) == color){
				counter++;
			}
	return counter;
}

int obwod(cv::Mat& I, cv::Mat& res, cv::Vec3b color){
	int counter = 0;
	cv::Vec3b gray = { 128, 128, 128 };
	cv::Vec3b white = { 255,255,255 };
	for (int i = 1; i < I.rows-1; ++i)
		for (int j = 1; j < I.cols-1; ++j){
			if (I.at<cv::Vec3b>(i, j) == color)
				if ((I.at<cv::Vec3b>(i + 1, j) != color)
					|| (I.at<cv::Vec3b>(i - 1, j) != color)
					|| (I.at<cv::Vec3b>(i - 1, j+1) != color)
					|| (I.at<cv::Vec3b>(i - 1, j - 1) != color)
					|| (I.at<cv::Vec3b>(i + 1, j + 1) != color)
					|| (I.at<cv::Vec3b>(i + 1, j - 1) != color)
					|| (I.at<cv::Vec3b>(i, j + 1) != color)
					|| (I.at<cv::Vec3b>(i, j - 1) != color)){
					counter++;
					res.at<cv::Vec3b>(i, j) = gray;
				}
		}
	return counter;
}

float ksztaltMalinowskiej(float s, float l){
	return ((l/(2.0*sqrt(3.14*s)))-1.0);
}

Moments* moments(cv::Mat& I, cv::Vec3b color){
	cv::Mat_<cv::Vec3b> _I = I;
	Moments* m = new Moments();
	for (int i = 0; i < I.rows; ++i)
		for (int j = 0; j < I.cols; ++j){
			if (I.at<cv::Vec3b>(i, j) == color){
				m->m00 += 1.0;
				m->m01 += j*1.0;
				m->m10 += i*1.0;
				m->m20 += std::pow(i,2)*1.0;
				m->m11 += i*j*1.0;
				m->m02 += std::pow(j,2)*1.0;
				m->m30 += std::pow(i,3)*1.0;
				m->m03 += std::pow(j,3)*1.0;
				m->m12 += i*std::pow(j,2)*1.0;
				m->m21 += std::pow(i,2) * j*1.0;
			}
		}
	return m;
}

void center(double m00, double m10, double m01,
	double& i, double& j){
	i = m10 / m00;
	j = m01 / m00;
}

double niezmiennikM3(const double& i, const double& j, Moments* m){
	double M30 = m->m30 - 3.0 * m->m20*i + 2.0 * m->m10 *std::pow(i,2);
	double M03 = m->m03 - 3.0 * m->m02*j + 2.0 * m->m01*std::pow(j,2);
	double M21 = m->m21 - 2.0 * m->m11*i - m->m20*j + 2.0 * m->m01*std::pow(i,2);
	double M12 = m->m12 - 2.0 * m->m11*j - m->m02*i + 2.0 * m->m10*std::pow(j,2);
	return (std::pow((M30 - 3.0 * M12), 2) + std::pow((3.0 * M21 - M03), 2)) / std::pow(m->m00, 5);
}

double niezmiennikM7(const double& i, const double& j, Moments* m){
	double M02 = m->m02-std::pow(m->m01,2)/m->m00;
	double M20 = m->m20-std::pow(m->m10,2)/m->m00;
	double M11 = m->m11-m->m10*m->m01/m->m00;
	return (M20*M02-std::pow(M11,2))/std::pow(m->m00,4);
}

void draw_rect(cv::Mat& I, int min_x, int min_y, int max_x, int max_y, 
	cv::Vec3b rectCol){
	for (int x = 0; x < I.cols; ++x)
		for (int y = 0; y < I.rows; ++y){
			if((x==min_x || x==max_x) && (y>=min_y && y<=max_y))
				I.at<cv::Vec3b>(y, x) = rectCol;
			if((y==min_y || y==max_y) && (x>=min_x && x<=max_x))
				I.at<cv::Vec3b>(y, x) = rectCol;
		}
}

int* get_surrounding_box(cv::Mat& I, cv::Vec3b color){
	int max_x = 0, max_y = 0;
	int min_x = I.rows;
	int min_y = I.cols;
	for (int i = 0; i < I.rows; ++i)
		for (int j = 0; j < I.cols; ++j)
			if (I.at<cv::Vec3b>(i, j) == color){
				if (i < min_y) min_y = i;
				if (i > max_y) max_y = i;
				if (j < min_x) min_x = j;
				if (j > max_x) max_x = j;
			}
	int* t = new int[4];
	t[0] = min_x;
	t[1] = min_y ;
	t[2] = max_x ;
	t[3] = max_y;
	return t;
}