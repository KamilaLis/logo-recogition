#include <stdio.h>
#include <math.h> 
// #include <opencv2/opencv.hpp>
#include "LogoRecognition.h"


cv::Mat rgb2hsv(const cv::Mat& image){
    float hue, sat, val;
    float xd, f, i;
    float red, grn, blu;
    cv::Mat result = cv::Mat(image.clone());

    for(int y=0; y<image.rows; y++){ 
        for(int x=0; x<image.cols; x++){
            blu = image.at<cv::Vec3b>(y,x)[0];
            grn = image.at<cv::Vec3b>(y,x)[1];
            red = image.at<cv::Vec3b>(y,x)[2];

            xd = MIN(MIN(red, grn), blu);
            val = MAX(MAX(red, grn), blu);
            // if (xd == val){
            if (val == 0){
                hue = 0;
                sat = 0;
            }
            else {
                f = (red == xd) ? grn-blu : ((grn == xd) ? blu-red : red-grn);
                i = (red == xd) ? 3 : ((grn == xd) ? 5 : 1);
                hue = fmod((i-f/(val-xd))*60, 360);
                sat = ((val-xd)/val);
            }
            result.at<cv::Vec3b>(y,x)[0] = hue/2;
            result.at<cv::Vec3b>(y,x)[1] = sat*255;
            result.at<cv::Vec3b>(y,x)[2] = val*255;
        }
    }
    return result;
}

cv::Mat hsvTresholding(cv::Mat hsv_image, int h_min, int h_max,
                        int s_min, int s_max){
    cv::Mat p_img = cv::Mat::ones(hsv_image.rows, hsv_image.cols, CV_8UC1)*0;
    for(int y=0; y<hsv_image.rows; y++){
        for(int x=0; x<hsv_image.cols; x++){
            int h = hsv_image.at<cv::Vec3b>(y,x)[0];
            int s = hsv_image.at<cv::Vec3b>(y,x)[1];
            if (h_min > h_max){
                if ((h <= h_max || h >= h_min) && s<= s_max && s >= s_min)
                    p_img.at<uchar>(y,x) = 255;
            }
            else {
                if (h <= h_max && h >= h_min && s<= s_max && s >= s_min)
                    p_img.at<uchar>(y,x) = 255;
            }
        }
    }
    return p_img;
}

cv::Vec3b meanIntensity(cv::Mat image){
    int sum[] = {0,0,0};
    int pxs = image.rows*image.cols;
    for(int y=0; y<image.rows; y++)
        for(int x=0; x<image.cols; x++)
            for(int i=0; i<3; ++i){
                sum[i] += image.at<cv::Vec3b>(y,x)[i];
            }
    cv::Vec3b mean = cv::Vec3b();
    for(int i=0; i<3; ++i){
        mean[i]=sum[i]/pxs;
        std::cout<<int(mean[i])<<std::endl;
    }
    return mean;
}

// BGR tresholding
cv::Mat tresholding(cv::Mat image, cv::Vec3b color){
    cv::Mat p_img = cv::Mat::ones(image.rows, image.cols, CV_8UC1)*0;
    // (sy, sx, ch) = img.shape
    for(int y=0; y<image.rows; y++){
        for(int x=0; x<image.cols; x++){
            if(image.at<cv::Vec3b>(y, x)[2]>190//143 
                && image.at<cv::Vec3b>(y, x)[1]>156//165
                && image.at<cv::Vec3b>(y, x)[0]<160){//100)
                p_img.at<uchar>(y,x)=255;
            }
        }
    }
    return p_img;
}

// // Wyznacza, w którym zbiorze jest dany element, 
// // pozwalając na sprawdzenie, czy dwa elementy są w tym samym zbiorze.
// uchar find(uchar x)  {
//     uchar y = x;
//     while (labels[y] != y)
//         y = labels[y];
//     while (labels[x] != x)  {
//         uchar z = labels[x];
//         labels[x] = y;
//         x = z;
//     }
//     return y;
// }
std::vector<int> parent {0};

int find (int x){
    if (parent[x] == 0) return x;
    parent[x] = find(parent[x]);
    return parent[x];
}

void _union(int x, int y)  {
    parent[find(x)] = find(y);
}

// Connected-component labeling
cv::Mat twoPass(cv::Mat mask, int& max_label){
    uchar background = 0;
    max_label = 0;
    cv::Mat labels = cv::Mat::ones(mask.rows, mask.cols, CV_32SC1)*background;
    // first pass
    for(int y=1; y<mask.rows; y++){
        for(int x=1; x<mask.cols; x++){
            if (mask.at<uchar>(y, x) != background){
                // left is not background
                if (mask.at<uchar>(y-1, x) == mask.at<uchar>(y, x)){
                    labels.at<int>(y, x) = labels.at<int>(y-1, x);
                }
                // left and up is not back but have different labels
                else if (mask.at<uchar>(y-1, x)==mask.at<uchar>(y, x) 
                    && mask.at<uchar>(y, x-1)==mask.at<uchar>(y, x)
                    && labels.at<int>(y-1, x)!=labels.at<int>(y, x-1)){
                    labels.at<int>(y, x) = std::min(labels.at<int>(y-1, x),labels.at<int>(y, x-1));
                    // 2 -> 1
                    // parent[std::max(labels.at<int>(y-1, x),labels.at<int>(y, x-1))] = labels.at<int>(y, x);
                    _union(std::max(labels.at<int>(y-1, x),labels.at<int>(y, x-1)), 
                        labels.at<int>(y, x));
                }
                // up is not background
                else if (mask.at<uchar>(y-1, x) != mask.at<uchar>(y, x)
                    && mask.at<uchar>(y, x-1) == mask.at<uchar>(y, x)){
                    labels.at<int>(y, x) = labels.at<int>(y, x-1);
                }
                // left and up is background
                else{
                    max_label += 1;
                    labels.at<int>(y, x) = max_label;
                    parent.insert(parent.begin()+max_label, 0);
                }

            }
        }
    }
    // second pass
    for(int y=0; y<mask.rows; y++){
        for(int x=0; x<mask.cols; x++){
            if (mask.at<uchar>(y, x) != background){
                // std::cout<<labels.at<int>(y, x)<<std::endl;
                labels.at<int>(y, x) = find(labels.at<int>(y, x));
            }
        }
    }
    return labels;
}


cv::Mat labels;// = cv::Mat::ones(mask.rows, mask.cols, CV_8UC1)*background;
// depth-first-search
// Visit all the cells reachable from the starting cell. 
// Each cell will be marked with current_label
void dfs(int x, int y, int current_label, const cv::Mat& m) {
    // direction vectors
    const int dx[] = {+1, 0, -1, 0};
    const int dy[] = {0, +1, 0, -1};

    if (x < 0 || x == m.rows) return; // out of bounds
    if (y < 0 || y == m.cols) return; // out of bounds
    if (labels.at<int>(x,y)!=0 
        || m.at<uchar>(x,y)==0) return; // already labeled or not marked with 1 in m

    // mark the current cell
    labels.at<int>(x,y) = current_label;

    // recursively mark the neighbors
    for (int direction = 0; direction < 4; ++direction){
        dfs(x + dx[direction], y + dy[direction], current_label, m);
    }
}

cv::Mat labelComponents(const cv::Mat& mask, int& component){
    const uchar background = 0;
    labels = cv::Mat::zeros(mask.rows, mask.cols, CV_32SC1);
    component = 0;
    for (int i = 0; i < mask.rows; ++i) 
        for (int j = 0; j < mask.cols; ++j) {
            // std::cout<<"labelComponents: "<<i<<" ,"<<j<<std::endl;
            if (labels.at<int>(i,j)==int(background) 
                && mask.at<uchar>(i,j)!=background) {
                dfs(i, j, ++component, mask);
            }
        }

    return labels;
}

cv::Mat getObject(cv::Mat labels, int labelID){
    cv::Mat p_img = cv::Mat::ones(labels.rows, labels.cols, CV_8UC1)*0;
    for (int i = 0; i < labels.rows; ++i) 
        for (int j = 0; j < labels.cols; ++j) 
            if (labels.at<int>(i,j)==labelID)
                p_img.at<uchar>(i,j) = 255;
    return p_img;
}

std::vector<cv::Mat>
getAllObjects(cv::Mat labels, int no_labels){
    std::vector<cv::Mat> v;
    // from 1: ignore background
    for (int i = 1; i<no_labels; ++i){
        cv::Mat obj = getObject(labels, i);
        if (pole(obj) > 50)    // TODO: fix
        // if (pole(obj) > 10)
            v.push_back(obj);
        // v.push_back(obj);
    }
    return v;
}

bool isMetroYellowBackground(const cv::Mat& object){
    Moments* m = moments(object);
    double i=0, j=0;
    center(m->m00, m->m10, m->m01, i, j);
    double M7 = niezmiennikM7(i, j, m);
    // double M1 = niezmiennikM1(i, j, m);
    float mal = ksztaltMalinowskiej(pole(object), obwod(object));
    // std::cout<<M7<<", "<<mal<<std::endl;
    // if (M7 > 0.025 && M7 < 0.052 && mal > 2.2 && mal < 4.3){
    if (M7 > 0.017 && M7 < 0.052 && mal > 1.2 && mal < 4.3){
        // std::cout<<M7<<", "<<mal<<std::endl;
        return true;
    }
    return false;
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: logo-recognition <Image_Path>\n");
        return -1;
    }

    cv::Mat image;
    image = cv::imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // RGB -> HSV
    cv::Mat hsv = rgb2hsv(image); 

    // progowanie
    cv::Mat mask = hsvTresholding(hsv, 20, 31, 95, 255);//(hsv, 21, 30, 100, 255);
    cv::imshow("mask", mask);

    cv::Mat red_mask = hsvTresholding(hsv, 174, 15, 88, 255);//(hsv, 174, 15, 93, 255);
    cv::imshow("red_mask.jpg", red_mask);

    // usuniecie szumow
    int kernel = 3;
    cv::Mat dst = rankFilter(mask, kernel, kernel*kernel/2);
    cv::imshow("rankFilter.jpg", dst);
    
    // segmentacja
    int component;
    // cv::Mat labels = twoPass(dst, component);
    cv::Mat labels = labelComponents(dst, component);
    // std::cout<<"components: "<<component<<std::endl;
    ++component;
    std::vector<cv::Mat> objects = getAllObjects(labels, component);
    // std::cout<<"found components: "<<objects.size()<<std::endl;
            
    for(cv::Mat obj : objects){
        // isMetroYellowBackground(obj);
        // cv::imshow("obj", obj);
        // cv::waitKey(0);
        if (isMetroYellowBackground(obj)){
            // czy na masce czerwonego jest w tym miejscu czerwone M?
            int* points = get_surrounding_box(obj);
            cv::Point pt1 = cv::Point(points[0], points[1]);
            cv::Point pt2 = cv::Point(points[2], points[3]);
            
            // cv::imshow("yellow", obj(cv::Rect(pt1, pt2)));
            // std::cout<<"[0]"<<points[0]<<", [1]"<< points[1]
            // <<", [2]"<<points[2]<<", [3]"<< points[3]<<std::endl;
            
            cv::Mat redM = red_mask(cv::Rect(pt1, pt2));
            float sum = pole(obj)+pole(redM);
            float elipse = 3.14*(points[2]-points[0])/2*(points[3]-points[1])/2;

            // cv::imshow("red", redM);
            if (fabs(1.0 - elipse/sum) < 0.4 && fabs(1.0 - float(pole(obj))/float(pole(redM))) < 3){
                std::cout<<"pole zolty:"<<pole(obj)
                    <<", pole czerwone:"<<pole(redM)
                    <<", sum:"<<sum
                    <<", pole elipsy:"<<elipse
                    <<", diff:"<<elipse/sum
                    <<", ->"<<float(pole(obj))/float(pole(redM))
                    <<std::endl;
                draw_rect(image, points[0], points[1], points[2], points[3], cv::Vec3b(0,255,0));
                cv::imshow("metro.jpg", obj);
                cv::waitKey(0);
            }
        }
    }
    cv::imshow("image.jpg", image);



    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", object1);

    cv::waitKey(0);

    return 0;
}