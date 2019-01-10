#include <stdio.h>
#include <opencv2/opencv.hpp>


cv::Mat tresholding(cv::Mat image){
    cv::Mat p_img = cv::Mat::ones(image.rows, image.cols, CV_8UC1)*255;
    // (sy, sx, ch) = img.shape
    for(int y=0; y<image.rows; y++){
        for(int x=0; x<image.cols; x++){
            // if(image.at<cv::Vec3b>(y, x)[2]>143 
            //     && image.at<cv::Vec3b>(y, x)[1]>156//165
            //     && image.at<cv::Vec3b>(y, x)[0]<160)//100)
            //     p_img.at<uchar>(y,x)=0;
            // if(image.at<cv::Vec3b>(y, x)[2]>200 
            //     && image.at<cv::Vec3b>(y, x)[1]>156
            //     && image.at<cv::Vec3b>(y, x)[0]<160)
            //     p_img.at<uchar>(y,x)=0;
            if(image.at<cv::Vec3b>(y, x)[2]>143 
                && image.at<cv::Vec3b>(y, x)[1]<90
                && image.at<cv::Vec3b>(y, x)[0]<80)
                p_img.at<uchar>(y,x)=0;
        }
    }
    return p_img;
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

    //check img size
    // cv::Size s = image.size();
    // int sx = s.width;
    // int sy = s.height;

    // segmentacja ()
    cv::Mat mask = tresholding(image);

    // znalezienie koła


    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", mask);

    cv::waitKey(0);

    return 0;
}