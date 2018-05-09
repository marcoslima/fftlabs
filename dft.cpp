#include "dft.h"
using namespace cv;

void Merge(Mat ch1, Mat ch2, Mat& target)
{
    Mat planes[2] = {ch1, ch2};
    merge(planes, 2, target);
}

CDft::CDft(const cv::Mat &source, bool multichannel, uint16_t size)
{
    (void)multichannel;

    _dft_size = size;
    Mat src, planes, gray;
    cvtColor(source, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(_dft_size, _dft_size));
    gray.convertTo(src, CV_32FC1, 1);

    Merge(src, Mat::zeros(src.size(), src.type()), planes);
    Mat imgDft;
    dft(src, imgDft, DFT_COMPLEX_OUTPUT);
    split(imgDft, _dft);
    cv::magnitude(_dft[0], _dft[1], _magnitude);
    _magnitude += Scalar::all(1);
    cv::log(_magnitude, _magnitude);
    cv::normalize(_magnitude, _magnitude, 255, 0, NORM_MINMAX, CV_8UC1);

    Mat M = (Mat_<double>(2,3) << 1.0 , 0.0 , _dft_size/2,
                                  0.0 , 1.0 , _dft_size/2);
    warpAffine(_magnitude, _magnitude, M, Size(_dft_size, _dft_size), 0, BORDER_WRAP);
    resize(_magnitude, _magnitude, source.size());
}

Mat &CDft::magnitude()
{
    return _magnitude;
}

