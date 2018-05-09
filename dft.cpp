#include "dft.h"
using namespace cv;

void Merge(Mat ch1, Mat ch2, Mat& target)
{
    Mat planes[2] = {ch1, ch2};
    merge(planes, 2, target);
}

CDft::CDft(const cv::Mat &source, uint16_t size)
{
    _dft_size = size;
    Mat src, planes, gray;
    if(source.channels() > 1)
    {
        cvtColor(source, gray, COLOR_BGR2GRAY);
    }
    else
    {
        source.copyTo(gray);
    }
    resize(gray, gray, Size(_dft_size, _dft_size));
    gray.convertTo(src, CV_32FC1, 1);

    Merge(src, Mat::zeros(src.size(), src.type()), planes);
    Mat imgDft;
    dft(src, imgDft, DFT_COMPLEX_OUTPUT);
    split(imgDft, _dft);
    _make_magnitude(source.size());
}

CDft::CDft(uint16_t size)
{
    _dft_size = size;
}

void CDft::_make_magnitude(Size source_size)
{
    cv::magnitude(_dft[0], _dft[1], _magnitude);
    _magnitude += Scalar::all(1);
    cv::log(_magnitude, _magnitude);
    cv::normalize(_magnitude, _magnitude, 255, 0, NORM_MINMAX, CV_8UC1);

    Mat M = (Mat_<double>(2,3) << 1.0 , 0.0 , _dft_size/2,
                                  0.0 , 1.0 , _dft_size/2);
    warpAffine(_magnitude, _magnitude, M, Size(_dft_size, _dft_size), 0, BORDER_WRAP);
    resize(_magnitude, _magnitude, source_size);
}

Mat &CDft::magnitude()
{
    return _magnitude;
}

void CDft::set_planes(Mat* planes, Size size)
{
    planes[0].copyTo(_dft[0]);
    planes[1].copyTo(_dft[1]);
    _make_magnitude(size);
}

CDft CDft::mul_spec(CDft& other, Size size)
{
    Mat f1, f2, mult, planes[2];
    CDft result;

    merge(_dft, 2, f1);
    merge(other._dft, 2, f2);
    mulSpectrums(f1, f2, mult, 0);
    split(mult, planes);
    result.set_planes(planes, size);
    return result;
}

Mat CDft::idft(void)
{
    Mat org, result;
    merge(_dft, 2, org);

    dft(org, result, DFT_INVERSE | DFT_REAL_OUTPUT);
    result.convertTo(result, CV_8UC1, 255);
    return result;
}
