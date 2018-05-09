#ifndef DFT_H
#define DFT_H
#include <opencv2/opencv.hpp>
#include <vector>

class CDft
{
private:
    uint16_t _dft_size;
    cv::Mat _dft[2];
    cv::Mat _magnitude;

    void _make_magnitude(cv::Size source_size);
public:
    CDft(const cv::Mat& source, uint16_t size = 512);
    CDft(uint16_t size = 512);
    cv::Mat& magnitude(void);
    CDft mul_spec(CDft &other, cv::Size size);
    void set_planes(cv::Mat *planes, cv::Size size);
    cv::Mat idft();
};

#endif // DFT_H
