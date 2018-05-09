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

public:
    CDft(const cv::Mat& source, bool multichannel = false, uint16_t size = 512);
    cv::Mat& magnitude(void);
};

#endif // DFT_H
