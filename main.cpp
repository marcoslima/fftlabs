#include <opencv2/opencv.hpp>
#include <QMessageBox>
#include <vector>
using namespace cv;
using namespace std;
#include "dft.h"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    namedWindow("Frame", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("dft", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("bg", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("bgDft", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("fg", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("fgDft", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("mulspec", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("inverse", WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);

    VideoCapture vc(0);
//    vc.set(CAP_PROP_FRAME_WIDTH, 1280);
//    vc.set(CAP_PROP_FRAME_HEIGHT, 720);
    Mat frame;
    int ss_no = 1;
    Ptr<BackgroundSubtractorMOG2> bsm =  cv::createBackgroundSubtractorMOG2(500, 16, false);
    Mat fgMask, fgMaskDft;
    Mat bgImage, bgImageDft;

    while(true)
    {
        vc >> frame;
        if(frame.empty())
            break;

        CDft adft(frame);
        bsm->apply(frame, fgMask);
        bsm->getBackgroundImage(bgImage);

        CDft adftFg(fgMask);
        CDft adftBg(bgImage);

        CDft mulspec;
        mulspec = adftFg.mul_spec(adftBg, frame.size());

        imshow("Frame", frame);
        imshow("dft", adft.magnitude());
        imshow("bg", bgImage);
        imshow("bgDft", adftBg.magnitude());
        imshow("fg", fgMask);
        imshow("fgDft", adftFg.magnitude());
        imshow("mulspec", mulspec.magnitude());
        imshow("inverse", adft.idft());

        int tecla = waitKey(1);
        if(tecla == 27)
            break;
        if(tecla == 'P' || tecla == 'p')
            waitKey();
        if(toupper(tecla) == 'S')
        {
            stringstream ss;
            ss << "screenshoot" << ss_no << ".png";
            imwrite(ss.str(), frame);

            stringstream ss2;
            ss2 << "screenshoot" << ss_no << "_fft.png";
            imwrite(ss2.str(), adft.magnitude());
            ss_no++;
        }
    }
    return 0;
}
