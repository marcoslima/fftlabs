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
    VideoCapture vc(0);
//    vc.set(CAP_PROP_FRAME_WIDTH, 1280);
//    vc.set(CAP_PROP_FRAME_HEIGHT, 720);
    Mat frame;
    int ss_no = 1;
    while(true)
    {
        vc >> frame;
        if(frame.empty())
            break;

        CDft adft(frame, false, 1024);

        imshow("Frame", frame);
        imshow("dft", adft.magnitude());
        int tecla = waitKey(1);
        if(tecla == 27)
            break;
        if(tecla == 'P' || tecla == 'p')
            waitKey();
        if(toupper(tecla) == 'S')
        {
            stringstream ss;
            ss << "screenshoot" << ss_no++ << ".png";
            imwrite(ss.str(), frame);

            stringstream ss2;
            ss2 << "screenshoot" << ss_no++ << "_fft.png";
            imwrite(ss2.str(), adft.magnitude());
        }
    }
    return 0;
}
