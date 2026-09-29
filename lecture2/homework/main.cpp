#include "io/camera.hpp"
#include "tasks/yolo.hpp"
#include "opencv2/opencv.hpp"
#include "tools/img_tools.hpp"

int main()
{
    Camera camera;
    auto_aim::YOLO yolo("./configs/yolo.yaml");
    int frame_count = 0;
    while (true) {
        cv::Mat image = camera.read();
        if (image.empty()) continue;

        const auto armors = yolo.detect(image, frame_count++);
        for (const auto & armor : armors) {
            tools::draw_points(image, armor.points, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat display;
        cv::resize(image, display, cv::Size(640, 480));
        cv::imshow("armor detection", display);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
