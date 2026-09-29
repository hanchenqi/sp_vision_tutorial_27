#pragma once

#include <opencv2/core.hpp>

class Camera
{
public:
  Camera();
  ~Camera();
  cv::Mat read();

private:
  void * handle_ = nullptr;
  bool opened_ = false;
  bool grabbing_ = false;
};
