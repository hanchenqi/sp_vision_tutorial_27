#include "camera.hpp"

#include "hikrobot/include/MvCameraControl.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{
void check(unsigned int result, const char * operation)
{
  if (result != MV_OK) {
    throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
  }
}
}

Camera::Camera()
{
  MV_CC_DEVICE_INFO_LIST devices{};
  check(MV_CC_EnumDevices(MV_USB_DEVICE, &devices), "MV_CC_EnumDevices");
  if (devices.nDeviceNum == 0) {
    throw std::runtime_error("no Hikrobot USB camera found");
  }

  check(MV_CC_CreateHandle(&handle_, devices.pDeviceInfo[0]), "MV_CC_CreateHandle");
  try {
    check(MV_CC_OpenDevice(handle_), "MV_CC_OpenDevice");
    opened_ = true;
    check(MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS),
          "MV_CC_SetEnumValue(BalanceWhiteAuto)");
    check(MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF),
          "MV_CC_SetEnumValue(ExposureAuto)");
    check(MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF),
          "MV_CC_SetEnumValue(GainAuto)");
    check(MV_CC_SetFloatValue(handle_, "ExposureTime", 10000.0),
          "MV_CC_SetFloatValue(ExposureTime)");
    check(MV_CC_SetFloatValue(handle_, "Gain", 10.0), "MV_CC_SetFloatValue(Gain)");
    check(MV_CC_StartGrabbing(handle_), "MV_CC_StartGrabbing");
    grabbing_ = true;
  } catch (...) {
    if (opened_) MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    opened_ = false;
    handle_ = nullptr;
    throw;
  }
}

Camera::~Camera()
{
  if (handle_ == nullptr) return;
  if (grabbing_) MV_CC_StopGrabbing(handle_);
  if (opened_) MV_CC_CloseDevice(handle_);
  MV_CC_DestroyHandle(handle_);
}

cv::Mat Camera::read()
{
  if (handle_ == nullptr || !grabbing_) return {};

  MV_FRAME_OUT raw{};
  check(MV_CC_GetImageBuffer(handle_, &raw, 100), "MV_CC_GetImageBuffer");
  cv::Mat image;
  try {
    const auto width = static_cast<int>(raw.stFrameInfo.nWidth);
    const auto height = static_cast<int>(raw.stFrameInfo.nHeight);
    const cv::Mat source(cv::Size(width, height), CV_8U, raw.pBufAddr);
    if (raw.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed) {
      const cv::Mat source_bgr(cv::Size(width, height), CV_8UC3, raw.pBufAddr);
      image = source_bgr.clone();
    } else if (raw.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8) {
      cv::cvtColor(source, image, cv::COLOR_GRAY2BGR);
    } else {
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> bayer_codes = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR}};
      const auto code = bayer_codes.find(raw.stFrameInfo.enPixelType);
      if (code == bayer_codes.end()) {
        throw std::runtime_error("unsupported camera pixel format");
      }
      cv::cvtColor(source, image, code->second);
    }
  } catch (...) {
    MV_CC_FreeImageBuffer(handle_, &raw);
    throw;
  }
  check(MV_CC_FreeImageBuffer(handle_, &raw), "MV_CC_FreeImageBuffer");
  return image;
}
