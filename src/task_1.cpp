#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;   // 注意plotter工具的使用

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    auto armors = yolo.detect(img, frame_count++);
    cv::waitKey(1);

    // 先完成所有装甲板的 PnP，再从有效检测中选择置信度最高的目标。
    auto best = armors.end();
    for (auto it = armors.begin(); it != armors.end(); ++it) {
      solver.solve(*it);
      const auto ypd = tools::xyz2ypd(it->xyz_in_gimbal);
      if (!it->xyz_in_gimbal.allFinite() || !ypd.allFinite() || ypd[2] <= 0.05) continue;

      if (best == armors.end() || it->confidence > best->confidence ||
          (it->confidence == best->confidence &&
           tools::xyz2ypd(it->xyz_in_gimbal)[2] < tools::xyz2ypd(best->xyz_in_gimbal)[2])) {
        best = it;
      }
    }

    if (best == armors.end()) {
      gimbal.send(false, false, 0.0F, 0.0F);
      plotter.plot({{"yaw", 0.0}, {"pitch", 0.0}, {"fire", 0.0}});
      continue;
    }

    const auto ypd = tools::xyz2ypd(best->xyz_in_gimbal);
    if (!ypd.allFinite() || ypd[2] <= 0.05) {
      gimbal.send(false, false, 0.0F, 0.0F);
      plotter.plot({{"yaw", 0.0}, {"pitch", 0.0}, {"fire", 0.0}});
      continue;
    }

    gimbal.send(true, false, static_cast<float>(ypd[0]), static_cast<float>(ypd[1]));
    plotter.plot({
      {"yaw", ypd[0]},
      {"pitch", ypd[1]},
      {"distance", ypd[2]},
      {"fire", 0.0}
    });
  }

  return 0;
}
