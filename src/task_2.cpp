#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/target.hpp"
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
  tools::Plotter plotter;  // 注意plotter工具的使用

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

  std::optional<auto_aim::Target> target;
  std::optional<std::chrono::steady_clock::time_point> last_measurement;
  std::optional<std::chrono::steady_clock::time_point> last_fire;
  int fired_count = 0;

  constexpr double TARGET_TIMEOUT = 0.35;
  constexpr double FIRE_INTERVAL = 0.2;  // 5 Hz，每次 fire 命令只请求一发
  constexpr int FIRE_LIMIT = 10;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    auto armors = yolo.detect(img, frame_count++);
    cv::waitKey(1);
    for (auto & armor : armors) solver.solve(armor);

    // 丢失目标超过一段时间后清空旧滤波器，避免重新捕获时污染状态。
    if (target && last_measurement &&
        tools::delta_time(t, *last_measurement) > TARGET_TIMEOUT) {
      target.reset();
      last_measurement.reset();
    }

    if (target) target->predict(t);

    auto selected = armors.end();
    if (!armors.empty()) {
      if (!target) {
        // 首次捕获时优先使用置信度高、距离近的装甲板。
        for (auto it = armors.begin(); it != armors.end(); ++it) {
          if (!it->xyz_in_world.allFinite() || !it->ypr_in_world.allFinite() ||
              !it->ypd_in_world.allFinite() || it->ypd_in_world[2] <= 0.05) {
            continue;
          }
          if (selected == armors.end() || it->confidence > selected->confidence ||
              (it->confidence == selected->confidence &&
               it->ypd_in_world[2] < selected->ypd_in_world[2])) {
            selected = it;
          }
        }
      } else {
        // 只把同一编号的检测送入 EKF；其余检测可能来自其他机器人。
        const auto predicted = target->armor_xyza_list();
        double best_cost = std::numeric_limits<double>::max();
        for (auto it = armors.begin(); it != armors.end(); ++it) {
          if (it->name != target->name || !it->xyz_in_world.allFinite() ||
              !it->ypr_in_world.allFinite() || !it->ypd_in_world.allFinite() ||
              it->ypd_in_world[2] <= 0.05) {
            continue;
          }

          double cost = std::numeric_limits<double>::max();
          for (const auto & xyza : predicted) {
            const auto predicted_ypd = tools::xyz2ypd(xyza.head<3>());
            const auto yaw_error =
              std::abs(tools::limit_rad(it->ypr_in_world[0] - xyza[3]));
            const auto distance_error = std::abs(it->ypd_in_world[2] - predicted_ypd[2]);
            const auto pitch_error = std::abs(it->ypd_in_world[1] - predicted_ypd[1]);
            cost = std::min(cost, yaw_error + 0.25 * distance_error + 0.25 * pitch_error);
          }

          if (cost < best_cost) {
            best_cost = cost;
            selected = it;
          }
        }

        // 超出这个阈值的观测不参与更新，但短暂漏检时仍使用 EKF 预测值。
        if (best_cost > 1.0) selected = armors.end();
      }
    }

    if (selected != armors.end()) {
      if (!target) {
        target.emplace(*selected, t);
      } else {
        target->update(*selected);
      }
      last_measurement = t;
    }

    if (target && target->diverged()) {
      target.reset();
      last_measurement.reset();
    }

    bool control = false;
    bool fire = false;
    double yaw = 0.0;
    double pitch = 0.0;
    double distance = 0.0;
    double omega = 0.0;

    if (target && !target->diverged()) {
      const auto x = target->ekf_x();
      const auto predicted = target->armor_xyza_list();
      if (x.size() == 11 && x.allFinite() && !predicted.empty()) {
        omega = x[7];
        int id = std::clamp(target->last_id, 0, static_cast<int>(predicted.size()) - 1);

        // 用 EKF 的角速度做简单前馈，补偿子弹飞行时间和处理延迟。
        const auto current_xyz = predicted[id].head<3>();
        distance = current_xyz.norm();
        const auto gimbal_state = gimbal.state();
        const double bullet_speed = gimbal_state.bullet_speed;
        const double flight_time =
          (std::isfinite(bullet_speed) && bullet_speed > 1.0)
            ? std::clamp(distance / bullet_speed, 0.0, 0.20)
            : 0.08;

        const auto armor_num = static_cast<int>(predicted.size());
        const auto angle =
          x[6] + static_cast<double>(id) * 2.0 * CV_PI / armor_num + omega * flight_time;
        const bool use_long_armor = armor_num == 4 && (id == 1 || id == 3);
        const auto radius = use_long_armor ? x[8] + x[9] : x[8];
        const auto z = use_long_armor ? x[4] + x[10] : x[4];
        const Eigen::Vector3d aim_world{x[0] - radius * std::cos(angle),
                                        x[2] - radius * std::sin(angle), z};
        const Eigen::Vector3d aim_gimbal = solver.R_gimbal2world().transpose() * aim_world;
        const auto ypd = tools::xyz2ypd(aim_gimbal);

        if (ypd.allFinite() && ypd[2] > 0.05) {
          control = true;
          yaw = ypd[0];
          pitch = ypd[1];

          const auto fire_ready =
            !last_fire || std::chrono::duration<double>(t - *last_fire).count() >= FIRE_INTERVAL;
          if (target->is_converged() && fire_ready && fired_count < FIRE_LIMIT) {
            fire = true;
            last_fire = t;
            ++fired_count;
          }
        }
      }
    }

    gimbal.send(control, fire, static_cast<float>(yaw), static_cast<float>(pitch));
    plotter.plot({
      {"yaw", yaw},
      {"pitch", pitch},
      {"omega", omega},
      {"distance", distance},
      {"control", control ? 1.0 : 0.0},
      {"fire", fire ? 1.0 : 0.0},
      {"fired_count", fired_count}
    });

  }

  return 0;
}
