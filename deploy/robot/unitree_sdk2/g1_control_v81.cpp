// g1_control_v81: HomieDeploy's g1_control.cpp (DDS <-> LCM bridge, 200 Hz) for the V8.1 push-cart deployment.
//
// Differences from HomieDeploy/unitree_sdk2/g1_control.cpp (everything else is unchanged):
//   1. Kp / Kd = the gains V8.1 was trained and validated with (deploy/robot/models/v81_config.json "kp" / "kd"):
//      legs = HOMIE training gains (G1RoughCfg; the original file had 150/150/150/300 instead of 100/100/100/150),
//      waist + arms = V8.1's explicit upper drive (Unitree unitree_rl_gym deploy_real g1.yaml arm_waist_kps/kds).
//   2. Watchdog: once the policy has started, no pd_plustau_targets for > 100 ms -> damping mode (latched).
//   3. Hardware e-stop: L2 + B on the Unitree remote -> damping mode (latched), independent of the Python process.
//      Damping mode = kp 0, kd 8, tau 0 on every motor (unitree_rl_gym create_damping_cmd).
//   4. rc_command.mode carries the raw remote key bitmask (WirelessController_.keys) for the Python state machine.
//   5. state_estimator_data also carries the IMU quaternion; LCM command structs are mutex-protected.
// NOTE: written against HomieDeploy's g1_control.cpp and the unitree_sdk2 API it uses; it has NOT been compiled in
// this repository (no unitree_sdk2 libraries here). Build it on the robot (see deploy/README.md).

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <mutex>
#include <shared_mutex>
#include <lcm/lcm-cpp.hpp>
#include <thread>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

// LCM
#include "pd_tau_targets_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "body_control_data_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "arm_action_lcmt.hpp"
// gamepad
#include "unitree/common/thread/thread.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"
#include "example/wireless_controller/advanced_gamepad.hpp"

#define TOPIC_JOYSTICK "rt/wirelesscontroller"

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

const int G1_NUM_MOTOR = 29; // not included 7*2 of two hands

// key bits of WirelessController_.keys (unitree xKeySwitchUnion)
const uint16_t KEY_L2 = 1u << 5;
const uint16_t KEY_B = 1u << 9;

const double WATCHDOG_S = 0.1;
const float DAMPING_KD = 8.0f;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
  std::array<float, 4> quat = {};
  std::array<float, 3> abody = {};
};

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};


// Stiffness for all G1 joints = V8.1 training (v81_config.json "kp"; waist roll / pitch, absent from HOMIE's model,
// are held at 0 with the waist gain)
std::array<float, G1_NUM_MOTOR> Kp{
    100, 100, 100, 150, 40, 40,      // left leg  (hip pitch, roll, yaw, knee, ankle pitch, roll)
    100, 100, 100, 150, 40, 40,      // right leg
    300, 300, 300,                   // waist yaw, roll, pitch
    100, 100, 50, 50, 20, 20, 20,    // left arm  (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw)
    100, 100, 50, 50, 20, 20, 20,    // right arm
};

// Damping for all G1 joints = V8.1 training (v81_config.json "kd")
std::array<float, G1_NUM_MOTOR> Kd{
    2, 2, 2, 4, 2, 2,                // left leg
    2, 2, 2, 4, 2, 2,                // right leg
    3, 3, 3,                         // waist
    2, 2, 2, 2, 1, 1, 1,             // left arm
    2, 2, 2, 2, 1, 1, 1,             // right arm
};


enum PRorAB { PR = 0, AB = 1 };

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

class G1Control {
 private:
  double time_;
  double control_dt_;  // [5ms] thus 200hz
  double duration_;    // [3 s]
  PRorAB mode_;
  uint8_t mode_machine_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_, joystick_thread_ptr_;

  lcm::LCM _simpleLCM;
  std::thread _simple_LCM_thread;
  std::mutex command_mutex_;
  bool _firstCommandReceived;
  std::chrono::steady_clock::time_point last_command_time_;
  std::atomic<bool> damping_{false};
  state_estimator_lcmt body_state_simple = {0};
  body_control_data_lcmt joint_state_simple = {0};
  pd_tau_targets_lcmt joint_command_simple = {0};
  arm_action_lcmt arm_action_simple = {0};
  rc_command_lcmt rc_command = {0};
  Gamepad gamepad;
  unitree_go::msg::dds_::WirelessController_ joystick_msg;
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
  std::mutex joystick_mutex;

 public:
  G1Control(std::string networkInterface)
      : time_(0.0),
        control_dt_(0.005),
        duration_(3.0),
        mode_(PR),
        mode_machine_(0) {
    ChannelFactory::Instance()->Init(0, networkInterface);

    // create publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // create subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1Control::LowStateHandler, this, std::placeholders::_1), 1);

    // create gamepad
    joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_subscriber->InitChannel(std::bind(&G1Control::JoystickHandler, this, std::placeholders::_1), 1);

    _firstCommandReceived = false;
    last_command_time_ = std::chrono::steady_clock::now();

    // create threads
    command_writer_ptr_ =
        CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 5000,
                                &G1Control::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 5000, &G1Control::Control, this);
    joystick_thread_ptr_ = CreateRecurrentThreadEx(
        "nn_ctrl", UT_CPU_ID_NONE, 4000, &G1Control::JoystickStep, this);

    // add lcm subscriber
    _simpleLCM.subscribe("pd_plustau_targets", &G1Control::handleActionLCM, this);
    _simpleLCM.subscribe("arm_action", &G1Control::handleArmLCM, this);
    _simple_LCM_thread = std::thread(&G1Control::_simpleLCMThread, this);
  }

  void JoystickHandler(const void *message)
    {
        std::lock_guard<std::mutex> lock(joystick_mutex);
        joystick_msg = *(unitree_go::msg::dds_::WirelessController_ *)message;
    }

  void _simpleLCMThread(){
    while(true){
        _simpleLCM.handle();
    }
    }

  void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg){
    (void) rbuf;
    (void) chan;
    std::lock_guard<std::mutex> lock(command_mutex_);
    joint_command_simple = *msg;
    last_command_time_ = std::chrono::steady_clock::now();

    if (_firstCommandReceived == false){
      _firstCommandReceived = true;
      std::cout << "First command received" << std::endl;
    }
  }

  void handleArmLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const arm_action_lcmt * msg){
    (void) rbuf;
    (void) chan;
    std::lock_guard<std::mutex> lock(command_mutex_);
    arm_action_simple = *msg;
  }

  void LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    if (low_state.crc() !=
        Crc32Core((uint32_t *)&low_state,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
      std::cout << "low_state CRC Error" << std::endl;
      return;
    }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_tmp.quat = low_state.imu_state().quaternion();
    imu_tmp.abody = low_state.imu_state().accelerometer();
    imu_state_buffer_.SetData(imu_tmp);
    // update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0)
        std::cout << "G1 type: " << unsigned(low_state.mode_machine())
                  << std::endl;
      mode_machine_ = low_state.mode_machine();
    }
  }

  void LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
        dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
        dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
        dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
        dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
      }

      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                        (sizeof(dds_low_command) >> 2) - 1);
      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  void JoystickStep() {
    uint16_t keys = 0;
    {
      std::lock_guard<std::mutex> lock(joystick_mutex);
      gamepad.Update(joystick_msg);
      keys = joystick_msg.keys();
    }

    if ((keys & KEY_L2) && (keys & KEY_B) && !damping_.load()) {
      damping_.store(true);
      std::cout << "[g1_control_v81] L2+B: DAMPING (restart this program to recover)" << std::endl;
    }

    rc_command.mode = static_cast<int16_t>(keys);   // raw key bitmask, decoded by the Python side
    rc_command.left_stick[0] = gamepad.lx;
    rc_command.left_stick[1] = gamepad.ly; // +up is 1, +right is 0
    rc_command.right_stick[0] = gamepad.rx;
    rc_command.right_stick[1] = gamepad.ry;
    rc_command.right_lower_right_switch = gamepad.R2.pressed;

    _simpleLCM.publish("rc_command", &rc_command);
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = 0.0;
      motor_command_tmp.kd.at(i) = 0.0;
    }

    if (ms) {
      time_ += control_dt_;
      if (!damping_.load() && _firstCommandReceived) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        double silent = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_command_time_).count();
        if (silent > WATCHDOG_S) {
          damping_.store(true);
          std::cout << "[g1_control_v81] no command for " << silent << " s: DAMPING (restart to recover)" << std::endl;
        }
      }

      if (damping_.load()) {
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          motor_command_tmp.kd.at(i) = DAMPING_KD;
        }
      } else if (time_ < duration_) {
        // [Stage 1]: set robot to zero posture
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);

          double q_des = 0;
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.q_target.at(i) =
              (q_des - ms->q.at(i)) * ratio + ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp[i];
          motor_command_tmp.kd.at(i) = Kd[i];
        }
      } else {
          const std::shared_ptr<const ImuState> imu_tmp_ptr = imu_state_buffer_.GetData();
          if (imu_tmp_ptr) {
            for (int i = 0; i < 3; i++){
              body_state_simple.rpy[i] = imu_tmp_ptr->rpy.at(i);
              body_state_simple.omegaBody[i] = imu_tmp_ptr->omega.at(i);
              body_state_simple.aBody[i] = imu_tmp_ptr->abody.at(i);
            }
            for (int i = 0; i < 4; i++){
              body_state_simple.quat[i] = imu_tmp_ptr->quat.at(i);
            }
          }

          for (int i = 0; i < G1_NUM_MOTOR; ++i){
            joint_state_simple.q[i] = ms->q.at(i);
            joint_state_simple.qd[i] = ms->dq.at(i);
          }
          _simpleLCM.publish("state_estimator_data", &body_state_simple);
          _simpleLCM.publish("body_control_data", &joint_state_simple);

          std::lock_guard<std::mutex> lock(command_mutex_);
          for (int i = 0; i < 15; ++i){
            motor_command_tmp.q_target.at(i) = joint_command_simple.q_des[i];
            motor_command_tmp.dq_target.at(i) = 0.0;
            motor_command_tmp.kp.at(i) = Kp[i];
            motor_command_tmp.kd.at(i) = Kd[i];
          }
          for (int i = 0; i < 14; ++i){
            motor_command_tmp.q_target.at(i+15) = arm_action_simple.act[i];
            motor_command_tmp.dq_target.at(i+15) = 0.0;
            motor_command_tmp.kp.at(i+15) = Kp[i+15];
            motor_command_tmp.kd.at(i+15) = Kd[i+15];
          }
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: g1_control_v81 <network interface, e.g. eth0>" << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];
  G1Control custom(networkInterface);
  while (true) usleep(20000);
  return 0;
}
