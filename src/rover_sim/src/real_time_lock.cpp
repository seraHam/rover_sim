// real_time_lock: a world plugin that keeps Gazebo's sim time locked to the wall clock.
//
// The stack runs on wall clock and livox_sim_bridge maps sim stamps to wall time with one
// offset, so sim time has to advance exactly as fast as wall time. Gazebo's real-time cap
// (real_time_update_rate) only paces each step from the start of the previous one: time a
// step overruns is lost for good. Every MID-360 scan holds the physics lock while it casts
// (0.3-1 ms), so with 2 agents the world ran at RTF 0.9988 and the offset drifted 79 ms a
// minute; with it, 4 agents hold the offset within 10 ms over 11 minutes.
//
// This plugin anchors (wall, sim) at the first step after an unpause and raises the update
// rate in proportion to how far sim time is behind that anchor (lag / time_constant, at
// most max_boost), so lost time is made up within a few steps. It never sets the rate
// below the world's own, so sim time never runs ahead of the wall clock. A lag over
// max_lag is a stall or an overload, not jitter: it is accepted (re-anchored, with a
// warning) and shows up as a low sim/rtf in the bridges, as it should.
//
//   <plugin name="real_time_lock" filename="librover_sim_real_time_lock.so">
//     <time_constant>0.5</time_constant>  <!-- s of lag per 100 % boost -->
//     <max_boost>0.5</max_boost>          <!-- rate up to (1 + max_boost) x the world's -->
//     <max_lag>0.5</max_lag>              <!-- s -->
//   </plugin>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/PhysicsEngine.hh>
#include <gazebo/physics/World.hh>

namespace rover_sim
{

class RealTimeLock : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    engine_ = world->Physics();
    base_rate_ = engine_->GetRealTimeUpdateRate();
    applied_rate_ = base_rate_;
    if (sdf->HasElement("time_constant")) {
      time_constant_ = sdf->Get<double>("time_constant");
    }
    if (sdf->HasElement("max_boost")) {
      max_boost_ = sdf->Get<double>("max_boost");
    }
    if (sdf->HasElement("max_lag")) {
      max_lag_ = sdf->Get<double>("max_lag");
    }
    if (base_rate_ <= 0 || time_constant_ <= 0 || max_boost_ < 0 || max_lag_ <= 0) {
      std::cerr << "[real_time_lock] off: needs a real_time_update_rate > 0 (have " << base_rate_
                << ") and time_constant, max_lag > 0, max_boost >= 0" << std::endl;
      return;
    }
    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&RealTimeLock::OnUpdate, this));
    // Paused time is not lost time: re-anchor on the next step after any pause.
    pause_conn_ = gazebo::event::Events::ConnectPause([this](bool) {anchored_ = false;});
    // std::cout/cerr: Gazebo's own console is silent unless gzserver runs --verbose.
    std::cout << "[real_time_lock] holding sim time to the wall clock: rate " << base_rate_
              << " Hz, up to x" << 1.0 + max_boost_ << " while behind (time constant "
              << time_constant_ << " s, max lag " << max_lag_ << " s)" << std::endl;
  }

private:
  void OnUpdate()
  {
    const double wall = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    const double sim = world_->SimTime().Double();
    if (!anchored_ || sim < last_sim_) {      // first step, after a pause, or a world reset
      Anchor(wall, sim);
      return;
    }
    last_sim_ = sim;
    const double lag = (wall - wall0_) - (sim - sim0_);
    if (lag > max_lag_) {
      std::cerr << "[real_time_lock] sim time fell " << lag << " s behind the wall clock (a stall, "
                << "or more load than real time allows): not catching that up" << std::endl;
      Anchor(wall, sim);
      return;
    }
    // Whole Hz: the rate changes only every ~time_constant / base_rate s of lag.
    const double rate = std::round(base_rate_ * (1.0 + std::clamp(lag / time_constant_, 0.0, max_boost_)));
    if (rate != applied_rate_) {
      engine_->SetRealTimeUpdateRate(rate);
      applied_rate_ = rate;
    }
  }

  void Anchor(double wall, double sim)
  {
    wall0_ = wall;
    sim0_ = sim;
    last_sim_ = sim;
    anchored_ = true;
    if (applied_rate_ != base_rate_) {
      engine_->SetRealTimeUpdateRate(base_rate_);
      applied_rate_ = base_rate_;
    }
  }

  gazebo::physics::WorldPtr world_;
  gazebo::physics::PhysicsEnginePtr engine_;
  gazebo::event::ConnectionPtr update_conn_;
  gazebo::event::ConnectionPtr pause_conn_;
  double base_rate_{0.0};
  double applied_rate_{0.0};
  double time_constant_{0.5};
  double max_boost_{0.5};
  double max_lag_{0.5};
  std::atomic<bool> anchored_{false};    // cleared from the pause event's thread
  double wall0_{0.0};
  double sim0_{0.0};
  double last_sim_{0.0};
};

GZ_REGISTER_WORLD_PLUGIN(RealTimeLock)

}  // namespace rover_sim
