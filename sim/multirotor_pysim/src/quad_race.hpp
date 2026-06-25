// Layer 2: QuadRaceBatch — the MonoRace gate-racing per-step math in C++/OpenMP.
//
// Faithful port of rl/env/quadrace_vec_env.py (step_wait + update_states),
// working in the policy's NED/FRD ("their") frame so it matches the Python
// oracle bit-for-bit. Owns its own Dynamics vector so dynamics + transform +
// obs + reward run in a single parallel loop over N. Reset stays in Python: it
// pushes sampled init states + per-env DR params via reset_envs().
//
// Their-frame state row (17): x,y,z, vx,vy,vz, qw,qx,qy,qz, p,q,r, w1..w4(norm)

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "batch_sim.hpp"

namespace mps {

constexpr double W_MAX_N = 3000.0;   // their motor-speed normalization
constexpr double TWO_PI = 6.283185307179586;
constexpr double PI = 3.141592653589793;

inline double wrap_pi(double a) {
  a = std::fmod(a + PI, TWO_PI);
  if (a < 0) a += TWO_PI;
  return a - PI;
}

// Their Tait-Bryan euler from quaternion (matches _euler_from_quat).
inline void euler_from_quat(double qw, double qx, double qy, double qz,
                            double& phi, double& theta, double& psi) {
  phi = std::atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
  double t = 2 * (qw * qy - qz * qx);
  t = t > 1 ? 1 : (t < -1 ? -1 : t);
  theta = std::asin(t);
  psi = std::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

class QuadRaceBatch {
public:
  explicit QuadRaceBatch(int num_envs) : n_(num_envs) {
    if (num_envs <= 0) throw std::runtime_error("num_envs must be > 0");
    ModelParams mp = default_params();
    dyn_.reserve(n_);
    for (int i = 0; i < n_; ++i) dyn_.emplace_back(mp, State());
    world_.assign(n_, std::array<double, 17>{});
    target_.assign(n_, 0);
    steps_.assign(n_, 0);
    prev_act_.assign(n_, Vec4::Zero());
    for (int m = 0; m < 4; ++m) { perm_[m] = m; inv_[m] = m; }
  }

  int num_envs() const { return n_; }
  void set_model_params(const py::dict& d) { set_models_from_dict(d, dyn_); }

  // their-frame track tables
  void set_track(ArrD gate_pos, ArrD gate_yaw, ArrD gate_pos_rel, ArrD gate_yaw_rel,
                 py::object bounds_xy) {
    auto gp = gate_pos.unchecked<2>(); auto gy = gate_yaw.unchecked<1>();
    auto gpr = gate_pos_rel.unchecked<2>(); auto gyr = gate_yaw_rel.unchecked<1>();
    num_gates_ = (int)gp.shape(0);
    gate_pos_.resize(num_gates_); gate_yaw_.resize(num_gates_);
    gate_pos_rel_.resize(num_gates_); gate_yaw_rel_.resize(num_gates_);
    for (int i = 0; i < num_gates_; ++i) {
      gate_pos_[i] = {gp(i, 0), gp(i, 1), gp(i, 2)};
      gate_yaw_[i] = gy(i);
      gate_pos_rel_[i] = {gpr(i, 0), gpr(i, 1), gpr(i, 2)};
      gate_yaw_rel_[i] = gyr(i);
    }
    has_bounds_ = !bounds_xy.is_none();
    if (has_bounds_) {
      auto b = bounds_xy.cast<ArrD>().unchecked<2>();
      bx0_ = b(0, 0); bx1_ = b(0, 1); by0_ = b(1, 0); by1_ = b(1, 1);
    }
  }

  // scalars + reward weights (settable from Python)
  void set_config(const py::dict& c) {
    auto g = [&](const char* k, double dflt) {
      return c.contains(k) ? c[k].cast<double>() : dflt;
    };
    dt_ = g("dt", 0.01);
    wmin_ = g("w_min", 341.75); wmax_ = g("w_max", 3100.0); k_ = g("k", 0.5);
    cam_ = g("cam_angle", 45 * PI / 180);
    gate_size_ = g("gate_size", 0.8); gate_thick_ = g("gate_thickness", 0.5);
    speed_limit_ = g("speed_limit", 99.0);
    loop_ = g("loop_gates", 1.0) != 0.0;
    ground_h_ = g("ground_height", 0.0); v_ground_ = g("v_ground", 2.0);
    max_steps_ = (int)g("max_steps", 2000);
    progress_ = g("progress_reward", 1.0); gate_rew_ = g("gate_reward", 1.0);
    rate_pen_ = g("angular_rate_penalty", 0.001);
    offset_pen_ = g("gate_offset_penalty", 1.0);
    perc_pen_ = g("perception_penalty", 0.0);
    motor_pen_ = g("motor_penalty", 0.0); motor_thr_ = g("motor_penalty_threshold", 0.0);
    low_pen_ = g("low_action_penalty", 0.0); crash_ = g("crash_penalty", 10.0);
    if (c.contains("motor_perm")) {
      auto p = c["motor_perm"].cast<ArrD>().unchecked<1>();
      for (int m = 0; m < 4; ++m) perm_[m] = (int)p(m);
      for (int m = 0; m < 4; ++m) inv_[perm_[m]] = m;
    }
  }

  // Python pushes sampled inits (their frame) + DR params for done envs.
  // Returns the fresh obs rows for those envs (M x 20).
  py::array_t<double> reset_envs(const std::vector<int>& idx, ArrD states_their,
                                 ArrD target_gates, const py::dict& model_params) {
    auto s = states_their.unchecked<2>();
    auto tg = target_gates.unchecked<1>();
    int m = (int)idx.size();
    if (s.shape(0) != m || s.shape(1) != 17 || tg.shape(0) != m)
      throw std::runtime_error("reset_envs shape mismatch");
    set_models_from_dict(model_params, dyn_, &idx);
    py::array_t<double> obs({m, 20});
    auto o = obs.mutable_unchecked<2>();
    for (int k = 0; k < m; ++k) {
      int e = idx[k];
      for (int j = 0; j < 17; ++j) world_[e][j] = s(k, j);
      target_[e] = (int)tg(k);
      steps_[e] = 0;
      prev_act_[e].setZero();
      double enu[17];
      their_to_enu(world_[e].data(), enu);
      set_dyn_from_enu(dyn_[e], enu);
      double row[20];
      compute_obs(world_[e].data(), target_[e], row);
      for (int j = 0; j < 20; ++j) o(k, j) = row[j];
    }
    return obs;
  }

  // The hot path: one OpenMP loop over N. Does NOT reset (Python does).
  py::tuple step(ArrD actions_arr) {
    auto act = actions_arr.unchecked<2>();
    if (act.shape(0) != n_ || act.shape(1) != 4)
      throw std::runtime_error("step expects (num_envs, 4)");
    py::array_t<double> obs({n_, 20});
    py::array_t<double> rew(n_), done(n_), passed(n_), trunc(n_);
    auto O = obs.mutable_unchecked<2>();
    auto R = rew.mutable_unchecked<1>();
    auto D = done.mutable_unchecked<1>();
    auto P = passed.mutable_unchecked<1>();
    auto Tr = trunc.mutable_unchecked<1>();
    const Vec3 zero = Vec3::Zero();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_; ++i) {
      // ---- action U -> Wc (their order) -> our order -> integrate ----
      Vec4 wc_their, wc_our;
      for (int mm = 0; mm < 4; ++mm) {
        double u = act(i, mm); u = u < -1 ? -1 : (u > 1 ? 1 : u);
        double cap = (u + 1) * 0.5;
        double s2 = k_ * cap * cap + (1 - k_) * cap;
        wc_their[mm] = (wmax_ - wmin_) * std::sqrt(s2 > 0 ? s2 : 0) + wmin_;
      }
      for (int j = 0; j < 4; ++j) wc_our[j] = wc_their[inv_[j]];
      dyn_[i].process_euler_explicit(wc_our, dt_, zero, false);

      // ---- read ENU, transform to their frame ----
      double enu[17], nw[17];
      read_dyn_to_enu(dyn_[i], enu);
      enu_to_their(enu, nw);

      const double* ow = world_[i].data();   // old (their) — pos_old/d2g_old
      int tgt = target_[i] % num_gates_;
      const auto& gp = gate_pos_[tgt];
      double gyaw = gate_yaw_[tgt];

      // ---- reward ----
      double d2g_old = dist3(ow, gp.data());
      double d2g_new = dist3(nw, gp.data());
      double prog = d2g_old - d2g_new;
      double cap_prog = speed_limit_ * dt_;
      if (prog > cap_prog) prog = cap_prog;
      double r = progress_ * prog;
      r -= rate_pen_ * std::sqrt(nw[10] * nw[10] + nw[11] * nw[11] + nw[12] * nw[12]);
      // motor / low-action penalties (scaled actions)
      for (int mm = 0; mm < 4; ++mm) {
        double sa = (act(i, mm) + 1) * 0.5;
        double spa = (prev_act_[i][mm] + 1) * 0.5;
        if (low_pen_ > 0) { double lo = 0.5 - sa; if (lo > 0) r -= low_pen_ * lo; }
        if (motor_pen_ > 0) { double ex = std::fabs(sa - spa) - motor_thr_; if (ex > 0) r -= motor_pen_ * ex; }
      }
      // perception (only > 60 deg)
      if (perc_pen_ > 0) {
        double pa = perc_angle(nw, gp.data());
        if (pa > PI / 3) r -= perc_pen_ * pa;
      }

      // ---- gate plane crossing (target gate) ----
      double cN = std::cos(gyaw), sN = std::sin(gyaw);
      double proj_old = (ow[0] - gp[0]) * cN + (ow[1] - gp[1]) * sN;
      double proj_new = (nw[0] - gp[0]) * cN + (nw[1] - gp[1]) * sN;
      bool crossed = proj_old < 0 && proj_new > 0;
      double half = gate_size_ * 0.5;
      double chl = cheby3(nw, gp.data());
      bool gate_passed = crossed && chl < half;
      bool gate_collision = crossed && chl > half;
      // ---- frame-collision boxes (each gate's own z) ----
      double half_out = 2.7 * 0.5, dthk = gate_thick_ * 0.5;
      for (int gi = 0; gi < num_gates_; ++gi) {
        const auto& gpi = gate_pos_[gi];
        double ci = std::cos(gate_yaw_[gi]), si = std::sin(gate_yaw_[gi]);
        double ddx = nw[0] - gpi[0], ddy = nw[1] - gpi[1];
        double nrm = ddx * ci + ddy * si;
        double lat = -ddx * si + ddy * ci;
        double dz = nw[2] - gpi[2];
        if (std::fabs(nrm) < dthk &&
            (std::fabs(lat) > half || std::fabs(dz) > half) &&
            std::fabs(lat) < half_out && std::fabs(dz) < half_out)
          gate_collision = true;
      }
      if (gate_passed) r = gate_rew_ - offset_pen_ * d2g_new / half;
      if (gate_collision) r = -crash_;

      // ---- ground / OOB / time ----
      double vmag = std::sqrt(nw[3] * nw[3] + nw[4] * nw[4] + nw[5] * nw[5]);
      bool ground = (nw[2] > -ground_h_) && (vmag > v_ground_);
      if (ground) r = -crash_;
      bool oob = false;
      if (has_bounds_) oob = nw[0] < bx0_ || nw[0] > bx1_ || nw[1] < by0_ || nw[1] > by1_;
      if (nw[2] < -10) oob = true;
      double rlim = 1700 * PI / 180;
      if (std::fabs(nw[10]) > rlim || std::fabs(nw[11]) > rlim || std::fabs(nw[12]) > rlim) oob = true;
      if (oob) r = -crash_;

      steps_[i] += 1;
      bool max_reached = steps_[i] >= max_steps_;
      if (gate_passed) {
        target_[i] += 1;
        if (loop_) target_[i] %= num_gates_;
      }
      bool final = (!loop_) && (target_[i] >= num_gates_);
      bool d = max_reached || ground || gate_collision || oob || final;

      // ---- obs relative to (possibly advanced) target ----
      double row[20];
      compute_obs(nw, target_[i], row);
      for (int j = 0; j < 20; ++j) O(i, j) = row[j];
      R(i) = r; D(i) = d ? 1.0 : 0.0; P(i) = gate_passed ? 1.0 : 0.0;
      Tr(i) = max_reached ? 1.0 : 0.0;

      for (int j = 0; j < 17; ++j) world_[i][j] = nw[j];
      for (int mm = 0; mm < 4; ++mm) prev_act_[i][mm] = act(i, mm);
    }
    return py::make_tuple(obs, rew, done, passed, trunc);
  }

private:
  // -------- transforms (their <-> ENU); see _their_to_sim/_sim_to_their --- //
  void their_to_enu(const double* W, double* S) const {
    S[0] = W[0]; S[1] = -W[1]; S[2] = -W[2];
    S[3] = W[3]; S[4] = -W[4]; S[5] = -W[5];
    S[6] = W[6]; S[7] = W[7]; S[8] = -W[8]; S[9] = -W[9];
    S[10] = W[10]; S[11] = -W[11]; S[12] = -W[12];
    double wt[4];
    for (int m = 0; m < 4; ++m) wt[m] = (W[13 + m] + 1) * 0.5 * W_MAX_N;  // their order
    for (int j = 0; j < 4; ++j) S[13 + j] = wt[inv_[j]];                  // our[j]=their[inv[j]]
  }
  void enu_to_their(const double* S, double* W) const {
    W[0] = S[0]; W[1] = -S[1]; W[2] = -S[2];
    W[3] = S[3]; W[4] = -S[4]; W[5] = -S[5];
    W[6] = S[6]; W[7] = S[7]; W[8] = -S[8]; W[9] = -S[9];
    W[10] = S[10]; W[11] = -S[11]; W[12] = -S[12];
    for (int m = 0; m < 4; ++m) W[13 + m] = 2 * S[13 + perm_[m]] / W_MAX_N - 1;  // their[m]=our[perm[m]]
  }
  static void read_dyn_to_enu(const Dyn& dyn, double* S) {
    const auto& k = dyn.get_state().kinematics;
    S[0] = k.position.x(); S[1] = k.position.y(); S[2] = k.position.z();
    S[3] = k.linear_velocity.x(); S[4] = k.linear_velocity.y(); S[5] = k.linear_velocity.z();
    S[6] = k.orientation.w(); S[7] = k.orientation.x(); S[8] = k.orientation.y(); S[9] = k.orientation.z();
    S[10] = k.angular_velocity.x(); S[11] = k.angular_velocity.y(); S[12] = k.angular_velocity.z();
    const auto& w = dyn.get_state().actuators.motor_angular_velocity;
    S[13] = w(0); S[14] = w(1); S[15] = w(2); S[16] = w(3);
  }
  static void set_dyn_from_enu(Dyn& dyn, const double* S) {
    State st = dyn.get_state();
    st.kinematics.position = Vec3(S[0], S[1], S[2]);
    st.kinematics.linear_velocity = Vec3(S[3], S[4], S[5]);
    st.kinematics.orientation = Eigen::Quaterniond(S[6], S[7], S[8], S[9]).normalized();
    st.kinematics.angular_velocity = Vec3(S[10], S[11], S[12]);
    st.actuators.motor_angular_velocity = Vec4(S[13], S[14], S[15], S[16]);
    dyn.set_state(st);
  }

  // -------- obs (port of update_states) ---------------------------------- //
  void compute_obs(const double* W, int target, double* o) const {
    int tgt = target % num_gates_;
    const auto& gp = gate_pos_[tgt];
    double gyaw = gate_yaw_[tgt];
    double c = std::cos(gyaw), s = std::sin(gyaw);
    double dx = W[0] - gp[0], dy = W[1] - gp[1];
    o[0] = c * dx + s * dy;
    o[1] = -s * dx + c * dy;
    o[2] = W[2] - gp[2];
    o[3] = c * W[3] + s * W[4];
    o[4] = -s * W[3] + c * W[4];
    o[5] = W[5];
    double phi, theta, psi;
    euler_from_quat(W[6], W[7], W[8], W[9], phi, theta, psi);
    o[6] = wrap_pi(phi); o[7] = wrap_pi(theta); o[8] = wrap_pi(psi - gyaw);
    o[9] = W[10]; o[10] = W[11]; o[11] = W[12];
    o[12] = W[13]; o[13] = W[14]; o[14] = W[15]; o[15] = W[16];
    int nxt = target + 1;
    bool valid = loop_ ? true : (nxt < num_gates_);
    nxt %= num_gates_;
    if (valid) {
      o[16] = gate_pos_rel_[nxt][0]; o[17] = gate_pos_rel_[nxt][1]; o[18] = gate_pos_rel_[nxt][2];
      o[19] = gate_yaw_rel_[nxt];
    } else { o[16] = o[17] = o[18] = o[19] = 0; }
  }

  // perception angle (their FRD): acos( oa . (R^T rel) )
  double perc_angle(const double* W, const double* look) const {
    double rel[3] = {look[0] - W[0], look[1] - W[1], look[2] - W[2]};
    double qw = W[6], qx = W[7], qy = W[8], qz = W[9];
    // body = R^T rel
    double R00 = 1 - 2 * (qy * qy + qz * qz), R01 = 2 * (qx * qy - qz * qw), R02 = 2 * (qx * qz + qy * qw);
    double R10 = 2 * (qx * qy + qz * qw), R11 = 1 - 2 * (qx * qx + qz * qz), R12 = 2 * (qy * qz - qx * qw);
    double R20 = 2 * (qx * qz - qy * qw), R21 = 2 * (qy * qz + qx * qw), R22 = 1 - 2 * (qx * qx + qy * qy);
    double bx = R00 * rel[0] + R10 * rel[1] + R20 * rel[2];
    double by = R01 * rel[0] + R11 * rel[1] + R21 * rel[2];
    double bz = R02 * rel[0] + R12 * rel[1] + R22 * rel[2];
    double oax = std::cos(cam_), oaz = -std::sin(cam_);  // [cos,0,-sin]
    double num = bx * oax + bz * oaz;
    double den = std::sqrt(bx * bx + by * by + bz * bz) * 1.0 + 1e-9;
    double v = num / den; v = v > 1 ? 1 : (v < -1 ? -1 : v);
    return std::acos(v);
  }

  static double dist3(const double* a, const double* b) {
    double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  static double cheby3(const double* a, const double* b) {
    double dx = std::fabs(a[0] - b[0]), dy = std::fabs(a[1] - b[1]), dz = std::fabs(a[2] - b[2]);
    return std::max(dx, std::max(dy, dz));
  }

  int n_, num_gates_ = 0;
  double dt_ = 0.01, wmin_ = 341.75, wmax_ = 3100.0, k_ = 0.5, cam_ = 0.785;
  double gate_size_ = 0.8, gate_thick_ = 0.5, speed_limit_ = 99.0;
  double ground_h_ = 0.0, v_ground_ = 2.0;
  double progress_ = 1, gate_rew_ = 1, rate_pen_ = 0.001, offset_pen_ = 1, perc_pen_ = 0;
  double motor_pen_ = 0, motor_thr_ = 0, low_pen_ = 0, crash_ = 10;
  int max_steps_ = 2000;
  bool loop_ = true, has_bounds_ = false;
  double bx0_ = 0, bx1_ = 0, by0_ = 0, by1_ = 0;
  int perm_[4], inv_[4];

  std::vector<Dyn> dyn_;
  std::vector<std::array<double, 17>> world_;   // their frame mirror
  std::vector<int> target_, steps_;
  std::vector<Vec4> prev_act_;
  std::vector<std::array<double, 3>> gate_pos_, gate_pos_rel_;
  std::vector<double> gate_yaw_, gate_yaw_rel_;
};

}  // namespace mps
