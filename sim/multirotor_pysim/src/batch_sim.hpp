// Layer 1: generic batched multirotor integrator + shared helpers.
//
// Holds N independent multirotor::dynamics::Dynamics<double,4> and exposes
// batched numpy I/O. The model-param builder and the state row read/write are
// factored as reusable free functions so the racing layer (quad_race.hpp) can
// own its own Dynamics vector and reuse them.
//
// State row layout (17): px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r, w1,w2,w3,w4
// Frame: ENU world / FLU body (gravity [0,0,-9.81]).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>
#include <string>

#include "multirotor_dynamic_model/dynamics.hpp"

namespace mps {

namespace py = pybind11;
using Dyn = multirotor::dynamics::Dynamics<double, 4>;
using ModelParams = multirotor::model::ModelParams<double, 4>;
using MotorParams = multirotor::model::MotorParams<double>;
using Model = multirotor::model::Model<double, 4>;
using State = multirotor::state::State<double, 4>;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline ModelParams default_params() {
  ModelParams mp;
  mp.vehicle_mass = 1.0;
  mp.vehicle_inertia = Vec3(1e-2, 1e-2, 2e-2).asDiagonal();
  mp.gravity = Vec3(0.0, 0.0, -9.81);
  mp.motors_params = Model::create_quadrotor_x_config(
      1.5e-6, 2e-8, 0.13, 0.13, 0.0, 3100.0, 0.03, 1.75e-6);
  return mp;
}

// Build one ModelParams from row `r` of the per-env param dict-of-arrays.
inline ModelParams model_params_from_dict(const py::dict& d, int r) {
  auto A = [&](const char* k) -> ArrD {
    if (!d.contains(k)) throw std::runtime_error(std::string("missing param '") + k + "'");
    return d[k].cast<ArrD>();
  };
  // Cache the casts in static-free locals per call (cheap relative to a reset).
  auto mass = A("mass").unchecked<1>();
  auto iner = A("inertia").unchecked<2>();
  auto drag = A("drag").unchecked<1>();
  auto rdrag = A("rotor_drag").unchecked<1>();
  auto bquad = A("body_quad").unchecked<2>();
  auto tka = A("thrust_k_angle").unchecked<1>();
  auto tkh = A("thrust_k_hor").unchecked<1>();
  auto tar = A("thrust_aero_radius").unchecked<1>();
  auto aerom = A("aero_moment").unchecked<2>();
  auto tc = A("thrust_coeff").unchecked<1>();
  auto qc = A("torque_coeff").unchecked<1>();
  auto vmin = A("min_speed").unchecked<1>();
  auto vmax = A("max_speed").unchecked<1>();
  auto tau = A("time_constant").unchecked<1>();
  auto rinr = A("rotational_inertia").unchecked<1>();
  auto mx = A("motors_x").unchecked<2>();
  auto my = A("motors_y").unchecked<2>();
  auto mdir = A("motors_direction").unchecked<2>();

  ModelParams mp;
  mp.vehicle_mass = mass(r);
  mp.vehicle_inertia = Vec3(iner(r, 0), iner(r, 1), iner(r, 2)).asDiagonal();
  mp.vehicle_drag_coefficient = drag(r);
  mp.rotor_drag_coefficient = rdrag(r);
  mp.body_quadratic_drag = Vec3(bquad(r, 0), bquad(r, 1), bquad(r, 2));
  mp.thrust_k_angle = tka(r);
  mp.thrust_k_hor = tkh(r);
  mp.thrust_aero_radius = tar(r);
  mp.vehicle_aero_moment_coefficient =
      Vec3(aerom(r, 0), aerom(r, 1), aerom(r, 2)).asDiagonal();
  mp.gravity = Vec3(0.0, 0.0, -9.81);
  mp.force_process_noise_auto_correlation = 0.0;
  mp.moment_process_noise_auto_correlation = 0.0;
  std::vector<MotorParams> motors(4);
  for (int m = 0; m < 4; ++m) {
    motors[m].thrust_coefficient = tc(r);
    motors[m].torque_coefficient = qc(r);
    motors[m].min_speed = vmin(r);
    motors[m].max_speed = vmax(r);
    motors[m].time_constant = tau(r);
    motors[m].rotational_inertia = rinr(r);
    motors[m].motor_rotation_direction = (mdir(r, m) >= 0.0) ? 1 : -1;
    motors[m].pose = MotorParams::IsometryTypeP::Identity();
    motors[m].pose.translation() = Vec3(mx(r, m), my(r, m), 0.0);
  }
  mp.motors_params = motors;
  return mp;
}

// Apply per-env params to `dyn`. envs==nullptr => row i -> dyn[i] for all;
// else row k -> dyn[(*envs)[k]] (subset reset).
inline void set_models_from_dict(const py::dict& d, std::vector<Dyn>& dyn,
                                 const std::vector<int>* envs = nullptr) {
  int rows = envs ? static_cast<int>(envs->size()) : static_cast<int>(dyn.size());
  for (int r = 0; r < rows; ++r) {
    int e = envs ? (*envs)[r] : r;
    dyn[e].set_model(Model(model_params_from_dict(d, r)));
  }
}

// State row <-> Dynamics helpers (templated on the unchecked numpy accessor).
template <typename U>
inline void write_state_to_dyn(Dyn& dyn, const U& s, int row) {
  State st = dyn.get_state();
  st.kinematics.position = Vec3(s(row, 0), s(row, 1), s(row, 2));
  st.kinematics.linear_velocity = Vec3(s(row, 3), s(row, 4), s(row, 5));
  st.kinematics.orientation =
      Eigen::Quaterniond(s(row, 6), s(row, 7), s(row, 8), s(row, 9)).normalized();
  st.kinematics.angular_velocity = Vec3(s(row, 10), s(row, 11), s(row, 12));
  st.actuators.motor_angular_velocity = Vec4(s(row, 13), s(row, 14), s(row, 15), s(row, 16));
  dyn.set_state(st);
}

template <typename O>
inline void read_dyn_to_row(const Dyn& dyn, O& o, int row) {
  const auto& k = dyn.get_state().kinematics;
  o(row, 0) = k.position.x(); o(row, 1) = k.position.y(); o(row, 2) = k.position.z();
  o(row, 3) = k.linear_velocity.x(); o(row, 4) = k.linear_velocity.y(); o(row, 5) = k.linear_velocity.z();
  o(row, 6) = k.orientation.w(); o(row, 7) = k.orientation.x();
  o(row, 8) = k.orientation.y(); o(row, 9) = k.orientation.z();
  o(row, 10) = k.angular_velocity.x(); o(row, 11) = k.angular_velocity.y(); o(row, 12) = k.angular_velocity.z();
  const auto& w = dyn.get_state().actuators.motor_angular_velocity;
  o(row, 13) = w(0); o(row, 14) = w(1); o(row, 15) = w(2); o(row, 16) = w(3);
}

// ----------------------------------------------------------------------- //
// Generic batched integrator (unchanged behaviour from the original binding).
// ----------------------------------------------------------------------- //
class BatchSim {
public:
  explicit BatchSim(int num_envs) : n_(num_envs) {
    if (num_envs <= 0) throw std::runtime_error("num_envs must be > 0");
    ModelParams mp = default_params();
    dyn_.reserve(n_);
    cmds_.assign(n_, Vec4::Zero());
    for (int i = 0; i < n_; ++i) dyn_.emplace_back(mp, State());
  }

  int num_envs() const { return n_; }
  void set_model_params(const py::dict& d) { set_models_from_dict(d, dyn_); }

  void set_state(ArrD arr) {
    auto s = arr.unchecked<2>();
    if (s.shape(0) != n_ || s.shape(1) != 17)
      throw std::runtime_error("set_state expects (num_envs, 17)");
    for (int i = 0; i < n_; ++i) write_state_to_dyn(dyn_[i], s, i);
  }

  void reset_envs(const std::vector<int>& idx, ArrD arr) {
    auto s = arr.unchecked<2>();
    if ((int)idx.size() != s.shape(0) || s.shape(1) != 17)
      throw std::runtime_error("reset_envs expects len(idx) rows of 17");
    for (size_t k = 0; k < idx.size(); ++k) write_state_to_dyn(dyn_[idx[k]], s, (int)k);
  }

  py::array_t<double> get_state() const {
    py::array_t<double> out({n_, 17});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < n_; ++i) read_dyn_to_row(dyn_[i], o, i);
    return out;
  }

  py::array_t<double> get_motor_speeds() const {
    py::array_t<double> out({n_, 4});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < n_; ++i) {
      const auto& w = dyn_[i].get_state().actuators.motor_angular_velocity;
      for (int m = 0; m < 4; ++m) o(i, m) = w(m);
    }
    return out;
  }

  void set_motor_command(ArrD arr) {
    auto c = arr.unchecked<2>();
    if (c.shape(0) != n_ || c.shape(1) != 4)
      throw std::runtime_error("set_motor_command expects (num_envs, 4)");
    for (int i = 0; i < n_; ++i) cmds_[i] = Vec4(c(i, 0), c(i, 1), c(i, 2), c(i, 3));
  }

  void step(double dt) {
    const Vec3 zero = Vec3::Zero();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_; ++i)
      dyn_[i].process_euler_explicit(cmds_[i], dt, zero, /*enable_noise=*/false);
  }

private:
  int n_;
  std::vector<Dyn> dyn_;
  std::vector<Vec4> cmds_;
};

}  // namespace mps
