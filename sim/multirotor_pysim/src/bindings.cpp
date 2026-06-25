// pybind11 module: a generic batched multirotor sim (BatchSim) and the racing
// RL layer (QuadRaceBatch) that does obs/reward in C++/OpenMP. ROS-free,
// fixed-dt, deterministic. One process, one .so, two layers.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "batch_sim.hpp"
#include "quad_race.hpp"

namespace py = pybind11;
using mps::BatchSim;
using mps::QuadRaceBatch;

PYBIND11_MODULE(multirotor_pysim, mod) {
  mod.doc() = "Batched ROS-free fixed-dt multirotor sim + MonoRace RL layer.";

  py::class_<BatchSim>(mod, "BatchSim")
      .def(py::init<int>(), py::arg("num_envs"))
      .def_property_readonly("num_envs", &BatchSim::num_envs)
      .def("set_model_params", &BatchSim::set_model_params, py::arg("params"))
      .def("set_state", &BatchSim::set_state, py::arg("state"))
      .def("reset_envs", &BatchSim::reset_envs, py::arg("indices"), py::arg("state"))
      .def("get_state", &BatchSim::get_state)
      .def("get_motor_speeds", &BatchSim::get_motor_speeds)
      .def("set_motor_command", &BatchSim::set_motor_command, py::arg("motor_w"))
      .def("step", &BatchSim::step, py::arg("dt"));

  py::class_<QuadRaceBatch>(mod, "QuadRaceBatch")
      .def(py::init<int>(), py::arg("num_envs"))
      .def_property_readonly("num_envs", &QuadRaceBatch::num_envs)
      .def("set_model_params", &QuadRaceBatch::set_model_params, py::arg("params"))
      .def("set_track", &QuadRaceBatch::set_track,
           py::arg("gate_pos"), py::arg("gate_yaw"), py::arg("gate_pos_rel"),
           py::arg("gate_yaw_rel"), py::arg("bounds_xy"))
      .def("set_config", &QuadRaceBatch::set_config, py::arg("config"))
      .def("reset_envs", &QuadRaceBatch::reset_envs,
           py::arg("indices"), py::arg("states_their"), py::arg("target_gates"),
           py::arg("model_params"))
      .def("step", &QuadRaceBatch::step, py::arg("actions"));
}
