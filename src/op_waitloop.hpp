// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC
// All rights reserved.

#pragma once

/// Op namespace
namespace op {

/// Action type we'll use over and over again
using ActionFn = std::function<void()>;

/**
 *  Define a simple state messaging scheme
 *
 * n = constraint.size()
 *
 * -4 => solution found
 * -3 => update variables
 * -2 => obj_grad
 * -1 => obj_eval
 * [0,n-1] => constraint_eval
 * [n,2n-1] => constraint_eval
 *
 */

enum State : int
{
  OTHER,
  SOLUTION_FOUND   = -4,
  UPDATE_VARIABLES = -3,
  OBJ_GRAD         = -2,
  OBJ_EVAL         = -1
};

/** A functor-pattern for serialOptimizer WaitLoops
    Follows the Fluent interface pattern
*/
class WaitLoop {
public:
  using ConstraintActionFn = std::function<void(int)>;
  using UnknownActionFn    = std::function<void(int)>;

  /// Construct WaitLoop
  WaitLoop(const std::function<int()>& get_size, double& final_obj, MPI_Comm comm = MPI_COMM_WORLD)
      : get_size_(get_size), final_obj_(final_obj), comm_(comm)
  {
  }

  /// Set action to perform in Update state
  WaitLoop& onUpdate(const ActionFn& update)
  {
    update_ = update;
    return *this;
  }

  /// Set action to perform in Objective Grad state
  WaitLoop& onObjectiveGrad(const ActionFn& obj_grad)
  {
    obj_grad_ = obj_grad;
    return *this;
  }

  /// Set action to perform in Objective Eval state
  WaitLoop& onObjectiveEval(const ActionFn& obj_eval)
  {
    obj_eval_ = obj_eval;
    return *this;
  }

  /// Set action to perform in Constraint Eval state
  WaitLoop& onConstraintsEval(const ConstraintActionFn& constraints_states)
  {
    constraints_states_ = constraints_states;
    return *this;
  }

  /// Set action to perform in Constraint Grad state
  WaitLoop& onConstraintsGrad(const ConstraintActionFn& constraints_grad_states)
  {
    constraints_grad_states_ = constraints_grad_states;
    return *this;
  }

  /// Set action to perform in Solution state
  WaitLoop& onSolution(const ActionFn& solution_state)
  {
    solution_state_ = solution_state;
    return *this;
  }

  /// Set action to perform in Unknown state
  WaitLoop& onUnknown(const UnknownActionFn& unknown_state)
  {
    unknown_state_ = unknown_state;
    return *this;
  }

  // Start the whileloop
  void operator()()
  {
    int nconstraints = get_size_();  // get runtime-defered size
    while (final_obj_ == std::numeric_limits<double>::max()) {
      // set up to recieve
      int opt_state;
      op::mpi::Broadcast(opt_state, 0, comm_);

      if (opt_state == op::SOLUTION_FOUND) {
        solution_state_();
        break;
      } else if (opt_state >= nconstraints && opt_state < nconstraints * 2) {
        constraints_grad_states_(opt_state % nconstraints);
      } else if (opt_state >= 0 && opt_state < nconstraints) {
        constraints_states_(opt_state);
      } else {
        switch (opt_state) {
          case op::State::UPDATE_VARIABLES:
            update_();
            break;
          case op::State::OBJ_GRAD:
            obj_grad_();
            break;
          case op::State::OBJ_EVAL:
            obj_eval_();
            break;
          default:
            unknown_state_(opt_state);
            break;
        }
      }
    }
  }

protected:
  std::function<int()> get_size_;
  double&              final_obj_;
  MPI_Comm             comm_;
  ActionFn             update_;
  ActionFn             obj_grad_;
  ActionFn             obj_eval_;
  ConstraintActionFn   constraints_states_;
  ConstraintActionFn   constraints_grad_states_;
  ActionFn             solution_state_;
  UnknownActionFn      unknown_state_;
};

}  // namespace op
