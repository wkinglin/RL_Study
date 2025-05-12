import time
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from utils import kinematic, constraint

# This demo references: https://github.com/RyYAO98/CILQR_Motion_Planning

# CILQR parameter
DT = 0.1
HORIZON_LENGTH = 60
MAX_ACC = 2.0


class CILQR:
    def __init__(self) -> None:
        # planning-related settings
        self.N = HORIZON_LENGTH
        self.dt = DT
        self.nx = 4
        self.nu = 2
        self.state_weight = np.array(
            [[1.0, 0., 0., 0.],
             [0., 1.0, 0., 0.],
             [0., 0., 0.5, 0.],
             [0., 0., 0., 0.]])
        self.ctrl_weight = np.array(
            [[1.0, 0.0],
             [0.0, 1.0]])
        self.exp_q1 = 5.5
        self.exp_q2 = 5.75

        # iteration-related settings
        self.max_iter = 50
        self.init_lamb = 20
        self.lamb_decay = 0.7
        self.lamb_amplify = 2.0
        self.max_lamb = 10000.0
        self.alpha_options = [1., 0.5, 0.25, 0.125, 0.0625]
        self.tol = 0.001

        # ego vehicle-related settings
        self.wheelbase = WB
        self.width = WIDTH
        self.length = LENGTH
        self.velo_max = 10.0
        self.velo_min = 0.0
        self.yaw_lim = 3.14
        self.acc_max = MAX_ACC
        self.acc_min = -MAX_ACC
        self.stl_lim = 1.57


    def get_init_traj(self, x0):
        # zero ctrl initialization
        init_u = np.zeros((self.nu, self.N))
        init_x = kinematic.const_velo_prediction(x0, self.N, self.dt, self.wheelbase)

        return init_u, init_x


    def get_total_cost(self, u, x, ref_waypoints, ref_velo, obs_attrs, obs_preds):
        num_obstacles = obs_attrs.shape[0]

        # part 1: costs included in the prime objective
        ref_exact_points = kinematic.get_ref_exact_points(x[:2], ref_waypoints)
        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.N + 1, ref_velo),
            np.zeros(self.N + 1)
        ])

        states_devt = np.sum(((x - ref_states).T @ self.state_weight) * (x - ref_states).T)
        ctrl_energy = np.sum((u.T @ self.ctrl_weight) * u.T)

        J_prime = states_devt + ctrl_energy

        # part 2: costs of the barrier function terms
        J_barrier = 0.
        for k in range(1, self.N + 1):
            u_k_minus_1 = u[:, k - 1]
            x_k = x[:, k]  # there is no need to count for the current state cost
            obs_preds_k = obs_preds[:, :, k]

            # acceleration constraints
            acc_up_constr = constraint.get_bound_constr(u_k_minus_1[0],
                                                        self.acc_max, bound_type='upper')
            acc_lo_constr = constraint.get_bound_constr(u_k_minus_1[0],
                                                        self.acc_min, bound_type='lower')

            # steering angle constraints
            stl_up_constr = constraint.get_bound_constr(u_k_minus_1[1],
                                                        self.stl_lim, bound_type='upper')
            stl_lo_constr = constraint.get_bound_constr(u_k_minus_1[1],
                                                        -self.stl_lim, bound_type='lower')

            # velocity constraints
            velo_up_constr = constraint.get_bound_constr(x_k[2], self.velo_max, bound_type='upper')
            velo_lo_constr = constraint.get_bound_constr(x_k[2], self.velo_min, bound_type='lower')

            # obstacle avoidance constraints
            obs_front_constr_list, obs_rear_constr_list = [], []
            for j in range(num_obstacles):
                obs_j_pred_k = obs_preds_k[j]
                obs_j_attr = obs_attrs[j]
                obs_j_front_constr, obs_j_rear_constr = constraint.get_obstacle_avoidance_constr(
                    x_k, obs_j_pred_k, self.wheelbase, self.width, obs_j_attr)
                obs_front_constr_list.append(obs_j_front_constr)
                obs_rear_constr_list.append(obs_j_rear_constr)

            J_barrier_k = constraint.exp_barrier(acc_up_constr, self.exp_q1, self.exp_q2) \
                          + constraint.exp_barrier(acc_lo_constr, self.exp_q1, self.exp_q2) \
                          + constraint.exp_barrier(stl_up_constr, self.exp_q1, self.exp_q2) \
                          + constraint.exp_barrier(stl_lo_constr, self.exp_q1, self.exp_q2) \
                          + constraint.exp_barrier(velo_up_constr, self.exp_q1, self.exp_q2) \
                          + constraint.exp_barrier(velo_lo_constr, self.exp_q1, self.exp_q2) \
                          + np.sum([constraint.exp_barrier(ofc, self.exp_q1, self.exp_q2) \
                                    for ofc in obs_front_constr_list]) \
                          + np.sum([constraint.exp_barrier(orc, self.exp_q1, self.exp_q2) \
                                    for orc in obs_rear_constr_list])

            J_barrier += J_barrier_k

        # Get the total cost
        J_tot = J_prime + J_barrier

        return J_tot


    def get_total_cost_derivatives_and_Hessians(
            self, u, x, ref_waypoints, ref_velo, obs_attrs, obs_preds):
        ref_exact_points = kinematic.get_ref_exact_points(x[ : 2], ref_waypoints)
        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.N + 1, ref_velo),
            np.zeros(self.N + 1)
        ])
        num_obstacles = obs_attrs.shape[0]

        # part 1: cost derivatives due to the prime objective
        l_u_prime = 2 * (u.T @ self.ctrl_weight).T
        l_uu_prime = np.repeat((2 * self.ctrl_weight)[:, :, np.newaxis], self.N, axis=2)
        l_x_prime = 2 * ((x - ref_states).T @ self.state_weight).T
        l_xx_prime = np.repeat((2 * self.state_weight)[:, :, np.newaxis], self.N + 1, axis=2)

        # part 2: cost derivatives due to the barrier terms
        l_u_barrier = np.zeros((self.nu, self.N))
        l_uu_barrier = np.zeros((self.nu, self.nu, self.N))
        l_x_barrier = np.zeros((self.nx, self.N + 1))
        l_xx_barrier = np.zeros((self.nx, self.nx, self.N + 1))

        for k in range(self.N + 1):
            # ---- Ctrl: only N steps ----
            if k < self.N:
                u_k = u[:, k]

                # acceleration constraints derivatives and Hessians
                acc_up_constr = constraint.get_bound_constr(u_k[0], self.acc_max, bound_type='upper')
                acc_up_constr_over_u = np.array([1., 0.])
                acc_up_barrier_over_u, acc_up_barrier_over_uu = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        acc_up_constr, acc_up_constr_over_u, self.exp_q1, self.exp_q2)

                acc_lo_constr = constraint.get_bound_constr(u_k[0], self.acc_min, bound_type='lower')
                acc_lo_constr_over_u = np.array([-1., 0.])
                acc_lo_barrier_over_u, acc_lo_barrier_over_uu = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        acc_lo_constr, acc_lo_constr_over_u, self.exp_q1, self.exp_q2)

                # steering angle constraints derivatives and Hessians
                stl_up_constr = constraint.get_bound_constr(u_k[1], self.stl_lim, bound_type='upper')
                stl_up_constr_over_u = np.array([0., 1.])
                stl_up_barrier_over_u, stl_up_barrier_over_uu = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        stl_up_constr, stl_up_constr_over_u, self.exp_q1, self.exp_q2)

                stl_lo_constr = constraint.get_bound_constr(u_k[1], -self.stl_lim, bound_type='lower')
                stl_lo_constr_over_u = np.array([0., -1.])
                stl_lo_barrier_over_u, stl_lo_barrier_over_uu = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        stl_lo_constr, stl_lo_constr_over_u, self.exp_q1, self.exp_q2)

                # fill the ctrl-related spaces
                l_u_barrier[:, k] = acc_up_barrier_over_u + acc_lo_barrier_over_u \
                                  + stl_up_barrier_over_u + stl_lo_barrier_over_u

                l_uu_barrier[:, :, k] = acc_up_barrier_over_uu + acc_lo_barrier_over_uu \
                                      + stl_up_barrier_over_uu + stl_lo_barrier_over_uu

            # ---- State: (N + 1) steps ----
            x_k = x[:, k]
            obs_pred_k = obs_preds[:, :, k]

            # velocity constraints derivatives and Hessians
            velo_up_constr = constraint.get_bound_constr(x_k[2], self.velo_max, bound_type='upper')
            velo_up_constr_over_x = np.array([0., 0., 1., 0.])
            velo_up_barrier_over_x, velo_up_barrier_over_xx = \
                constraint.exp_barrier_derivative_and_Hessian(
                    velo_up_constr, velo_up_constr_over_x, self.exp_q1, self.exp_q2)

            velo_lo_constr = constraint.get_bound_constr(x_k[2], self.velo_min, bound_type='lower')
            velo_lo_constr_over_x = np.array([0., 0., -1., 0.])
            velo_lo_barrier_over_x, velo_lo_barrier_over_xx = \
                constraint.exp_barrier_derivative_and_Hessian(
                    velo_lo_constr, velo_lo_constr_over_x, self.exp_q1, self.exp_q2)

            # obstacle avoidance constraints derivatives and Hessians
            obs_front_barrier_over_x_list, obs_front_barrier_over_xx_list = [], []
            obs_rear_barrier_over_x_list, obs_rear_barrier_over_xx_list = [], []
            for j in range(num_obstacles):
                obs_j_pred_k = obs_pred_k[j]
                obs_j_attr = obs_attrs[j]

                obs_j_front_constr, obs_j_rear_constr = constraint.get_obstacle_avoidance_constr(
                    x_k, obs_j_pred_k, self.wheelbase, self.width, obs_j_attr
                )
                obs_j_front_constr_over_x, obs_j_rear_constr_over_x = \
                    constraint.get_obstacle_avoidance_constr_derivatives(
                        x_k, obs_j_pred_k, self.wheelbase, self.width, obs_j_attr)
                obs_j_front_barrier_over_x, obs_j_front_barrier_over_xx = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        obs_j_front_constr, obs_j_front_constr_over_x, self.exp_q1, self.exp_q2)
                obs_j_rear_barrier_over_x, obs_j_rear_barrier_over_xx = \
                    constraint.exp_barrier_derivative_and_Hessian(
                        obs_j_rear_constr, obs_j_rear_constr_over_x, self.exp_q1, self.exp_q2)

                obs_front_barrier_over_x_list.append(obs_j_front_barrier_over_x)
                obs_front_barrier_over_xx_list.append(obs_j_front_barrier_over_xx)
                obs_rear_barrier_over_x_list.append(obs_j_rear_barrier_over_x)
                obs_rear_barrier_over_xx_list.append(obs_j_rear_barrier_over_xx)

            # fill the state-related spaces
            l_x_barrier[:, k] = velo_up_barrier_over_x + velo_lo_barrier_over_x \
                                + np.sum(obs_front_barrier_over_x_list, axis=0) \
                                + np.sum(obs_rear_barrier_over_x_list, axis=0)

            l_xx_barrier[:, :, k] = velo_up_barrier_over_xx + velo_lo_barrier_over_xx \
                                    + np.sum(obs_front_barrier_over_xx_list, axis=0) \
                                    + np.sum(obs_rear_barrier_over_xx_list, axis=0)

        # Get the results by combining both components
        l_u = l_u_prime + l_u_barrier
        l_uu = l_uu_prime + l_uu_barrier
        l_x = l_x_prime + l_x_barrier
        l_xx = l_xx_prime + l_xx_barrier

        l_ux = np.zeros((self.nu, self.nx, self.N))

        return l_u, l_uu, l_x, l_xx, l_ux


    def backward_pass(self, u, x, lamb, ref_waypoints, ref_velo, obs_attrs, obs_preds):
        l_u, l_uu, l_x, l_xx, l_ux = \
            self.get_total_cost_derivatives_and_Hessians(
                u, x, ref_waypoints, ref_velo, obs_attrs, obs_preds)
        df_dx, df_du = \
            kinematic.get_kinematic_model_derivatives(x, u, self.dt, self.wheelbase, self.N)

        delt_V = 0
        V_x = l_x[:, -1]
        V_xx = l_xx[:, :, -1]

        d = np.zeros((self.nu, self.N))
        K = np.zeros((self.nu, self.nx, self.N))

        regu_I = lamb * np.eye(V_xx.shape[0])

        # Run a backwards pass from N-1 control step
        for i in range(self.N - 1, -1, -1):
            # This part of the implementation references:
            # https://github.com/Bharath2/iLQR/blob/main/ilqr/controller.py

            # Q_terms
            Q_x = l_x[:, i] + df_dx[:, :, i].T @ V_x
            Q_u = l_u[:, i] + df_du[:, :, i].T @ V_x
            Q_xx = l_xx[:, :, i] + df_dx[:, :, i].T @ V_xx @ df_dx[:, :, i]
            Q_uu = l_uu[:, :, i] + df_du[:, :, i].T @ V_xx @ df_du[:, :, i]
            Q_ux = l_ux[:, :, i] + df_du[:, :, i].T @ V_xx @ df_dx[:, :, i]

            # gains
            df_du_regu = df_du[:, :, i].T @ regu_I
            Q_ux_regu = Q_ux + df_du_regu @ df_dx[:, :, i]
            Q_uu_regu = Q_uu + df_du_regu @ df_du[:, :, i]
            Q_uu_inv = np.linalg.inv(Q_uu_regu)

            d[:, i] = -Q_uu_inv @ Q_u
            K[:, :, i] = -Q_uu_inv @ Q_ux_regu

            # Update value function for next time step
            V_x = Q_x + K[:, :, i].T @ Q_uu @ d[:, i] + K[:, :, i].T @ Q_u + Q_ux.T @ d[:, i]
            V_xx = Q_xx + K[:, :, i].T @ Q_uu @ K[:, :, i] + K[:, :, i].T @ Q_ux + Q_ux.T @ K[:, :, i]

            # expected cost reduction
            delt_V += 0.5 * d[:, i].T @ Q_uu @ d[:, i] + d[:, i].T @ Q_u

        return d, K, delt_V


    def forward_pass(self, u, x, d, K, alpha):
        new_u = np.zeros((self.nu, self.N))
        new_x = np.zeros((self.nx, self.N + 1))
        new_x[:, 0] = x[:, 0]

        for i in range(self.N):
            new_u_i = u[:, i] + alpha * d[:, i] + K[:, :, i] @ (new_x[:, i] - x[:, i])
            new_u[:, i] = constraint.get_bounded_ctrl(
                new_u_i, new_x[:, i], self.acc_max, self.acc_min,
                self.stl_lim, self.velo_min, self.velo_max
            )
            new_x[:, i + 1] = kinematic.kinematic_propagate(
                new_x[:, i], new_u[:, i], self.dt, self.wheelbase
            )

        return new_u, new_x


    def iter_step(self, u, x, J, lamb, ref_waypoints, ref_velo, obs_attrs, obs_preds):
        d, K, expc_redu = self.backward_pass(
                            u, x, lamb, ref_waypoints, ref_velo, obs_attrs, obs_preds)

        iter_effective_flag = False
        new_u, new_x, new_J = \
            np.zeros((self.nu, self.N)), np.zeros((self.nx, self.N + 1)), sys.float_info.max

        for alpha in self.alpha_options:
            new_u, new_x = self.forward_pass(u, x, d, K, alpha)
            new_J = self.get_total_cost(new_u, new_x, ref_waypoints, ref_velo, obs_attrs, obs_preds)

            if new_J < J:
                iter_effective_flag = True
                break

        return new_u, new_x, new_J, iter_effective_flag

    def solve(self, x0, ref_waypoints, ref_velo, obs_attrs, obs_preds):
        init_u, init_x = self.get_init_traj(x0)
        J = self.get_total_cost(init_u, init_x, ref_waypoints, ref_velo, obs_attrs, obs_preds)
        u, x = init_u, init_x

        lamb = self.init_lamb

        for itr in range(self.max_iter):
            new_u, new_x, new_J, iter_effective_flag = self.iter_step(
                u, x, J, lamb, ref_waypoints, ref_velo, obs_attrs, obs_preds)

            if iter_effective_flag:
                x = new_x
                u = new_u
                J_old = J
                J = new_J

                if abs(J - J_old) < self.tol:
                    print(f'Tolerance condition satisfied. {itr}')
                    break

                lamb *= self.lamb_decay
            else:
                lamb *= self.lamb_amplify

                if lamb > self.max_lamb:
                    print('Regularization parameter reached the maximum.')
                    break

        return u, x
        