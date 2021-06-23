import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import torch

T = 0.1  # sampling time [s]
N = 50  # prediction horizon

accel_max = 1
omega_max = np.pi/6

def get_first_action(u_hat: torch.Tensor, s_hat: torch.Tensor, z):
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')

    states = ca.vertcat(x, y, theta, v)
    n_states = states.size()[0]

    omega = ca.SX.sym('omega')
    accel = ca.SX.sym('accel')

    controls = ca.vertcat(omega, accel)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega, accel)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N + 1)

    ## paramters
    P = ca.SX.sym('P', n_states + n_states)

    ## 不同z对应不同的参数
    zz = z // 3
    if zz == 1:
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
        R = np.array([[0.5, 0.0], [0.0, 0.05]])
    else:
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
        R = np.array([[0.5, 0.0], [0.0, 0.05]])

    ## cost function
    obj = 0
    g = []  # equal constrains
    g.append(X[:, 0] - P[:, 3])

    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i] - P[3:]).T, Q, X[:, i] - P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i + 1] - x_next_)

    opt_variables = ca.vertcat( ca.reshape(U, -1, 1), ca.reshape(X, -1, 1) )

    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []

    for _ in range(N):
        lbx.append(-accel_max)
        lbx.append(-omega_max)
        ubx.append(accel_max)
        ubx.append(omega_max)
    for _ in range(N + 1):  # note that this is different with the method using structure
        lbx.append(-2.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)

    x0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    x_m = np.zeros((n_states, N + 1))
    next_states = x_m.copy().T
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1)  # final state
    u0 = np.array([1, 2] * N).reshape(-1, 2).T  # np.ones((N, 2)) # controls

    c_p = np.concatenate((x0, xs))
    init_control = np.concatenate((u0.reshape(-1, 1), next_states.reshape(-1, 1)))
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    estimated_opt = res['x'].full()  # the feedback is in the series [u0, x0, u1, x1, ...]
    u0 = estimated_opt[:200].reshape(N, n_controls)  # (N, n_controls)
    x_m = estimated_opt[200:].reshape(N + 1, n_states)  # (N+1, n_states)

    return u0[0, :]



