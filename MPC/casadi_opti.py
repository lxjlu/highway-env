import casadi as ca
import numpy as np


def get_first_action(s0, x_f, u0, next_states):
    # 参数
    N = 50
    T = 0.1
    v_max = 30.0
    omega_max = np.pi / 3
    accel_max = 5.0

    opti = ca.Opti()
    # control variables
    opt_controls = opti.variable(N, 2)
    omega = opt_controls[:, 0]
    accel = opt_controls[:, 1]
    # state variables
    opt_states = opti.variable(N + 1, 4)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    v = opt_states[:, 3]
    # parameters
    opt_x0 = opti.parameter(4)  # initial state
    opt_xs = opti.parameter(4)  # target_y and target speed
    # model
    f = lambda x_, u_: ca.vertcat(*[
        x_[3] * ca.cos(x_[2]),
        x_[3] * ca.sin(x_[2]),
        u_[0],
        u_[1]
    ])
    # init condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    # dynamic condition
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * T
        opti.subject_to(opt_states[i + 1, :] == x_next)

    Q = np.array([[8.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 1000.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    R = np.array([[1, 0.0], [0.0, 1]])

    obj = 0  #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - opt_xs.T), Q, (opt_states[i, :] - opt_xs.T).T]) + ca.mtimes(
            [opt_controls[i, :], R, opt_controls[i, :].T])

    opti.minimize(obj)

    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-accel_max, accel, accel))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-6,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)

    final_state = x_f
    opti.set_value(opt_xs, final_state)
    t0 = 0
    init_state = s0

    current_state = init_state.copy()

    x_c = []  # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    # sim_time = 20.0
    sim_time = 5.0

    opti.set_value(opt_x0, current_state)
    # set optimizing target withe init guess
    opti.set_initial(opt_controls, u0)  # (N, 2)
    opti.set_initial(opt_states, next_states)  # (N+1, 4)
    sol = opti.solve()
    u_res = sol.value(opt_controls)
    next_states_pred = sol.value(opt_states)

    return u_res, next_states_pred
