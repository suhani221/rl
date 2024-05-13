import numpy as np
def initial_1(N, K):

    p0 = np.array([[0.85, 0. , 0. , 0.,  0.15],
    [0.85, 0.15, 0. , 0. , 0.],
    [0. , 0.85, 0.15, 0., 0.],
    [0. , 0. , 0.85, 0.15, 0.], 
    [0., 0. , 0. , 0.85, 0.15]])
    p1 = np.array([[0.15, 0.85, 0  , 0  , 0],
    [0.  , 0.15, 0.85, 0.  , 0   ],
    [0.  , 0.  , 0.15, 0.85, 0.  ],
    [0.  , 0.  , 0.  , 0.15, 0.85],
    [0.15, 0.  , 0.  , 0.  , 0.85]])
    QA = np.array([[0, 0, 0, 0, 0]]*K)
    QP = np.zeros((K, N))
    HA = np.array([[0, 0, 0, 0, 0]] * K)
    HP = np.zeros((K, N))
    whittle_indices_initial = np.array([[0,0,0,0,0]]*K)
    revsys = np.array([0, 1, 3, 6, 10])
    worsys = np.array([8, 5, 0, 0, -1])
    policy = np.random.randint(0, 2, (N))
    current_state = [0, 0, 0, 0, 0]
    return N, K, p1, p0, QA, QP, HA, HP, whittle_indices_initial, revsys, worsys, policy, current_state
