INTERSECTION = {
    "DIR": 4,
    "JUNCTION": 9,
    "DURATION": 15,
    "YELLOW": 3,
    "GREEN": 15,
    "THRESHOLD": 50,
}

NORM = {
    'STATE_NORM_CLIP': False,
    'CLIP_STATE': 1.0,
    'NORM_STATE': 0.5,  ## (1.0, 5.0)
    'REWARD_NORM_CLIP': False,
    'CLIP_REWARD': -1,
    'NORM_REWARD': 2000.0,
}

SIMULATION = {
    "RUN": "sumo",
    # "RUN": "sumo-gui",
    "EP": 0,
    "RUNTIME": 3600,
    "ID": 7,
    "CNT": 0,
    "EXECUTE": False,
    "SEED": 25
}

COP = {
    "NEIGHBOR": 8,
    "DISCOUNT": 1,
    "ADJACENT": [[1, 1, 0, 1, 0, 0, 0, 0, 0],
                 [1, 1, 1, 0, 1, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 1, 0, 0, 0],
                 [1, 0, 0, 1, 1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 1, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1, 1, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 1, 0, 1, 1, 1],
                 [0, 0, 0, 0, 0, 1, 0, 1, 1]],
    "node1": 2,
    "node2": 3,
    "node3": 2,
    "node4": 3,
    "node5": 4,
    "node6": 3,
    "node7": 2,
    "node8": 3,
    "node9": 2,
}

ROUTER_RL = {
    "ALGO": "DQN",

    "ACTOR_LR": 0.0005,
    "CRITIC_LR": 0.0001,
    "TAU": 0.01,
    "GAMMA": 0.9,  # gamma 0.9
    "UPDATE": 10,
    "REWARD_DISCOUNT": 1,

    "MIX_HIDDEN_DIM": 32,
    "AGENT_HIDDEN_DIM": 32,

    "EDGE_DIM": 4,

    "Q_LR": 0.001,  # 0.001
    "EPSILON": 0.9,
    "EPSILON_DECAY_RATE": 0.92,
    "MIN_EPSILON": 0.05,

    "BUFFER_SIZE": 50000,  # 50000
    "BATCH_SIZE": 128,  # 128

    "DECAY": True,
}


RL = {
    "ACTOR_LR": 0.0005,
    "CRITIC_LR": 0.0001,
    "TAU": 0.01,
    "GAMMA": 0.9,  # gamma 0.9
    "UPDATE": 10,
    "REWARD_DISCOUNT": 1,

    "MIX_HIDDEN_DIM": 32,
    "AGENT_HIDDEN_DIM": 32,

    "EDGE_DIM": 4,

    "Q_LR": 0.001,
    "EPSILON": 0.8,
    "EPSILON_DECAY_RATE": 0.95,
    "MIN_EPSILON": 0.1,

    "BUFFER_SIZE": 1000,
    "BATCH_SIZE": 300,

    "DECAY": False,
}

SPATIAL = {
    "ALPHA": 0.2,
    "TYPE": "FC",
}

TEMPORAL = {
    "HIDDEN_SIZE": 32,
    "BATCH_SIZE": 1,
    "BIDIRECTIONAL": False,
    "NUM_LAYERS": 1,
}

direction_map = [
    {1: "11 2", 2: "2 4",
     3: "11 22", 4: "11 2",
     5: "22 4", 6: "22 11",
     7: "2", 8: "4", 9: "22"},
    {1: "3 12", 2: "3 5",
     3: "1", 4: "12", 5: "3",
     6: "1 5", 7: "1 12",
     8: "1", 9: "5", 10: "3"},
    {1: "13 14", 2: "14 6",
     3: "2 13", 4: "13 14",
     5: "2 6", 6: "2 13",
     7: "6 14", 8: "2 6"},
    {1: "1", 2: "5", 3: "7",
     4: "1 21", 5: "1 5",
     6: "5 7", 7: "7 21"},
    {1: "2", 2: "6", 3: "8",
     4: "4", 5: "2", 6: "6",
     7: "8", 8: "4", 9: "2",
     10: "6", 11: "8", 12: "4"},
    {1: "3", 2: "15", 3: "9",
     4: "3 5", 5: "3 15",
     6: "9", 7: "5", 8: "3",
     9: "9 15", 10: "5 9"},
    {1: "4", 2: "8", 3: "19",
     4: "4 20", 5: "4 8",
     6: "19", 7: "20", 8: "4",
     9: "8 19", 10: "19 20"},
    {1: "5", 2: "9", 3: "18",
     4: "7", 5: "5", 6: "9",
     7: "18", 8: "7", 9: "5",
     10: "9", 11: "18", 12: "7"},
    {1: "6", 2: "16", 3: "17",
     4: "6 8", 5: "6 16",
     6: "17", 7: "8", 8: "6",
     9: "16 17", 10: "8 17"},
]

out_flag = [
    {1, 2, 3, 4},
    {3, 4, 5},
    {3, 4, 5, 6},
    {1, 2, 3},
    {},
    {6, 7, 8},
    {1, 2, 3, 9, 10},
    {10, 11, 12},
    {1, 2, 3, 9, 10}
]

NEIGHBOR = {
    "node1": ["W1", "4", "2", "N1"],
    "node2": ["1", "5", "3", "N2"],
    "node3": ["2", "6", "E1", "N3"],
    "node4": ["W2", "7", "5", "1"],
    "node5": ["4", "8", "6", "2"],
    "node6": ["5", "9", "E2", "3"],
    "node7": ["W3", "S1", "8", "4"],
    "node8": ["7", "S2", "9", "5"],
    "node9": ["8", "S3", "E3", "6"],
}

AGENT = {
    "node1": {
        "ACTION_DIM": 4,
        "STATE_DIM": 20,
    },
    "node2": {
        "ACTION_DIM": 4,
        "STATE_DIM": 24,
    },
    "node3": {
        "ACTION_DIM": 4,
        "STATE_DIM": 20,
    },
    "node4": {
        "ACTION_DIM": 4,
        "STATE_DIM": 24,
    },
    "node5": {
        "ACTION_DIM": 4,
        "STATE_DIM": 28,
    },
    "node6": {
        "ACTION_DIM": 4,
        "STATE_DIM": 24,
    },
    "node7": {
        "ACTION_DIM": 4,
        "STATE_DIM": 24,
    },
    "node8": {
        "ACTION_DIM": 4,
        "STATE_DIM": 28,
    },
    "node9": {
        "ACTION_DIM": 4,
        "STATE_DIM": 24,
    },
}
