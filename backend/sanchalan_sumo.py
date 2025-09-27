import traci
import numpy as np
import gym
from gym import spaces

class SumoTrafficEnv(gym.Env):
    def __init__(self, sumo_cfg="city.sumocfg", max_steps=1000):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        traci.start(["sumo", "-c", self.sumo_cfg, "--no-step-log", "true"])
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        if self._ambulance_present():
            self._set_green_for_ambulance()
        else:
            self._set_traffic_lights(action)

        traci.simulationStep()
        self.step_count += 1
        state = self._get_state()
        reward = -sum(state[:-1])  # penalize queue length
        if state[-1] == 1:  # ambulance flag
            reward += 100
        done = self.step_count >= self.max_steps
        return state, reward, done, {}

    def _get_state(self):
        lanes = ["N2TL_0", "S2TL_0", "E2TL_0", "W2TL_0"]
        counts = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
        amb_flag = 1 if self._ambulance_present() else 0
        return np.array(counts + [amb_flag], dtype=np.float32)

    def _ambulance_present(self):
        return any("ambulance" in v for v in traci.vehicle.getIDList())

    def _set_green_for_ambulance(self):
        traci.trafficlight.setRedYellowGreenState("TL", "GGGG")

    def _set_traffic_lights(self, action):
        phases = ["GrGr", "rGrG", "GGrr", "rrGG"]
        traci.trafficlight.setRedYellowGreenState("TL", phases[action])

    def close(self):
        traci.close()
