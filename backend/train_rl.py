from stable_baselines3 import DQN
from sanchalan_sumo import SumoTrafficEnv

env = SumoTrafficEnv("city.sumocfg")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("traffic_rl")
