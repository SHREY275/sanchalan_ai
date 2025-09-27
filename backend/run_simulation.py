from stable_baselines3 import DQN
from sanchalan_sumo import SumoTrafficEnv

env = SumoTrafficEnv("city.sumocfg")
model = DQN.load("traffic_rl")

state = env.reset()
done = False
while not done:
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)

env.close()
