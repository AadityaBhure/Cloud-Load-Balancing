import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# ENVIRONMENT (Cloud Simulation)
# ==============================

class Server:
    def __init__(self, id):
        self.id = id
        self.cpu = random.uniform(0.2, 0.8)
        self.queue = random.randint(1, 10)
        self.latency = random.uniform(10, 100)

    def update(self):
        self.cpu = np.clip(self.cpu + random.uniform(-0.1, 0.1), 0, 1)
        self.queue = max(0, self.queue + random.randint(-2, 3))
        self.latency = np.clip(self.latency + random.uniform(-5, 5), 5, 200)

# ==============================
# FUZZY SYSTEM (HFS Simplified)
# ==============================

class FuzzySystem:
    def __init__(self):
        # parameters controlled by PPO
        self.cpu_weight = 0.4
        self.queue_weight = 0.3
        self.latency_weight = 0.3

    def compute_score(self, server):
        cpu_score = 1 - server.cpu
        queue_score = 1 / (1 + server.queue)
        latency_score = 1 / (1 + server.latency)

        score = (
            self.cpu_weight * cpu_score +
            self.queue_weight * queue_score +
            self.latency_weight * latency_score
        )
        return score

    def update_params(self, delta):
        self.cpu_weight += delta[0]
        self.queue_weight += delta[1]
        self.latency_weight += delta[2]

        # normalize
        total = self.cpu_weight + self.queue_weight + self.latency_weight
        self.cpu_weight /= total
        self.queue_weight /= total
        self.latency_weight /= total


# ==============================
# PPO MODEL
# ==============================

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ==============================
# SIMULATION CONTROLLER
# ==============================

class Controller:
    def __init__(self, num_servers=5):
        self.servers = [Server(i) for i in range(num_servers)]
        self.fuzzy = FuzzySystem()

        self.model = PPO(input_dim=3, output_dim=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_state(self):
        avg_cpu = np.mean([s.cpu for s in self.servers])
        avg_queue = np.mean([s.queue for s in self.servers])
        avg_latency = np.mean([s.latency for s in self.servers])
        return np.array([avg_cpu, avg_queue, avg_latency], dtype=np.float32)

    def select_server(self):
        scores = [self.fuzzy.compute_score(s) for s in self.servers]
        return np.argmax(scores)

    def step(self):
        # FAST LOOP
        selected = self.select_server()
        server = self.servers[selected]

        # simulate task
        server.cpu += 0.1
        server.queue += 1

        # update all servers
        for s in self.servers:
            s.update()

        # compute reward (inverse latency + low queue + low cpu)
        reward = -(
            server.latency * 0.5 +
            server.queue * 0.3 +
            server.cpu * 0.2
        )

        return reward

    def train(self, episodes=200):
        for ep in range(episodes):
            state = self.get_state()
            state_tensor = torch.tensor(state)

            action = self.model(state_tensor)
            delta = action.detach().numpy()

            # apply update (SLOW LOOP)
            self.fuzzy.update_params(delta * 0.01)

            reward = self.step()

            loss = -reward * action.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if ep % 20 == 0:
                print(f"Episode {ep}, Reward: {reward:.2f}, Weights: "
                      f"{self.fuzzy.cpu_weight:.2f}, "
                      f"{self.fuzzy.queue_weight:.2f}, "
                      f"{self.fuzzy.latency_weight:.2f}")


# ==============================
# RUN SIMULATION
# ==============================

if __name__ == "__main__":
    controller = Controller()
    controller.train(episodes=200)