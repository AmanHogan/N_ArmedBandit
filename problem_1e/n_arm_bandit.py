import os
import numpy as np
import matplotlib.pyplot as plt
import grapher as gf


VERBOSE = True
ACTIONS = [2, 3, 4, 5, 6]  # action choices
EXPERIMENTS = 10
STEPS = 1000
EPSILONS = [.1, .5, 1]

TITLE = "_1_ei"

epsilon_experiment_values = []

class ElevatorSimulation:
    
    def __init__(self, explore_probability):

        self.explore_probability = explore_probability
        self.exploit_probability = 1 - explore_probability
        self.q_history_each = {i: [] for i in range(1, 6 + 1)}
        self.action_sum_history = []
        self.call_floors_count = {s: 0 for s in range(6 + 1)}
        self.exit_floors_count = {s: 0 for s in range(6 + 1)}
        self.loss = {i: [] for i in range(1, 6 + 1)}
        self.count = {a: 0 for a in ACTIONS}
        self.q_val = {r: 0 for r in ACTIONS}
        self.avg_q_values = [] # List of avg q values over all actions for each step 

    def q_func(self, action, actual, r_hat):
        """
        Utility function (Expected long term reward)
        Q(a)_k+1 = Q(a)_k + (1/k+1) * (r_k+1 - Q(a)_k )

        Args:
            action (int): floor chosen
            r_k (int): actual reward from simulation
            r_hat (int): predicted reward (not used)
        """

        self.q_val[action] = self.q_val[action] + (1 / (self.count[action] + 1) ) * (actual - self.q_val[action])
        self.loss[action].append((1 / (self.count[action] + 1) ) * (actual - r_hat))
        self.count[action] += 1

        self.q_history_each[action].append(self.q_val[action])
        self.avg_q_values.append(sum(self.q_val.values())/len(ACTIONS))
        
    def reward_func(self, s_c, s_e):
        """
        Calcualtes reward based on minimum time it takes between old and new elevator.
        Args:
            s_c (int): call elevator
            s_e (int): exit elevator

        Returns:
            reward: min time between call and exit floor (-)
        """

        time_new = 5 * abs(s_c - s_e) + (2 * 7) 
        time_old = 7 * abs(s_c - s_e) + (2 * 7)
        max_reward = max(-time_new, -time_old)

        return int(max_reward)
    
    def epsilon_greedy(self):
        """
        Epsilon greedy policy.
        Choose random floor with P() = epsilon \n
        Choose action that provided max util with P() = 1 - epsilon \n
        Returns: int: action chosen
        """

        policy = np.random.choice(['explore', 'exploit'], 1, p=[self.explore_probability, self.exploit_probability])

        if policy == 'explore':
            return np.random.choice(ACTIONS)
        else:
            return max(self.q_val, key=self.q_val.get)

    def simulate(self):
        """
        Simulates people chosing an elevator. 
        Gets a random call and exit floor from the list call and exit floors\n
        Returns: (int, int): call floor and exit floor the person chose in simualtion
        """
        call_floor = 0
        exit_floor = 0  


        worker = np.random.choice(['night', 'day'], 1, p=[.10, .90])

        if worker == 'night':
            START_FLOORS = [2,3,4,5,6]
            EXIT_FLOORS = [1]
            START_PROB = [.20, .20, .20, .20, .20]
            EXIT_PROB = [1]


            call_floor = np.random.choice(START_FLOORS, 1, p=START_PROB)
            exit_floor = np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB)

        # day workers
        else:
            START_FLOORS = [1]
            EXIT_FLOORS = [2, 3, 4, 5, 6 ]
            START_PROB = [1]
            EXIT_PROB = [.20, .20, .20, .20, .20]
            call_floor = np.random.choice(START_FLOORS, 1, p=START_PROB)
            exit_floor = np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB)

        

        self.exit_floors_count[int(exit_floor)] += 1
        self.call_floors_count[int(call_floor)] += 1

        return call_floor, exit_floor
     
    def print_agent(self, action, e_floor, c_floor, r, r_hat):
        """Prints detaisl about agent

        Args:
            action (int): action chosen
            e_floor (int): exit floor
            c_floor (int): call floor
            r (int): reward actual
            r_hat (int): reward prediction
        """

        print("-----------------------------------")
        print(f"Actions Count = {self.count}")
        print(f"Q_array = {self.q_val}")
        print(f"Floors: Sc = {c_floor}, Se = {e_floor}")
        print(f"Actual Reward = r(Sc={c_floor}, Se={e_floor}) = {r}")
        print(f"Predicted Reward = r(Sc={action}, Se={e_floor})= {r_hat}")
        print(f"Q({action}) = {self.q_val[action]}")
        print("-----------------------------------")
        print(f"argmax Q_array = {max(self.q_val, key=self.q_val.get)}")
        print("-----------------------------------")

    def run(self):
        """Runs an agent with given epsilon value for the given amount of steps"""

        # Gets initial values for each action
        for _ in range(EXPERIMENTS):
            for a in ACTIONS:
                start_floor, exit_floor = self.simulate()
                actual_reward = self.reward_func(start_floor, exit_floor)
                predicted_reward = self.reward_func(a, exit_floor)
                self.q_func(a, actual_reward, predicted_reward)

        # Runs for the number of steps and gets q values
        for _ in range(STEPS):
            action = self.epsilon_greedy()
            start_floor, exit_floor = self.simulate()
            actual_reward = self.reward_func(start_floor, exit_floor)
            predicted_reward = self.reward_func(action, exit_floor)
            self.q_func(action, actual_reward, predicted_reward)
            self.print_agent(action, exit_floor, start_floor, actual_reward, predicted_reward)
           
        print(f"Action floor count = {self.count}")
        print(f"Exit floor count = {self.exit_floors_count}")
        print(f"Call floor count = {self.call_floors_count}")

        epsilon_experiment_values.append(self.avg_q_values)

        if VERBOSE:
            gf.graph_actions(ACTIONS, self.q_history_each, len(epsilon_experiment_values), TITLE)
            gf.graph_loss(ACTIONS, self.loss, len(epsilon_experiment_values), TITLE)
          
        
        

if __name__ == "__main__":

    agents = []
    best_floors = {}

    for e in range(len(EPSILONS)):
        agents.append(ElevatorSimulation(EPSILONS[e]))
        agents[e].run()

    gf.graph_epsilons(epsilon_experiment_values, EPSILONS, TITLE)
    
    print("-----------------------------------")
    for k in range(len(epsilon_experiment_values)):
        print(f"Epsilon ({agents[k].explore_probability}) with a total average reward of {round(np.mean(epsilon_experiment_values[k]),4)}. Best start floor = {max(agents[k].q_val, key=agents[k].q_val.get)}")
        best_floors[round(np.mean(epsilon_experiment_values[k]),4)] = max(agents[k].q_val, key=agents[k].q_val.get)
    print("-----------------------------------")
    print(f"Best floor was {best_floors[max(best_floors)]} with avg utility of {max(best_floors)} over {STEPS} steps and {len(epsilon_experiment_values)} experiments.")

    