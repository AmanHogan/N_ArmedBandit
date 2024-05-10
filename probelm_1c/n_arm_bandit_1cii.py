import os
import numpy as np
import matplotlib.pyplot as plt
import grapher as gf


START_FLOORS = [2,3,4,5,6]
EXIT_FLOORS = [1]
START_PROB = [.20, .20, .20, .20, .20]
EXIT_PROB = [1]
VERBOSE = True
ACTIONS = [2, 3, 4, 5, 6]
STEPS = 2000
EPSILONS = [1]
INIT_BIAS = -48
TITLE = "problem_1_c_ii"

epsilon_experiment_values = []

class ElevatorSimulation:
    """ Simulates elevator and calculates utility and reward"""
    
    def __init__(self, explore_probability):

        self.explore_probability = explore_probability # epsilon value
        self.exploit_probability = 1 - explore_probability # exploit value
        self.action_sum_history = []

        self.q_history_each = {i: [] for i in range(1, 6 + 1)}
        self.call_floors_count = {s: 0 for s in range(6 + 1)} # count of call floors
        self.exit_floors_count = {s: 0 for s in range(6 + 1)} # count of exit floors
        self.loss = {i: [] for i in range(1, 6 + 1)} # loss function values
        self.count = {a: 0 for a in ACTIONS} # count of each action
        self.q_val = {r: 0 for r in ACTIONS} # keeps avg q value of actions

        self.avg_q_values = []
        self.n_floor = 1 # location of the new elevator
        self.o_floor = 1 # location of the old elevator

        self.new_elevator_count = 0 # location of the new elevator
        self.old_elevator_count = 0 # location of the new elevator

        self.print_agent(len(ACTIONS), None, None, None, None)


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

        time_new = 5 * abs(s_c - s_e) + (2 * 7) + 5 * abs(self.n_floor - s_c)
        time_old = 7 * abs(s_c - s_e) + (2 * 7) + 7 * abs(self.o_floor - s_c)

        max_time = max(-time_new, -time_old)

        if max_time == -time_new:
            return -time_new, 'N'
        else:
            return -time_old, 'O'

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
        print(f"New elevator Floor {self.n_floor} Old elevator Floor {self.o_floor}")
        print(f"Actions Count = {self.count}")
        print(f"Q_array = {self.q_val}")
        print(f"Floors: Sc = {c_floor}, Se = {e_floor}")
        print(f"Actual Reward = r(Sc={c_floor}, Se={e_floor}) = {r}")
        print(f"Predicted Reward = r(Sc={action}, Se={e_floor})= {r_hat}")
        print(f"Q({action}) = {self.q_val[action]}")
        print("-----------------------------------")
        print(f"argmax Q_array = {max(self.q_val, key=self.q_val.get)}")
        print("-----------------------------------")
        print(f"Action floor count = {self.count}")
        print(f"Exit floor count = {self.exit_floors_count}")
        print(f"Call floor count = {self.call_floors_count}")
        print("-----------------------------------")

    def run(self):
        """Runs an agent with given epsilon value for the given amount of steps"""

        for a in ACTIONS:
            self.q_val[a] = INIT_BIAS
            self.count[a] += 1

        # Runs for the number of steps and gets q values
        for _ in range(STEPS):
            
            action = self.epsilon_greedy() # get action

            start_floor, exit_floor = self.simulate() # get start and exit floor

            actual_reward = self.reward_func(start_floor, exit_floor) # calculate r
            best_elevator = actual_reward[1] # get new or old elevator

            predicted_reward = self.reward_func(action, exit_floor) # get predicted elevator

            if best_elevator == 'N':
                self.n_floor = exit_floor
                self.new_elevator_count += 1

            else:
                self.o_floor = exit_floor
                self.old_elevator_count += 1


            self.q_func(action, actual_reward[0], predicted_reward[0])
            self.print_agent(action, exit_floor, start_floor, actual_reward[0], predicted_reward[0])
           

        epsilon_experiment_values.append(self.avg_q_values)

        if VERBOSE:
            gf.graph_actions(ACTIONS, self.q_history_each, len(epsilon_experiment_values), TITLE)
            gf.graph_loss(ACTIONS, self.loss, len(epsilon_experiment_values), TITLE)
          
        print(self.old_elevator_count, self.new_elevator_count) 
        

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

   