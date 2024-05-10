import os
import numpy as np
import matplotlib.pyplot as plt



def graph_loss(actions, losses, number, title):
    for action in actions:
        plt.plot([i for i in range(len(losses[action]))], losses[action], label=f'Action {action}')

    plt.xlabel('Steps')
    plt.ylabel('loss')
    plt.title('loss Over Time for Each Action')
    plt.legend()
    plt.savefig(title + '_loss_' + str(number) + '_.png')
    plt.show()
        
def graph_actions(actions, util_history, number, title):

    for action in actions:
        plt.plot([i for i in range(len(util_history[action]))], util_history[action], label=f'Action {action}')
    
    plt.xlabel('Steps [#]')
    plt.ylabel('Utility [#]')
    plt.title('Utility Over Time for Each Action')
    plt.legend()
    plt.savefig(title + '_utility_' + str(number) + '_.png')
    plt.show()


def graph_epsilons(experiemnts, epsilons, title):
    for k in range(len(experiemnts)): 
        plt.plot([i for i in range(len(experiemnts[k]))], experiemnts[k], label=str(epsilons[k]))
        
    plt.xlabel('Steps')
    plt.ylabel('Total Average Rewards')
    plt.title('Rewards for Epsilons')
    plt.legend()
    plt.savefig(title + '_epsilons.png')
    plt.show()