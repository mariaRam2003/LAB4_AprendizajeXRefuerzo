import numpy as np
import pprint

class State(object):
    """
    Represents a state or a point in the grid.

    coord: coordinate in grid world
    """
    def __init__(self, coord, is_terminal):
        self.coord = coord
        self.action_state_transitions = self._getActionStateTranstions()
        self.is_terminal = is_terminal
        self.reward = 5 if is_terminal else -1

    # Returns a dictionary mapping each action to the following state
    # it would put the agent in from the currrent state
    def _getActionStateTranstions(self):
        action_state_transitions = {}
        # Action 0 - up
        if self._isFirstRowState():
            action_state_transitions[0] = self.coord
        else:
            # prev row, same col
            action_state_transitions[0] = (self.coord[0]-1, self.coord[1])

        # Action 1 - right
        if self._isLastColState():
            action_state_transitions[1] = self.coord
        else:
            # same row, next col
            action_state_transitions[1] = (self.coord[0], self.coord[1]+1)

        # Action 2 - down
        if self._isLastRowState():
            action_state_transitions[2] = self.coord
        else:
            # next row, same col
            action_state_transitions[2] = (self.coord[0]+1, self.coord[1])

        # Action 3 - left
        if self._isFirstRowState():
            action_state_transitions[3] = self.coord
        else:
            # same row, prev col
            action_state_transitions[3] = (self.coord[0], self.coord[1]-1)

        return action_state_transitions

    def _isFirstRowState(self):
        return self.coord[0] == 0

    def _isLastRowState(self):
        return self.coord[0] == 3

    def _isFirstColState(self):
        return self.coord[1] == 0

    def _isLastColState(self):
        return self.coord[1] == 3

    # Returns if the current state is a terminal state
    def isTerminal(self):
        return self.is_terminal

    # Gets the action required to move the agent from the current state
    # to some state s2. If the agent cannot move to s2 it returns None
    def getActionTransiton(self, s2):
        for action, next_state in self.action_state_transitions.items():
            if next_state == s2.coord:
                return action
        return None

    # Returns the likelihood of ending up in state s_prime after taking
    # action a from the current state
    def getNextStateLikelihood(self, a, s_prime):
        if self.action_state_transitions[a] == s_prime.coord:
            return 1
        else:
            return 0

    # Returrn the reward for stepping into this state
    def getReward(self):
        return self.reward


class DynamicProgrammingAgent(object):
    """
    Base implementation of a Dynamic Programming Agent for the Grid World Problem

    env: Gym env the agent will be trained on
    """
    def __init__(self, gamma):
        self.gamma = gamma

        # of states and actions for the grid world problem
        self.num_states = 16
        self.num_actions = 4

    # Prints the values of each state on the grid
    def _printStateValues(self, V):
        grid = np.zeros([4,4])

        for state, value in V.items():
            x = state.coord[0]
            y = state.coord[1]
            grid[x,y] = value

        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')

    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.zeros([4,4])

        for state, actions in pi.items():
            x = state.coord[0]
            y = state.coord[1]
            action = np.argmax(actions)
            grid[x,y] = action

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = ''
                if (row_index == 0 and col_index == 0) or (row_index == 3 and col_index == 3):
                    arrow_grid_row.append(arrow_char)
                else:
                    if action == 0:
                        arrow_char = '↑'
                    elif action == 1:
                        arrow_char = '→'
                    elif action == 2:
                        arrow_char = '↓'
                    elif action == 3:
                        arrow_char = '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid)
        print('\n')

    # # Initialize the states (S), state value function (V), and the policy (pi)
    def initSVAndPi(self):
        self.S = []
        V = {}
        pi = {}
        for r in range(4):
            for c in range(4):
                # Create the state
                is_terminal = False
                if (r == 0 and c == 0) or (r == 3 and c == 3):
                    is_terminal = True
                s = State((r,c,), is_terminal)
                self.S.append(s)
                # Initialize the value of every state to 0
                V[s] = 0
                # Begin with a policy that selects every  action with equal probability
                pi[s] = self.num_actions * [0.25]
        return V, pi

    # Gets the action values for a state by getting the expected return
    # of taking each action
    def getActionValuesForState(self, s, V):
        action_values = []
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.getNextStateLikelihood(action, s_prime)
                action_value += p * (s_prime.getReward() + self.gamma * V[s_prime])
            action_values.append(action_value)
        return action_values


class PolicyIterationAgent(DynamicProgrammingAgent):
    def __init__(self, gamma):
        super().__init__(gamma)

    def policyIterate(self, theta=1e-4):
        # Inicializar valores y política
        V, pi = self.initSVAndPi()
        iteration = 1

        while True:
            print(f"\n======= Iteración {iteration} =======")
            # --- Evaluación de política ---
            eval_steps = 0
            while True:
                delta = 0
                for s in self.S:
                    v = V[s]
                    new_v = 0
                    for a, action_prob in enumerate(pi[s]):
                        for s_prime in self.S:
                            p = s.getNextStateLikelihood(a, s_prime)
                            new_v += action_prob * p * (s_prime.getReward() + self.gamma * V[s_prime])
                    V[s] = new_v
                    delta = max(delta, abs(v - new_v))
                eval_steps += 1
                if delta < theta:
                    break  # convergió la evaluación

            print(f"[Evaluación de política] Iteraciones internas: {eval_steps}")
            self._printStateValues(V)

            # --- Mejora de política ---
            policy_stable, pi = self.policyImprove(pi, V)
            self._printPolicy(pi)

            if policy_stable:
                print("✅ Política estable encontrada. Algoritmo finalizado.")
                break

            iteration += 1

    def policyImprove(self, pi, V):
        policy_stable = True
        new_pi = {}

        for s in self.S:
            old_action = np.argmax(pi[s])
            action_values = self.getActionValuesForState(s, V)

            best_value = max(action_values)
            best_actions = [a for a, val in enumerate(action_values) if val == best_value]

            new_pi[s] = [1.0 / len(best_actions) if a in best_actions else 0.0 for a in range(self.num_actions)]

            if old_action != np.argmax(new_pi[s]):
                policy_stable = False

        return policy_stable, new_pi