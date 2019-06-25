# %%
from collections import OrderedDict, Counter
from functools import reduce

import numpy as np

from data import process_file, split_on_ips, get_infected, get_normal
from flow_discretization import discrete_flow, count_two


def get_sequence(flows: list):
    # Make a sequence of the netflow data by applying the discretize function
    sequence = list()
    for flow in flows:
        sequence.append(discrete_flow(flow))

    return sequence


def get_transitions(states: list, sequence: list):
    # Transition matrix is states x states
    transitions = np.zeros((len(states), len(states)))

    # Sliding window approach
    start = 0
    while (start + 1) < len(sequence):
        # Count the transition that occurred
        s_state = states.index(sequence[start])
        n_state = states.index(sequence[start + 1])
        transitions[s_state, n_state] += 1

        start += 1

    # Add one for Laplacian smoothing
    transitions = transitions + 1

    return transitions


def get_initial_probs(states: list, sequence: list):
    # Count which states occured in the sequence
    counts = Counter(sequence)
    total = sum(counts.values())
    probs = OrderedDict()

    # Change to probabilities
    for state in states:
        probs[state] = counts[state] / total

    return probs


def count_to_prob(transitions):
    # Change counts in a row to probability vector that sums to one
    row_sums = transitions.sum(axis=1)
    probs = transitions / row_sums[:, np.newaxis]

    return probs


class MarkovChain:

    def __init__(self, states, infected_data: tuple, significant_level=0.05):
        self.states = states
        self.state_len = len(states)
        self.alpha = significant_level  # Standard significant level of 5%

        # Create a sequence of the infected_data
        _, netflows = infected_data
        seq = get_sequence(netflows)

        # Get the transition matrix based on the infected host
        trans = get_transitions(states, seq)
        self.transition = count_to_prob(trans)
        self.initial_probs = get_initial_probs(states, seq)

    def _chain(self, start_state, length):
        # Compute the markov property x(n+1) = x(n) * P
        return np.dot(start_state, reduce(np.dot, [self.transition] * length))

    def predict(self, host_data: tuple):
        _, flows = host_data

        # Make the netflows discrete and compute the starting probability for each state
        seq = get_sequence(flows)
        initial = get_initial_probs(self.states, seq)
        initial_probs = np.array(list(initial.values()))

        # Use the learned Markov chain to predict the probability for each state in this sequence
        start = np.full(self.state_len, fill_value=(1 / self.state_len))  # Start with equal probability for each state
        pred = self._chain(start, len(seq))  # Simulate the markov property for sequence length

        # Ignore probabilities smaller than 5% and see if the prediction matches the real values
        significant_start = initial_probs > self.alpha
        significant_pred = pred > self.alpha
        all_equal = np.array_equal(significant_start, significant_pred)

        # Return the predictions
        return {
            'Infected': all_equal,
            'Normal': not all_equal
        }


if __name__ == '__main__':
    # %% Load the data and split on infected and botnet
    data_path = "../data/capture20110818.pcap.netflow.labeled"
    infected = list(process_file(data_path, lambda l: "Botnet" in l))
    normal = list(process_file(data_path, lambda l: "LEGITIMATE" in l))

    # %% Split the data based on the IP
    infected_hosts = split_on_ips(infected, get_infected(data_path))
    normal_hosts = split_on_ips(normal, get_normal(data_path))

    # Make lists of the hosts as this is easier for indexing
    infected_hosts = list(iter(infected_hosts.items()))
    normal_hosts = list(iter(normal_hosts.items()))

    # %% Determine the states
    infected_comb = count_two(infected)
    normal_comb = count_two(normal)

    # Sort on dict keys alphabetically
    states = sorted(list(set(list(infected_comb.keys()) + list(normal_comb.keys()))))

    # %% Train the markov model on one infected host
    markov = MarkovChain(states, infected_hosts[0])

    # %% Predict all other hosts

    for n in normal_hosts:
        uid, _ = n
        print(f"IP : {uid} -- Infected : {False}")
        print(markov.predict(n))
        print("\n")

    for i in infected_hosts:
        uid, _ = i
        print(f"IP : {uid} -- Infected : {True}")
        print(markov.predict(i))
        print("\n")

    # Note: it predicts the probabilites of each state - thus Protocol and duration
    # Infected hosts match because they use ICMP and small duration
    # Normal hosts dont match because they use TCP and longer duration
