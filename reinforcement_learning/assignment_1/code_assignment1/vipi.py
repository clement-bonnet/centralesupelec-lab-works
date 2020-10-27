import sys
sys.path.insert(0, './utils')
import numpy as np
import matplotlib.pyplot as plt
import math
from cliffwalk import CliffWalk
import timeit

# to export matplotlib plot to latex
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })



def policy_evaluation(P, R, policy, gamma=0.9, tol=1e-2):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        policy: np.array
            matrix mapping states to action (Ns)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        value_function: np.array
            The value function of the given policy
    """
    Ns, _ = R.shape
    # ====================================================
	# YOUR IMPLEMENTATION HERE 
    #
    value_function = np.zeros(Ns)
    stop_crit = False
    while not stop_crit:
        # synchronous value iteration
        old_value_function = value_function.copy()
        for s in range(Ns):
            value_function[s] = R[s,policy[s]] + gamma*P[s,policy[s],:]@old_value_function
        # the following is a vectorized version but it is slower since it requires casting
        # value_function = np.diagonal(R[:,policy]) + gamma*np.diagonal(P[:,policy,:]).transpose()@old_value_function
        stop_crit = (np.max(np.abs(old_value_function - value_function)) < tol)
    # ====================================================
    return value_function

def policy_iteration(P, R, gamma=0.9, tol=1e-3):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        policy: np.array
            the final policy
        V: np.array
            the value function associated to the final policy
    """
    Ns, _ = R.shape
    V = np.zeros(Ns)
    policy = np.zeros(Ns, dtype=np.int)
    # ====================================================
	# YOUR IMPLEMENTATION HERE 
    #
    stop_crit = False
    while not stop_crit:
        old_V = V.copy()
        # policy evaluation
        V = policy_evaluation(P, R, policy, gamma=gamma, tol=tol)
        # policy improvement
        # below is a non-vectorized version
        # for s in range(Ns):
        #     policy[s] = np.argmax(R[s,:] + gamma*P[s,:,:]@V)
        policy = np.argmax(R + gamma*P@V, axis=1)
        # stop if value function has not changed much
        # one can also check whether the policy remained the same
        stop_crit = (np.max(np.abs(old_V - V)) < tol)
    # ====================================================
    return policy, V

def value_iteration(P, R, gamma=0.9, tol=1e-5): # changed default tol from 1e-3 to 1e-5
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        Q: final Q-function (at iteration n)
        greedy_policy: greedy policy wrt Qn
        Qfs: all Q-functions generated by the algorithm (for visualization)
    """
    Ns, Na = R.shape
    Q = np.zeros((Ns, Na))
    Qfs = [Q.copy()]
    greedy_policy = np.zeros(Ns)
    V = np.zeros(Ns)
    # ====================================================
	# YOUR IMPLEMENTATION HERE 
    #
    stop_crit = False
    while not stop_crit:
        # synchronous update
        old_V = V.copy()
        # below is a non-vectorized version
        # for s in range(Ns):
        #     for a in range(Na):
        #         Q[s,a] = R[s,a] + gamma*P[s,a,:] @ old_V
        Q = R + gamma*P@old_V
        V = Q.max(axis=1)
        Qfs.append(Q.copy())
        # check whether V did not change much
        stop_crit = (np.max(np.abs(old_V - V)) < tol)
    greedy_policy = Q.argmax(axis=1)
    # ====================================================
    return Q, greedy_policy, Qfs



# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    tol = 1e-5
    env = CliffWalk(proba_succ=1)
    print(env.R.shape)
    print(env.P.shape)
    env.render()


    # run value iteration to obtain Q-values
    VI_Q, VI_greedypol, all_qfunctions = value_iteration(env.P, env.R, gamma=env.gamma, tol=tol)

    # render the policy
    print("[VI]Greedy policy: ")
    env.render_policy(VI_greedypol)

    # compute the value function of the greedy policy using matrix inversion
    greedy_V = np.zeros(env.Ns)     # changed from 2d-array to 1d-array
    # ====================================================
    # YOUR IMPLEMENTATION HERE 
    # compute value function of the greedy policy
    #
    Ns = env.R.shape[0]
    # reward for action taken from state s using greedy policy
    R = np.array([env.R[s, VI_greedypol[s]] for s in range(Ns)])
    # transition proba for action taken from state s using greedy policy
    P = np.array([[env.P[s, VI_greedypol[s], next_s] for next_s in range(Ns)] for s in range(Ns)])
    # resolution of the linear system
    greedy_V = np.linalg.solve(np.identity(Ns) - env.gamma*P, R)
    # ====================================================

    # show the error between the computed V-functions and the final V-function
    # (that should be the optimal one, if correctly implemented)
    # as a function of time
    norms = [ np.linalg.norm(q.max(axis=1) - greedy_V) for q in all_qfunctions]
    plt.xlabel('Iterations in Value Iteration Algorithm')
    plt.ylabel('Error in V-function')
    plt.plot(norms)

    #### POLICY ITERATION ####
    PI_policy, PI_V = policy_iteration(env.P, env.R, gamma=env.gamma, tol=tol)
    print("\n[PI]final policy: ")
    env.render_policy(PI_policy)

    # control that everything is correct
    assert np.allclose(PI_policy, VI_greedypol),\
        "You should check the code, the greedy policy computed by VI is not equal to the solution of PI"
    assert np.allclose(PI_V, greedy_V),\
        "Since the policies are equal, even the value function should be"
    
    # for visualizing the execution of a policy, you can use the following code
    # state = env.reset()
    # env.render()
    # for i in range(15):
    #     action = VI_greedypol[state]
    #     state, reward, done, _ = env.step(action)
    #     env.render()

    plt.xlim(plt.xlim()[0]+30,plt.xlim()[1]-800)

    # to save matplotlib plot and use it in latex
    # folder_path = "C:/Users/clem6/Google Drive/On laptop + drive/Documents/Cours/ENS-MVA/Reinforcement Learning/assignments/Assignment_1/"
    # plt.savefig(folder_path + "multi_plots.pgf")

    plt.show()


    # the code below compares running time of both algorithms
    # # value iteration
    # print("\nValue iteration")
    # s = "value_iteration(env.P,env.R,gamma=env.gamma,tol=tol)"
    # t = timeit.Timer(stmt=s, setup="from __main__ import value_iteration, env, tol")
    # print(">> running time: {:.0f}ms.".format(1000*min(t.repeat(repeat=5, number=1))),"\n")
    # # policy iteration
    # print("Policy iteration")
    # s = "policy_iteration(env.P,env.R,gamma=env.gamma,tol=tol)"
    # t = timeit.Timer(stmt=s, setup="from __main__ import policy_iteration, env, tol")
    # print(">> running time: {:.0f}ms.".format(1000*min(t.repeat(repeat=5, number=1))),"\n")
