import numpy as np
import pyspiel
import os
import torch
import collections
import matplotlib.pyplot as plt
import pickle as pkl
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import tabular_qlearner, projected_replicator_dynamics
from briscola_build import _GAME_TYPE, BriscolaGame, get_heuristic_action

pyspiel.register_game(_GAME_TYPE, BriscolaGame)

StepOutput = collections.namedtuple("step_output", ["action"])

class RandomBot:
    def step(self, time_step, is_evaluation=True):
        legal_actions = time_step.observations["legal_actions"][time_step.observations["current_player"]]
        return StepOutput(action=np.random.choice(legal_actions))

class HeuristicBot:
    def __init__(self, env):
        self.env = env
    def step(self, time_step, is_evaluation=True):
        legal_actions = time_step.observations["legal_actions"][time_step.observations["current_player"]]
        action = get_heuristic_action(self.env._state, legal_actions)
        return StepOutput(action=action)

class TabularWrapper:
    def __init__(self, agent, env, player_id):
        self.agent = agent
        self.env = env
        self.player_id = player_id
    def step(self, time_step, is_evaluation=True):
        info_string = self.env._state.information_state_string(self.player_id)
        time_step.observations["info_state"][self.player_id] = info_string
        return self.agent.step(time_step, is_evaluation=is_evaluation)
    
class PSROBot:
    def __init__(self, policies, probabilities):
        self.policies = policies
        self.probabilities = probabilities
        self.current_policy = policies[-1]

    def episode_reset(self):
        idx = np.random.choice(len(self.policies), p=self.probabilities)
        self.current_policy = self.policies[idx]

    def step(self, time_step, is_evaluation=True):
        return self.current_policy.step(time_step, is_evaluation=is_evaluation)


def gather_competitors(env):
    """Loads all available agents, keeping pairs of [P0_Agent, P1_Agent]."""
    competitors = {
        "Random": [RandomBot(), RandomBot()],
        "Heuristic": [HeuristicBot(env), HeuristicBot(env)]
    }

    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iql_agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
        tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)
    ]
    for i, agent in enumerate(iql_agents):
        iql_path = os.path.join("./checkpoints_iql", f"agent_{i}_qvalues.pkl")
        if os.path.exists(iql_path):
            with open(iql_path, "rb") as f:
                agent._q_values = pkl.load(f)
            print(f"Loaded Agent {i} Q-table with {len(agent._q_values)} states.")
    competitors["Tabular Q"] = [TabularWrapper(iql_agents[0], env, 0), TabularWrapper(iql_agents[1], env, 1)]

    hidden_layers = [256, 256]
    agents_idqn = [
        dqn.DQN(player_id=0, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers),
        dqn.DQN(player_id=1, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers)
    ]
    for i, agent in enumerate(agents_idqn):
        idqn_path = os.path.join("./checkpoints_idqn", f"agent_{i}.pt")
        checkpoint_idqn = torch.load(idqn_path, map_location=device)
        agent._q_network.load_state_dict(checkpoint_idqn['q_network'])
        agent._target_q_network.load_state_dict(checkpoint_idqn['target_q_network'])
        agent._optimizer.load_state_dict(checkpoint_idqn['optimizer'])
        agent._step_counter = checkpoint_idqn['step_counter']
        print(f"Loaded IDQN Agent {i} state.")
    competitors["IDQN"] = agents_idqn

    agents_selfplay = [
        dqn.DQN(player_id=0, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers),
        dqn.DQN(player_id=1, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers)
    ]
    for i, agent in enumerate(agents_selfplay):
        selfplay_path = os.path.join("./checkpoints_selfplay", f"agent_{i}.pt")
        checkpoint_selfplay = torch.load(selfplay_path, map_location=device)
        agent._q_network.load_state_dict(checkpoint_selfplay['q_network'])
        agent._target_q_network.load_state_dict(checkpoint_selfplay['target_q_network'])
        agent._optimizer.load_state_dict(checkpoint_selfplay['optimizer'])
        agent._step_counter = checkpoint_selfplay['step_counter']
        print(f"Loaded Self-Play Agent {i} state.")
    competitors["selfplay"] = agents_selfplay

    '''agents_psro = [
        dqn.DQN(player_id=0, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers),
        dqn.DQN(player_id=1, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers)
    ]
    meta_path_psro = os.path.join("./checkpoints_psro", "meta.pt") 
    meta_psro = torch.load(meta_path_psro, weights_only=True)
    iteration = meta_psro["iteration"]
    for p in range(2):
        path = f"./checkpoints_psro/player_{p}_policy_{iteration}.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        agents_psro[p]._q_network.load_state_dict(checkpoint['q_network'])
    competitors["PSRO"] = agents_psro'''

    meta = torch.load("./checkpoints_psro/meta.pt", weights_only=True)
    it = meta["iteration"]
    meta_games = np.load("./checkpoints_psro/meta_game.npy")
    probs = projected_replicator_dynamics.projected_replicator_dynamics(
        [np.array(m) for m in meta_games], prd_iterations=10000, prd_gamma=1e-10
    )
    psro_bots = []
    for p in range(2):
        policies = []
        for i in range(it + 1):
            agent = dqn.DQN(player_id=p, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=hidden_layers)
            checkpoint = torch.load(f"./checkpoints_psro/player_{p}_policy_{i}.pt", map_location=device, weights_only=True)
            agent._q_network.load_state_dict(checkpoint['q_network'])
            policies.append(agent)
        
        p_probs = probs[p][:len(policies)]
        psro_bots.append(PSROBot(policies, p_probs))
        
    competitors["PSRO"] = psro_bots
    print(f"Loaded PSRO as Mixed Strategy (Iteration {it}).")

    return competitors

def play_matchup(env, agent0, agent1, num_games=1000):
    """Plays num_games where Agent 0 acts as P0 and Agent 1 acts as P1."""
    wins = 0
    for _ in range(num_games):
        if hasattr(agent0, "episode_reset"): agent0.episode_reset()
        if hasattr(agent1, "episode_reset"): agent1.episode_reset()
        time_step = env.reset()
        while not time_step.last():
            curr = time_step.observations["current_player"]
            if curr == 0:
                action = agent0.step(time_step, is_evaluation=True).action
            else:
                action = agent1.step(time_step, is_evaluation=True).action
            time_step = env.step([action])
            
        if time_step.rewards[0] > 0:
            wins += 1
            
    return (wins / num_games) * 100.0

def run_tournament():
    print("Gathering competitors...")
    env = rl_environment.Environment(BriscolaGame())
    
    competitors = gather_competitors(env)
    names = list(competitors.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    print(f"\nStarting Tournament with {n} competitors: {names}")
    games_per_matchup = 1000 

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 50.0
                continue
                
            nameA = names[i]
            nameB = names[j]
            print(f"Match: {nameA} vs {nameB} ... ", end="")

            wr_A_as_p0 = play_matchup(env, competitors[nameA][0], competitors[nameB][1], num_games=games_per_matchup//2)
            wr_B_as_p0 = play_matchup(env, competitors[nameB][0], competitors[nameA][1], num_games=games_per_matchup//2)

            wr_A_as_p1 = 100.0 - wr_B_as_p0
            
            final_wr = (wr_A_as_p0 + wr_A_as_p1) / 2.0
            matrix[i][j] = final_wr
            print(f"{final_wr:.1f}%")

    print("TOURNAMENT RESULTS (Row Win % vs Column)")
    
    header = f"{'Agent':>15} | " + " | ".join([f"{name[:7]:>7}" for name in names])
    print(header)
    print("-" * len(header))
    
    for i in range(n):
        row = f"{names[i]:>15} | " + " | ".join([f"{matrix[i][j]:>6.1f}%" for j in range(n)])
        print(row)
        
    plot_heatmap(matrix, names)

def plot_heatmap(matrix, names):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(label='Row Win Rate (%)')
    
    plt.xticks(np.arange(len(names)), names, rotation=45, ha='right')
    plt.yticks(np.arange(len(names)), names)
    
    for i in range(len(names)):
        for j in range(len(names)):
            text_color = "black" if 30 < matrix[i][j] < 70 else "white"
            plt.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center", color=text_color, fontweight='bold')
            
    plt.title("Cross-Evaluation Tournament Matrix")
    plt.tight_layout()
    plt.savefig("tournament_matrix.png")
    print("\nSaved heatmap to tournament_matrix.png")

run_tournament()