import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_environment
import os
import pickle
from briscola_build import BriscolaGame, play_and_render_game, evaluate, _GAME_TYPE, plot_learning_curve

pyspiel.register_game(_GAME_TYPE, BriscolaGame)

def save_iql_agents(agents, checkpoint_dir):
    """saves q-tables for the tabular agents."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        with open(os.path.join(checkpoint_dir, f"agent_{i}_qvalues.pkl"), "wb") as f:
            pickle.dump(agent._q_values, f)
    print("Checkpoint saved")

def load_iql_agents(agents, checkpoint_dir):
    """loads q-tables into the tabular agents if they exist."""
    for i, agent in enumerate(agents):
        path = os.path.join(checkpoint_dir, f"agent_{i}_qvalues.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                agent._q_values = pickle.load(f)
            print(f"Loaded Agent {i} Q-table with {len(agent._q_values)} states.")

def train_tabular_q_learning():
    game = BriscolaGame()
    env = rl_environment.Environment(game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0),
        tabular_qlearner.QLearner(player_id=1, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0)
    ]
    
    checkpoint_dir = "./checkpoints_iql"
    load_iql_agents(agents, checkpoint_dir)

    num_episodes = 500000 
    eval_interval = 10000
    
    print(f"Starting Independent Tabular Q-Learning for {num_episodes} episodes.")

    eval_rewards_rand = []
    eval_rewards_p0_rand = []
    eval_rewards_p1_rand = []
    eval_rewards_heur = []
    eval_rewards_p0_heur = []
    eval_rewards_p1_heur = []

    for episode in range(num_episodes):
        time_step = env.reset()

        accumulated_rewards = [0.0, 0.0]
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]

            info_string = env._state.information_state_string(player_id)
            time_step.observations["info_state"][player_id] = info_string

            current_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]

            custom_rewards = list(current_rewards)
            custom_rewards[player_id] = accumulated_rewards[player_id]
            modified_time_step = time_step._replace(rewards=custom_rewards)

            agent_output = agents[player_id].step(modified_time_step)

            accumulated_rewards[player_id] = 0.0

            time_step = env.step([agent_output.action])

            new_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]
            accumulated_rewards[0] += new_rewards[0]
            accumulated_rewards[1] += new_rewards[1]

        for player_id, agent in enumerate(agents):
            info_string = env._state.information_state_string(player_id)
            time_step.observations["info_state"][player_id] = info_string

            current_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]
            custom_rewards = list(current_rewards)
            custom_rewards[player_id] = accumulated_rewards[player_id]
            modified_time_step = time_step._replace(rewards=custom_rewards)
            
            agent.step(modified_time_step)

        # eval
        if (episode + 1) % eval_interval == 0:
            wr_total_rand, wr_p0_rand, wr_p1_rand = evaluate(env, agents, agent_type='tabular', bot_type='random')
            wr_total_heur, wr_p0_heur, wr_p1_heur = evaluate(env, agents, agent_type='tabular', bot_type='heuristic')
            eval_rewards_rand.append(wr_total_rand)
            eval_rewards_p0_rand.append(wr_p0_rand)
            eval_rewards_p1_rand.append(wr_p1_rand)
            eval_rewards_heur.append(wr_total_heur)
            eval_rewards_p0_heur.append(wr_p0_heur)
            eval_rewards_p1_heur.append(wr_p1_heur)
            q_table_size_0 = len(agents[0]._q_values)
            print(f"Episode {episode + 1}/{num_episodes} | P0 States: {q_table_size_0}")
            print(f"\nWR against Random Opponent: {wr_total_rand:.1f}% (P0: {wr_p0_rand:.1f}%, P1: {wr_p1_rand:.1f}%)")
            print(f"\nWR against Heuristic Opponent: {wr_total_heur:.1f}% (P0: {wr_p0_heur:.1f}%, P1: {wr_p1_heur:.1f}%)")
            plot_learning_curve(eval_rewards_p0_rand, eval_rewards_p1_rand, eval_interval, checkpoint_dir, filename="learning_curve_rand.png")
            plot_learning_curve(eval_rewards_p0_heur, eval_rewards_p1_heur, eval_interval, checkpoint_dir, filename="learning_curve_heur.png")
            save_iql_agents(agents, checkpoint_dir)

    print("\nTraining complete")
    print(f"\nLearning Curve Win Rate against Random Opponent: {eval_rewards_rand}")
    print(f"\nLearning Curve Win Rate against Heuristic Opponent: {eval_rewards_heur}")
    
    # visualization
    play_and_render_game(env, agents)

train_tabular_q_learning()