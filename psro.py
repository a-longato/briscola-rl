import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import os
import torch
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms.psro_v2 import psro_v2, rl_oracle, rl_policy
from briscola_build import _GAME_TYPE, BriscolaGame, evaluate, plot_learning_curve, play_and_render_game

pyspiel.register_game(_GAME_TYPE, BriscolaGame)

def save_psro_checkpoint(solver, iteration, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    meta_games = solver.get_meta_game()
    np.save(os.path.join(save_dir, "meta_game.npy"), meta_games)
    torch.save({"iteration": iteration}, os.path.join(save_dir, "meta.pt"))
    
    policies = solver.get_policies()
    for p, player_policies in enumerate(policies):
        for i, policy in enumerate(player_policies):
            agent = policy._policy 
            checkpoint = {
                'q_network': agent._q_network.state_dict(),
                'target_q_network': agent._target_q_network.state_dict(),
                'optimizer': agent._optimizer.state_dict(),
                'step_counter': getattr(agent, 'step_counter', getattr(agent, '_step_counter', 0)),
            }
            torch.save(checkpoint, os.path.join(save_dir, f"player_{p}_policy_{i}.pt"))
            
    print(f"Checkpoint saved successfully for iteration {iteration}")


def load_psro_checkpoint(solver, env, agent_class, dqn_kwargs, save_dir):
    meta_path = os.path.join(save_dir, "meta.pt")
    if not os.path.exists(meta_path):
        print("No existing PSRO checkpoints found. Starting from scratch.\n")
        return 0
        
    meta = torch.load(meta_path)
    start_iteration = meta["iteration"]
    
    meta_games = np.load(os.path.join(save_dir, "meta_game.npy"))
    solver._meta_games = [np.array(m) for m in meta_games]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = solver.get_policies()
    
    for p in range(2):
        for i in range(start_iteration + 1):
            path = os.path.join(save_dir, f"player_{p}_policy_{i}.pt")
            if not os.path.exists(path):
                continue
                
            if i >= len(policies[p]):
                new_policy = agent_class(env, p, **dqn_kwargs)
                new_policy.freeze()
                policies[p].append(new_policy)

            agent = policies[p][i]._policy 
            checkpoint = torch.load(path, map_location=device)
            
            agent._q_network.load_state_dict(checkpoint['q_network'])
            agent._target_q_network.load_state_dict(checkpoint['target_q_network'])
            agent._optimizer.load_state_dict(checkpoint['optimizer'])
            agent._step_counter = checkpoint['step_counter']
                
    solver.update_meta_strategies()
    print(f"Successfully loaded PSRO checkpoint from iteration {start_iteration}\n")
    return start_iteration


def train_psro():
    print("Initializing PSRO")
    game = BriscolaGame()
    env = rl_environment.Environment(game)

    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent_class = rl_policy.rl_policy_factory(dqn.DQN)

    dqn_kwargs = {
        "state_representation_size": info_state_size,
        "num_actions": num_actions,
        "hidden_layers_sizes": [256, 256],
        "replay_buffer_capacity": 100000,
        "batch_size": 128,
        "learning_rate": 0.001,
        "update_target_network_every": 500,
        "learn_every": 64,
        "discount_factor": 1.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_duration": 100000
    }

    oracle = rl_oracle.RLOracle(
        env,
        agent_class,
        dqn_kwargs,
        number_training_episodes= 250000
    )

    agents = [agent_class(env, player_id, **dqn_kwargs) for player_id in range(2)]
    for agent in agents:
        agent.freeze()

    solver = psro_v2.PSROSolver(
        game=game,
        oracle=oracle,
        initial_policies=agents,
        sims_per_entry=5000,
        meta_strategy_method="prd",
        prd_iterations=10000,
        prd_gamma=1e-10,
        training_strategy_selector='probabilistic',
        sample_from_marginals=True,
        symmetric_game=False
    )
    checkpoint_dir = "./checkpoints_psro"
    start_iteration = load_psro_checkpoint(solver, env, agent_class, dqn_kwargs, save_dir=checkpoint_dir)

    psro_iterations = 5
    
    if start_iteration >= psro_iterations:
        print(f"Model is already fully trained ({start_iteration}/{psro_iterations} iterations). Skipping training.")
    else:
        print(f"\nStarting PSRO for iterations {start_iteration + 1} to {psro_iterations}")

        eval_rewards_rand = []
        eval_rewards_p0_rand = []
        eval_rewards_p1_rand = []
        eval_rewards_heur = []
        eval_rewards_p0_heur = []
        eval_rewards_p1_heur = []

        for iteration in range(start_iteration + 1, psro_iterations + 1):
            print(f"PSRO Iteration {iteration}")

            solver.iteration()

            newest_p0_policy = solver.get_policies()[0][-1]._policy
            newest_p1_policy = solver.get_policies()[1][-1]._policy
            eval_agents = [newest_p0_policy, newest_p1_policy]

            wr_total_rand, wr_p0_rand, wr_p1_rand = evaluate(env, eval_agents, agent_type="deep", bot_type="random")
            wr_total_heur, wr_p0_heur, wr_p1_heur = evaluate(env, eval_agents, agent_type="deep", bot_type="heuristic")
            eval_rewards_rand.append(wr_total_rand)
            eval_rewards_p0_rand.append(wr_p0_rand)
            eval_rewards_p1_rand.append(wr_p1_rand)
            eval_rewards_heur.append(wr_total_heur)
            eval_rewards_p0_heur.append(wr_p0_heur)
            eval_rewards_p1_heur.append(wr_p1_heur)

            print(f"\nWR against Random Opponent: {wr_total_rand:.1f}% (P0: {wr_p0_rand:.1f}%, P1: {wr_p1_rand:.1f}%)")
            print(f"WR against Heuristic Opponent: {wr_total_heur:.1f}% (P0: {wr_p0_heur:.1f}%, P1: {wr_p1_heur:.1f}%)")
            
            meta_game_matrix = solver.get_meta_game()
            meta_strategies = solver.get_meta_strategies()
            
            print("\nCurrent Meta-Game Payoff Matrix (Player 0 Perspective):")
            print(np.round(meta_game_matrix[0], 2))
            
            print("\nNash Equilibrium (Strategy Distribution):")
            print(f"Player 0: {np.round(meta_strategies[0], 3)}")
            print(f"Player 1: {np.round(meta_strategies[1], 3)}")

            plot_learning_curve(eval_rewards_p0_rand, eval_rewards_p1_rand, 1, checkpoint_dir, filename="learning_curve_rand.png")
            plot_learning_curve(eval_rewards_p0_heur, eval_rewards_p1_heur, 1, checkpoint_dir, filename="learning_curve_heur.png")
            save_psro_checkpoint(solver, iteration, save_dir=checkpoint_dir)

        print("\nTraining completed.")
        print(f"Learning Curve Win Rate against Random Opponent: {eval_rewards_rand}")
        print(f"Learning Curve Win Rate against Heuristic Opponent: {eval_rewards_heur}")

    newest_p0_policy = solver.get_policies()[0][-1]._policy
    newest_p1_policy = solver.get_policies()[1][-1]._policy
    eval_agents = [newest_p0_policy, newest_p1_policy]
    play_and_render_game(env, eval_agents)

train_psro()