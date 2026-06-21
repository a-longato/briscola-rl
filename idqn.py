import pyspiel
from open_spiel.python.pytorch import dqn
from open_spiel.python import rl_environment
import os
import torch
from briscola_build import BriscolaGame, evaluate, play_and_render_game, _GAME_TYPE, plot_learning_curve

pyspiel.register_game(_GAME_TYPE, BriscolaGame)

def save_checkpoint(agents, episode, checkpoint_dir):
    """saves all information for both agents (weights, optimizer, step counter)"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for i, agent in enumerate(agents):
        checkpoint = {
            'q_network': agent._q_network.state_dict(),
            'target_q_network': agent._target_q_network.state_dict(),
            'optimizer': agent._optimizer.state_dict(),
            'step_counter': agent.step_counter,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"agent_{i}.pt"))

    meta = {'episode': episode}
    torch.save(meta, os.path.join(checkpoint_dir, "meta.pt"))
    print(f"\nCheckpoint saved at episode {episode}!")

def load_checkpoint(agents, checkpoint_dir):
    """check existance and loads previously trained agents"""
    meta_path = os.path.join(checkpoint_dir, "meta.pt")
    
    if not os.path.exists(meta_path):
        print("No existing checkpoints found. Starting training from scratch.\n")
        return 0

    meta = torch.load(meta_path)
    start_episode = meta['episode']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, agent in enumerate(agents):
        agent_path = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        if os.path.exists(agent_path):
            checkpoint = torch.load(agent_path, map_location=device)
            
            agent._q_network.load_state_dict(checkpoint['q_network'])
            agent._target_q_network.load_state_dict(checkpoint['target_q_network'])
            agent._optimizer.load_state_dict(checkpoint['optimizer'])
            agent._step_counter = checkpoint['step_counter']
            
            print(f"Loaded Agent {i} state (step counter resumed at: {agent._step_counter})")
        else:
            print(f"Warning: Checkpoint file for Agent {i} was not found.")
            
    print(f"Resuming training from episode {start_episode}\n")
    return start_episode

def train_deep_idqn():
    game = BriscolaGame()
    env = rl_environment.Environment(game)

    num_episodes = 250000

    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    hidden_layers = [256, 256]

    agents = [
        dqn.DQN(
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers,
            replay_buffer_capacity=100_000,
            batch_size=128,
            learning_rate=0.001,
            update_target_network_every=500,
            learn_every=64,
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_duration=num_episodes*0.4 
        ),
        dqn.DQN(
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers,
            replay_buffer_capacity=100_000,
            batch_size=128,
            learning_rate=0.001,
            update_target_network_every=500,
            learn_every=64,
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_duration=num_episodes*0.4
        )
    ]

    eval_interval = 10000
    checkpoint_dir = "./checkpoints_idqn"
    start_episode = load_checkpoint(agents, checkpoint_dir)
    if start_episode >= num_episodes:
        print(f"Model is already fully trained ({start_episode}/{num_episodes} episodes). Skipping training.")
    else:
        print(f"Starting Independent Deep Q-Learning from episode {start_episode + 1} to {num_episodes}.")

        eval_rewards_rand = []
        eval_rewards_p0_rand = []
        eval_rewards_p1_rand = []
        eval_rewards_heur = []
        eval_rewards_p0_heur = []
        eval_rewards_p1_heur = []

        for episode in range(start_episode, num_episodes):
            time_step = env.reset()

            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])

            for agent in agents:
                agent.step(time_step)

            # eval
            if (episode + 1) % eval_interval == 0:
                wr_total_rand, wr_p0_rand, wr_p1_rand = evaluate(env, agents, agent_type="deep", bot_type="random")
                wr_total_heur, wr_p0_heur, wr_p1_heur = evaluate(env, agents, agent_type="deep", bot_type="heuristic")
                eval_rewards_rand.append(wr_total_rand)
                eval_rewards_p0_rand.append(wr_p0_rand)
                eval_rewards_p1_rand.append(wr_p1_rand)
                eval_rewards_heur.append(wr_total_heur)
                eval_rewards_p0_heur.append(wr_p0_heur)
                eval_rewards_p1_heur.append(wr_p1_heur)
                loss_0 = agents[0].loss if agents[0].loss is not None else 0.0
                print(f"Episode {episode + 1}/{num_episodes} | NN Loss: {loss_0:.4f}")
                print(f"\nWR against Random Opponent: {wr_total_rand:.1f}% (P0: {wr_p0_rand:.1f}%, P1: {wr_p1_rand:.1f}%)")
                print(f"\nWR against Heuristic Opponent: {wr_total_heur:.1f}% (P0: {wr_p0_heur:.1f}%, P1: {wr_p1_heur:.1f}%)")
                plot_learning_curve(eval_rewards_p0_rand, eval_rewards_p1_rand, eval_interval, checkpoint_dir, filename="learning_curve_rand.png")
                plot_learning_curve(eval_rewards_p0_heur, eval_rewards_p1_heur, eval_interval, checkpoint_dir, filename="learning_curve_heur.png")
                save_checkpoint(agents, episode + 1, checkpoint_dir)

        print("\nTraining complete")
        print(f"\nLearning Curve Win Rate against Random Opponent: {eval_rewards_rand}")
        print(f"\nLearning Curve Win Rate against Heuristic Opponent: {eval_rewards_heur}")

    play_and_render_game(env, agents)

#train_deep_idqn()

def train_deep_self_play():
    print("Initializing Briscola Deep RL Environment")
    game = BriscolaGame()
    env = rl_environment.Environment(game)

    num_episodes = 250000

    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    hidden_layers = [256, 256]

    agents = [
        dqn.DQN(
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers,
            replay_buffer_capacity=100_000,
            batch_size=128,
            learning_rate=0.001,
            update_target_network_every=500,
            learn_every=64,
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_duration=num_episodes*0.4 
        ),
        dqn.DQN(
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers,
            replay_buffer_capacity=100_000,
            batch_size=128,
            learning_rate=0.001,
            update_target_network_every=500,
            learn_every=64,
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_duration=num_episodes*0.4
        )
    ]

    # parameter sharing
    agents[1]._q_network = agents[0]._q_network
    agents[1]._target_q_network = agents[0]._target_q_network
    agents[1]._optimizer = agents[0]._optimizer

    eval_interval = 10000
    checkpoint_dir = "./checkpoints_selfplay" 
    start_episode = load_checkpoint(agents, checkpoint_dir)
    if start_episode >= num_episodes:
        print(f"Model is already fully trained ({start_episode}/{num_episodes} episodes). Skipping training.")
    else:
        print(f"Starting Shared-Weight Deep Q-Learning from episode {start_episode + 1} to {num_episodes}.")

        eval_rewards_rand = []
        eval_rewards_p0_rand = []
        eval_rewards_p1_rand = []
        eval_rewards_heur = []
        eval_rewards_p0_heur = []
        eval_rewards_p1_heur = []

        for episode in range(start_episode, num_episodes):
            time_step = env.reset()

            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])

            for agent in agents:
                agent.step(time_step)

            # eval
            if (episode + 1) % eval_interval == 0:
                wr_total_rand, wr_p0_rand, wr_p1_rand = evaluate(env, agents, agent_type="deep", bot_type="random")
                wr_total_heur, wr_p0_heur, wr_p1_heur = evaluate(env, agents, agent_type="deep", bot_type="heuristic")
                eval_rewards_rand.append(wr_total_rand)
                eval_rewards_p0_rand.append(wr_p0_rand)
                eval_rewards_p1_rand.append(wr_p1_rand)
                eval_rewards_heur.append(wr_total_heur)
                eval_rewards_p0_heur.append(wr_p0_heur)
                eval_rewards_p1_heur.append(wr_p1_heur)
                loss_0 = agents[0].loss if agents[0].loss is not None else 0.0
                print(f"Episode {episode + 1}/{num_episodes} | NN Loss: {loss_0:.4f}")
                print(f"\nWR against Random Opponent: {wr_total_rand:.1f}% (P0: {wr_p0_rand:.1f}%, P1: {wr_p1_rand:.1f}%)")
                print(f"\nWR against Heuristic Opponent: {wr_total_heur:.1f}% (P0: {wr_p0_heur:.1f}%, P1: {wr_p1_heur:.1f}%)")
                plot_learning_curve(eval_rewards_p0_rand, eval_rewards_p1_rand, eval_interval, checkpoint_dir, filename="learning_curve_rand.png")
                plot_learning_curve(eval_rewards_p0_heur, eval_rewards_p1_heur, eval_interval, checkpoint_dir, filename="learning_curve_heur.png")
                save_checkpoint(agents, episode + 1, checkpoint_dir)

        print("\nTraining complete")
        print(f"\nLearning Curve Win Rate against Random Opponent: {eval_rewards_rand}")
        print(f"\nLearning Curve Win Rate against Heuristic Opponent: {eval_rewards_heur}")

    play_and_render_game(env, agents)

train_deep_self_play()