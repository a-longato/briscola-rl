import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import os
import torch
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms.psro_v2 import psro_v2, rl_oracle, rl_policy

# game info
POINTS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 2, 6: 3, 7: 4, 8: 10, 9: 11}
SUITS = ["Coins", "Cups", "Swords", "Clubs"]
RANKS = ["2", "4", "5", "6", "7", "Jack", "Knight", "King", "Three", "Ace"]

def get_card_name(card_id):
    if card_id == 40 or card_id == -1:
        return "None"
    return f"{RANKS[card_id % 10]} of {SUITS[card_id // 10]}"

NUM_CARDS = 40
EMPTY_CARD = 40
BRISCOLA_TARGET = 99

class BriscolaState(pyspiel.State):
    """the state of the briscola game"""
    def __init__(self, game):
        super().__init__(game)
        self.cards_played = []
        self.deck = list(range(NUM_CARDS))
        self.hands = [[], []]

        self.briscola_card = -1
        self.briscola_drawn = False
        self.table_card = -1
        self.lead_player = 0
        self._current_player = pyspiel.PlayerId.CHANCE

        self.points = [0, 0]
        self._instant_rewards = [0.0, 0.0]

        # initial dealing
        self.deal_targets = [0, 0, 0, 1, 1, 1, BRISCOLA_TARGET]

    def current_player(self):
        """Returns CHANCE if dealing, otherwise the player whose turn it is."""
        if len(self.deal_targets) > 0:
            return pyspiel.PlayerId.CHANCE
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL

        # rules for who goes
        if self.table_card == -1:
            return self.lead_player
        else:
            return 1 - self.lead_player

    def _legal_actions(self, player):
        """returns valid actions (card in the player's hand)"""
        if player != self.current_player():
            return []
        return sorted(self.hands[player])

    def chance_outcomes(self):
            """returns a list of (action, probability) for the chance node"""
            if len(self.deck) > 0:
                # draw
                prob = 1.0 / len(self.deck)
                return [(card, prob) for card in self.deck]
            else:
                # give briscola at the end
                return [(self.briscola_card, 1.0)]

    def _apply_action(self, action):
        """applies an action: deal or play"""
        self._instant_rewards = [0.0, 0.0]
        # dealing
        if self.is_chance_node():
            if action in self.deck:
                self.deck.remove(action)
            else:
                # briscola draw
                self.briscola_drawn = True

            target = self.deal_targets.pop(0)

            if target == BRISCOLA_TARGET:
                self.briscola_card = action
            else:
                self.hands[target].append(action)
            return

        # playing
        player = self.current_player()
        self.hands[player].remove(action)
        self.cards_played.append(action)

        if self.table_card == -1:
            # first card
            self.table_card = action
        else:
            # second card + resolution
            lead_card = self.table_card
            second_card = action
            lead_suit = lead_card // 10
            second_suit = second_card // 10
            briscola_suit = self.briscola_card // 10

            lead_wins = True
            if second_suit == briscola_suit and lead_suit != briscola_suit:
                lead_wins = False
            elif second_suit == lead_suit and (second_card % 10) > (lead_card % 10):
                lead_wins = False

            winner = self.lead_player if lead_wins else 1 - self.lead_player
            loser = 1 - winner

            # add points
            points = POINTS[lead_card % 10] + POINTS[second_card % 10]
            self.points[winner] += points

            # instant rewards
            instant_reward_scaling = 0.2
            self._instant_rewards[winner] = float(points)*instant_reward_scaling
            self._instant_rewards[loser] = float(-points)*instant_reward_scaling

            self.table_card = -1
            self.lead_player = winner

            # dealing condition
            if len(self.deck) > 0 or not self.briscola_drawn:
                self.deal_targets.extend([winner, loser])
            
            # win bonus
            if self.is_terminal():
                win_bonus = 50.0
                
                if self.points[0] > self.points[1]:
                    self._instant_rewards[0] += win_bonus
                    self._instant_rewards[1] -= win_bonus
                elif self.points[1] > self.points[0]:
                    self._instant_rewards[1] += win_bonus
                    self._instant_rewards[0] -= win_bonus

    def is_terminal(self):
        # terminal condition
        return len(self.hands[0]) == 0 and len(self.hands[1]) == 0 and len(self.deal_targets) == 0

    def returns(self):
        """returns final rewards: zero-sum"""
        if not self.is_terminal():
            return [0.0, 0.0]

        return [self.points[0] - 60.0, self.points[1] - 60.0]

    def information_state_string(self, player):
        """string for q-learning"""
        if self.is_chance_node():
            return "ChanceNode"

        hand_str = ",".join(map(str, sorted(self.hands[player])))
        b_suit = self.briscola_card // 10 if self.briscola_card != -1 else -1
        played_str = ",".join(map(str, self.cards_played))

        return f"P{player} | Hand:{hand_str} | B_Suit:{b_suit} | Table:{self.table_card} | Played:{played_str}"

    def information_state_tensor(self, player):
        """encodes game state into a 129 element numpy array"""
        tensor = []

        # hand
        hand_multihot = np.zeros(NUM_CARDS, dtype=np.float32)
        for card in self.hands[player]:
            if card != EMPTY_CARD:
                hand_multihot[card] = 1.0
        tensor.extend(hand_multihot)

        # suit
        briscola_suit_onehot = np.zeros(4, dtype=np.float32)
        if self.briscola_card != -1:
            briscola_suit_onehot[self.briscola_card // 10] = 1.0
        tensor.extend(briscola_suit_onehot)

        # table
        table_onehot = np.zeros(41, dtype=np.float32)
        if self.table_card == -1:
            table_onehot[40] = 1.0
        else:
            table_onehot[self.table_card] = 1.0
        tensor.extend(table_onehot)

        # played cards
        played_multihot = np.zeros(NUM_CARDS, dtype=np.float32)
        for card in self.cards_played:
            played_multihot[card] = 1.0
        tensor.extend(played_multihot)

        # points
        tensor.append(self.points[player] / 120.0)
        tensor.append(self.points[1 - player] / 120.0)

        # how many cards in deck
        tensor.append(len(self.deck) / 40.0)

        # who starts
        is_leading = 1.0 if (self.table_card == -1) else 0.0
        tensor.append(is_leading)

        return np.array(tensor, dtype=np.float32)

    def rewards(self):
        """returns hand rewards"""
        return self._instant_rewards



_GAME_TYPE = pyspiel.GameType(
    short_name="briscola",
    long_name="The Mighty Reinforcement Learning Briscola",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    parameter_specification={}
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_CARDS,
    max_chance_outcomes=NUM_CARDS,
    num_players=2,
    min_utility=-60.0,
    max_utility=60.0,
    utility_sum=0.0,
    max_game_length=40
)

class BriscolaObserver:
    """rl environment with player's perspective"""
    def __init__(self, iig_obs_type, params):
        self.tensor = np.zeros(129, dtype=np.float32)
        self.dict = {"info_state": self.tensor}

    def set_from(self, state, player):
        np.copyto(self.tensor, state.information_state_tensor(player))

    def string_from(self, state, player):
        return state.information_state_string(player)

class BriscolaGame(pyspiel.Game):
    """wrapper"""
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return BriscolaState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return BriscolaObserver(iig_obs_type, params)

pyspiel.register_game(_GAME_TYPE, BriscolaGame)

def evaluate_against_random(env, trained_agent, eval_player_id, num_games=1000):
    """evaluation, returns the win rate"""
    wins = 0
    
    for _ in range(num_games):
        time_step = env.reset()
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            legal_actions = time_step.observations["legal_actions"][current_player]

            if current_player == eval_player_id:
                agent_output = trained_agent.step(time_step, is_evaluation=True)
                action = agent_output.action
            else:
                action = np.random.choice(legal_actions)

            time_step = env.step([action])

        if time_step.rewards[eval_player_id] > 0:
            wins += 1
            
    return (wins / num_games) * 100.0


def play_and_render_game(env, agents):
    """show an example game"""
    print("===BRISCOLA GAME===")

    time_step = env.reset()
    raw_state = env._state
    print(f"Briscola: {SUITS[raw_state.briscola_card // 10]} (Card: {get_card_name(raw_state.briscola_card)})")

    turn = 1
    while not time_step.last():
        player = time_step.observations["current_player"]
        agent_output = agents[player].step(time_step, is_evaluation=True)
        action = agent_output.action

        print(f"\nTurn {turn}")
        print(f"Table Card: {get_card_name(raw_state.table_card)}")
        print(f"Player {player}'s Hand: {[get_card_name(c) for c in raw_state.hands[player]]}")
        print(f"Player {player} plays: {get_card_name(action)}")

        time_step = env.step([action])
        turn += 1

    print("GAME OVER")
    print(f"Player 0 Final Reward: {time_step.rewards[0]}")
    print(f"Player 1 Final Reward: {time_step.rewards[1]}")

def save_psro_checkpoint(solver, iteration, save_dir="./psro_checkpoints"):
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


def load_psro_checkpoint(solver, env, agent_class, dqn_kwargs, save_dir="./psro_checkpoints"):
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
    start_iteration = load_psro_checkpoint(solver, env, agent_class, dqn_kwargs)

    psro_iterations = 5
    print(f"\nStarting PSRO for iterations {start_iteration + 1} to {psro_iterations}")

    for iteration in range(start_iteration + 1, psro_iterations + 1):
        print(f"\n{'='*40}")
        print(f" PSRO Iteration {iteration}")
        print(f"{'='*40}")
        
        solver.iteration()

        newest_p0_policy = solver.get_policies()[0][-1]._policy
        win_rate = evaluate_against_random(env, newest_p0_policy, eval_player_id=0, num_games=1000)
        print(f"\n[Metric] Newest Player 0 Policy Win Rate vs Random: {win_rate}%")
        
        meta_game_matrix = solver.get_meta_game()
        meta_strategies = solver.get_meta_strategies()
        
        print("\nCurrent Meta-Game Payoff Matrix (Player 0 Perspective):")
        print(np.round(meta_game_matrix[0], 2))
        
        print("\nNash Equilibrium (Strategy Distribution):")
        print(f"Player 0: {np.round(meta_strategies[0], 3)}")
        print(f"Player 1: {np.round(meta_strategies[1], 3)}")

        save_psro_checkpoint(solver, iteration)

    print("Training completed.")

train_psro()