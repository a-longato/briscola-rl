import numpy as np
import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_environment
import os
import pickle

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
        """encodes game state into a 125 element numpy array"""
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
    """evaluate against opponent who play at random, returns the win rate"""
    wins = 0
    
    for _ in range(num_games):
        time_step = env.reset()
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            legal_actions = time_step.observations["legal_actions"][current_player]
            
            if current_player == eval_player_id:
                # Force clean string representation for tabular hashing
                info_string = env._state.information_state_string(current_player)
                time_step.observations["info_state"][current_player] = info_string
                
                agent_output = trained_agent.step(time_step, is_evaluation=True)
                action = agent_output.action
            else:
                action = np.random.choice(legal_actions)
                
            time_step = env.step([action])
            
        if time_step.rewards[eval_player_id] > 0:
            wins += 1
            
    return (wins / num_games) * 100.0

'''
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
'''

def play_and_render_game(env, agents):
    """show an example game"""
    print("===BRISCOLA GAME===")
    
    time_step = env.reset()

    raw_state = env._state
    
    print(f"Briscola: {SUITS[raw_state.briscola_card // 10]} (Card: {get_card_name(raw_state.briscola_card)})")
    
    turn = 1
    while not time_step.last():
        player = time_step.observations["current_player"]
        legal_actions = time_step.observations["legal_actions"][player]

        agent_output = agents[player].step(time_step, is_evaluation=True)
        action = agent_output.action
        
        print(f"\n--- Turn {turn} ---")
        print(f"Table Card: {get_card_name(raw_state.table_card)}")
        print(f"Player {player}'s Hand: {[get_card_name(c) for c in raw_state.hands[player]]}")
        print(f"Player {player} plays: {get_card_name(action)}")
        
        time_step = env.step([action])
        turn += 1

    print("GAME OVER")
    print(f"Player 0 Final Reward: {time_step.rewards[0]}")
    print(f"Player 1 Final Reward: {time_step.rewards[1]}")

def save_iql_agents(agents, checkpoint_dir="./checkpoints_iql"):
    """saves q-tables for the tabular agents."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        with open(os.path.join(checkpoint_dir, f"agent_{i}_qvalues.pkl"), "wb") as f:
            pickle.dump(agent._q_values, f)
    print("Checkpoint saved")

def load_iql_agents(agents, checkpoint_dir="./checkpoints_iql"):
    """loads q-tables into the tabular agents if they exist."""
    for i, agent in enumerate(agents):
        path = os.path.join(checkpoint_dir, f"agent_{i}_qvalues.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                agent._q_values = pickle.load(f)
            print(f"Loaded Agent {i} Q-table with {len(agent._q_values)} states.")

'''
def train_tabular_q_learning():
    game = BriscolaGame()
    env = rl_environment.Environment(game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0),
        tabular_qlearner.QLearner(player_id=1, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0)
    ]
    load_iql_agents(agents)

    num_episodes = 250000 
    eval_interval = 10000
    
    print(f"Starting Independent Tabular Q-Learning for {num_episodes} episodes.")

    eval_rewards = []

    for episode in range(num_episodes):
        time_step = env.reset()
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        
        for agent in agents:
            agent.step(time_step)

        # eval
        if (episode + 1) % eval_interval == 0:
            avg_reward = evaluate_against_random(env, agents[0], eval_player_id=0, num_games=1000)
            eval_rewards.append(avg_reward)
            
            q_table_size_0 = len(agents[0]._q_values)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"P0 States Discovered: {q_table_size_0} | "
                  f"Avg Win Rate vs Random: {avg_reward:.2f}%")
            save_iql_agents(agents)

    print("\nTraining complete")
    print(f"Learning Curve Rewards: {eval_rewards}")
    
    # visualization
    play_and_render_game(env, agents)
'''

def train_tabular_q_learning():
    game = BriscolaGame()
    env = rl_environment.Environment(game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0),
        tabular_qlearner.QLearner(player_id=1, num_actions=game.num_distinct_actions(), step_size=0.1, discount_factor=1.0)
    ]

    load_iql_agents(agents)

    num_episodes = 250000 
    eval_interval = 10000
    
    print(f"Starting Independent Tabular Q-Learning for {num_episodes} episodes.")

    eval_rewards = []

    for episode in range(num_episodes):
        time_step = env.reset()
        
        # Bank to accumulate rewards between turns for multi-agent support
        accumulated_rewards = [0.0, 0.0]
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            
            # 1. Force clean string representation for Tabular State Hashing
            info_string = env._state.information_state_string(player_id)
            time_step.observations["info_state"][player_id] = info_string

            current_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]
            
            # 2. Inject the banked rewards so the agent sees rewards from the opponent's turn
            custom_rewards = list(current_rewards)
            custom_rewards[player_id] = accumulated_rewards[player_id]
            modified_time_step = time_step._replace(rewards=custom_rewards)
            
            # Step the agent
            agent_output = agents[player_id].step(modified_time_step)
            
            # Reset the bank for the player that just acted
            accumulated_rewards[player_id] = 0.0
            
            # Step the environment
            time_step = env.step([agent_output.action])
            
            # Bank the newly generated rewards for both players
            new_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]
            accumulated_rewards[0] += new_rewards[0]
            accumulated_rewards[1] += new_rewards[1]

        # --- Terminal state update ---
        for player_id, agent in enumerate(agents):
            info_string = env._state.information_state_string(player_id)
            time_step.observations["info_state"][player_id] = info_string
            
            # Handle None rewards at terminal state just in case
            current_rewards = time_step.rewards if time_step.rewards is not None else [0.0, 0.0]
            
            # At terminal state, feed the agent whatever is left in their bank
            custom_rewards = list(current_rewards)
            custom_rewards[player_id] = accumulated_rewards[player_id]
            modified_time_step = time_step._replace(rewards=custom_rewards)
            
            agent.step(modified_time_step)

        # eval
        if (episode + 1) % eval_interval == 0:
            avg_reward = evaluate_against_random(env, agents[0], eval_player_id=0, num_games=1000)
            eval_rewards.append(avg_reward)
            
            q_table_size_0 = len(agents[0]._q_values)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"P0 States Discovered: {q_table_size_0} | "
                  f"Avg Win Rate vs Random: {avg_reward:.2f}%")
            
            save_iql_agents(agents)

    print("\nTraining complete")
    print(f"Learning Curve Rewards: {eval_rewards}")
    
    # visualization
    play_and_render_game(env, agents)

train_tabular_q_learning()