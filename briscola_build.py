import numpy as np
import pyspiel
import matplotlib.pyplot as plt
import os

# game info
POINTS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 2, 6: 3, 7: 4, 8: 10, 9: 11}
SEMI = ["Denari", "Coppe", "Spade", "Bastoni"]
CARTE = ["2", "4", "5", "6", "7", "Fante", "Cavallo", "Re", "3", "Asso"]

def get_card_name(card_id):
    if card_id == 40 or card_id == -1:
        return "None"
    return f"{CARTE[card_id % 10]} di {SEMI[card_id // 10]}"

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
            lead_seme = lead_card // 10
            second_seme = second_card // 10
            briscola = self.briscola_card // 10

            lead_wins = True
            if second_seme == briscola and lead_seme != briscola:
                lead_wins = False
            elif second_seme == lead_seme and (second_card % 10) > (lead_card % 10):
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

        briscola = self.briscola_card // 10 if self.briscola_card != -1 else -1

        t_card = self.table_card

        return f"P{player} | Hand:{hand_str} | Briscola:{briscola} | Table:{t_card}"

    def information_state_tensor(self, player):
        """encodes game state into a 129 element numpy array"""
        tensor = []

        # hand
        hand_multihot = np.zeros(NUM_CARDS, dtype=np.float32)
        for card in self.hands[player]:
            if card != EMPTY_CARD:
                hand_multihot[card] = 1.0
        tensor.extend(hand_multihot)

        # semi
        briscola_onehot = np.zeros(4, dtype=np.float32)
        if self.briscola_card != -1:
            briscola_onehot[self.briscola_card // 10] = 1.0
        tensor.extend(briscola_onehot)

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
    
def get_heuristic_action(state, legal_actions):
    """rule based briscola bot"""
    briscola = state.briscola_card // 10
    table_card = state.table_card
    
    def get_seme(c): return c // 10
    def get_card(c): return c % 10
    def get_points(c): return POINTS[c % 10]
    
    # sort hand
    hand = sorted(legal_actions, key=lambda c: (get_points(c), get_card(c)))

    # leading state
    if table_card == -1:
        # play lowest non briscola first, in case there are only briscole in hand, play lowest
        non_briscola = [c for c in hand if get_seme(c) != briscola]
        if non_briscola:
            return non_briscola[0]
        return hand[0]

    # second to play state
    lead_seme = get_seme(table_card)
    lead_card = get_card(table_card)
    lead_points = get_points(table_card)

    # winning using the same seme
    winning_same_seme = [c for c in hand if get_seme(c) == lead_seme and get_card(c) > lead_card]
    # taking with a briscola
    briscola_cards = [c for c in hand if get_seme(c) == briscola]

    if lead_seme != briscola:
        # if you can take with the same seed, do it with the lowest available card
        if winning_same_seme:
            return winning_same_seme[0]
            
        # if oppo plays a carico, take with the lowest briscola
        if lead_points >= 10 and briscola_cards:
            return briscola_cards[0]

    elif lead_seme == briscola:
        if winning_same_seme:
            # surpass briscola only to eat the 3 with asso
            if lead_points >= 10:
                return winning_same_seme[0]

    # if it's impossible to win or not worth it, play liscio
    non_briscola = [c for c in hand if get_seme(c) != briscola]
    if non_briscola:
        return non_briscola[0]
    return hand[0]

def evaluate(env, agents, agent_type, bot_type, num_games=2000):
    """
    evaluation, tests both player 0 and 1, returns total, p0, and p1 win rates
    agent_type: "deep" or "tabular"
    bot_type: "random" or "heuristic"
    """
    wins = {0: 0, 1: 0}
    games_per_player = num_games // 2
    
    for eval_player_id in [0, 1]:
        trained_agent = agents[eval_player_id]
        
        for _ in range(games_per_player):
            time_step = env.reset()
            while not time_step.last():
                current_player = time_step.observations["current_player"]
                legal_actions = time_step.observations["legal_actions"][current_player]

                if current_player == eval_player_id:
                    if agent_type == "tabular":
                        info_string = env._state.information_state_string(current_player)
                        time_step.observations["info_state"][current_player] = info_string
                    agent_output = trained_agent.step(time_step, is_evaluation=True)
                    action = agent_output.action            
                else:
                    if bot_type == "random":
                        action = np.random.choice(legal_actions)
                    elif bot_type == "heuristic":
                        action = get_heuristic_action(env._state, legal_actions)

                time_step = env.step([action])

            if time_step.rewards[eval_player_id] > 0:
                wins[eval_player_id] += 1
                
    wr_p0 = (wins[0] / games_per_player) * 100.0
    wr_p1 = (wins[1] / games_per_player) * 100.0
    wr_total = ((wins[0] + wins[1]) / num_games) * 100.0
    
    return wr_total, wr_p0, wr_p1

def play_and_render_game(env, agents):
    """shows an example game"""
    print("BRISCOLA GAME")

    time_step = env.reset()
    raw_state = env._state
    print(f"Briscola: {SEMI[raw_state.briscola_card // 10]} (Card: {get_card_name(raw_state.briscola_card)})")

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
    returns = env._state.returns()
    print(f"Player 0 point margin: {returns[0]}")
    print(f"Player 1 point margin: {returns[1]}")

def plot_learning_curve(p0_rewards, p1_rewards, eval_interval, save_dir, filename="learning_curve.png"):
    """generates learning curve plot"""
    os.makedirs(save_dir, exist_ok=True)
    episodes = [i * eval_interval for i in range(1, len(p0_rewards) + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, p0_rewards, marker='o', linestyle='-', color='blue', label='Player 0 Win Rate')
    plt.plot(episodes, p1_rewards, marker='s', linestyle='-', color='green', label='Player 1 Win Rate')
    plt.axhline(y=50, color='red', linestyle='--', label='50% Baseline')
    
    plt.title('Agents Win Rate vs Baseline over Time')
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
