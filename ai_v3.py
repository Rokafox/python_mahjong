import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from 牌 import 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算

# ---------------------------
# 環境
# ---------------------------
class MahjongEnvironment:
    """
    強化学習用の麻雀環境 (4人プレイ)
    """

    # N_TILE_TYPES = 34 * 2           # 捨て + ぽん
    N_TILE_TYPES = 34
    STATE_BITS = 5                  # 0,1,2,3,4 枚の one‑hot
    N_PLAYERS = 4                   # プレイヤー数

    def __init__(self):
        self.reset()


    def reset(self):
        # Initialize the wall (山)
        self.山 = 山を作成する()
        
        # Initialize all players' hands
        self.全手牌 = []  # List of hands for all players
        for i in range(self.N_PLAYERS):
            self.全手牌.append(self.山[:13])
            self.山 = self.山[13:]
        
        # The first player (index 0) is our agent
        self.手牌 = self.全手牌[0]
        
        # Initialize discard piles for all players
        self.捨て牌 = [[] for _ in range(self.N_PLAYERS)]
        self.副露 = [[] for _ in range(self.N_PLAYERS)]
        
        # Current player index (0 is the agent)
        self.current_player = 0
        
        self.score = 0
        self.turn = 0
        self.penalty_456 = 0
        self.tennpai = []
        self.mz_score = 0
        self.complete = []
        self.hand_when_complete = []
        
        return self._get_state()


    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS + 
                         self.N_TILE_TYPES * self.N_PLAYERS, dtype=np.float32)
        
        counter = Counter(self._tile_to_index(t) for t in self.手牌)
        for idx in range(self.N_TILE_TYPES):
            cnt = counter.get(idx, 0)
            state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
            
        discard_offset = self.N_TILE_TYPES * self.STATE_BITS
        for player_idx in range(self.N_PLAYERS):
            for tile in self.捨て牌[player_idx]:
                tile_idx = self._tile_to_index(tile)
                state[discard_offset + player_idx * self.N_TILE_TYPES + tile_idx] += 1.0
        
        return state


    def _tile_to_index(self, tile):
        """Convert tile to unique index"""
        if tile.何者 == "萬子":
            return tile.その上の数字 - 1
        elif tile.何者 == "筒子":
            return 9 + tile.その上の数字 - 1
        elif tile.何者 == "索子":
            return 18 + tile.その上の数字 - 1
        elif tile.何者 == "東風":
            return 27
        elif tile.何者 == "南風":
            return 28
        elif tile.何者 == "西風":
            return 29
        elif tile.何者 == "北風":
            return 30
        elif tile.何者 == "白ちゃん":
            return 31
        elif tile.何者 == "發ちゃん":
            return 32
        elif tile.何者 == "中ちゃん":
            return 33
    
    def _index_to_tile_type(self, idx):
        """Convert index to tile type (for action selection)"""
        if 0 <= idx < 9:
            return ("萬子", idx + 1)
        elif 9 <= idx < 18:
            return ("筒子", idx - 9 + 1)
        elif 18 <= idx < 27:
            return ("索子", idx - 18 + 1)
        elif idx == 27:
            return ("東風", 0)
        elif idx == 28:
            return ("南風", 0)
        elif idx == 29:
            return ("西風", 0)
        elif idx == 30:
            return ("北風", 0)
        elif idx == 31:
            return ("白ちゃん", 0)
        elif idx == 32:
            return ("發ちゃん", 0)
        elif idx == 33:
            return ("中ちゃん", 0)


    def get_valid_actions(self) -> list[int]:
        valid = set()
        # （1）捨て可能な牌
        valid.update({self._tile_to_index(t) for t in self.手牌})

        # （2）ぽん可能な牌
        last_disc = self.捨て牌[(self.current_player - 1) % self.N_PLAYERS]
        if last_disc:
            tile = last_disc[-1]
            count = sum(1 for t in self.全手牌[0] if (t.何者, t.その上の数字) == (tile.何者, tile.その上の数字))
            if count >= 2:
                idx = self._tile_to_index(tile)
                valid.add(idx + self.N_TILE_TYPES)  # ぽんアクション
        return sorted(valid)


    def step(self, action: int):
        reward = 0
        
        # Validate action (only for the agent)
        if self.current_player == 0:  # Agent's turn
            assert any((t_idx := self._tile_to_index(t)) == action for t in self.手牌), \
                "手牌に無い牌が指定されました"
        
        # Check if game is over
        if not self.山:
            return self._get_state(), -10, True, {}

        # Process all players' turns
        for _ in range(self.N_PLAYERS):
            player_idx = self.current_player
            player_hand = self.全手牌[player_idx]

            print(f"Player {player_idx}:")
            print(f"手牌: {len(player_hand)} 副露: {len([tile for meld in self.副露[player_idx] for tile in meld['tiles']])}")
            
            len_of_hand_and_exposed = len(player_hand) + len([tile for meld in self.副露[player_idx] for tile in meld['tiles']])

            # ①  draw&ツモ
            if len_of_hand_and_exposed > 14:
                raise ValueError("手牌が14枚を超えました。")
            elif len_of_hand_and_exposed == 14:
                pass
            elif len_of_hand_and_exposed == 13:
                drawn_tile = self.山.pop(0)
                player_hand.append(drawn_tile)
            else:
                raise ValueError("手牌が13枚未満です。")
            
            # Check for win (only for agent)
            if player_idx == 0:
                # Create a copy of the hand for scoring
                win_check_hand = player_hand.copy()
                # Add tiles from exposed sets if any
                if self.副露[player_idx]:
                    win_check_hand.extend([tile for meld in self.副露[player_idx] for tile in meld['tiles']])
                
                a, b, c = 点数計算(win_check_hand)  # return 点数, 役リスト, 和了形?
                if c:
                    print(f"ツモ！{b}")
                    self.complete = b
                    reward += int(a * 2)
                    done = True
                    remaining_turns = 126 - self.turn
                    assert remaining_turns >= 0
                    reward += int((75 * remaining_turns) * 0.6)
                    self.mz_score = 16
                    self.score += reward 
                    self.hand_when_complete = [str(t) for t in win_check_hand]
                    return self._get_state(), reward, done, {
                        "score": self.score, 
                        "turn": self.turn,
                        "penalty_456": self.penalty_456,
                        "complete": self.complete,
                        "mz_score": self.mz_score,
                        "hand_when_complete": self.hand_when_complete
                    }
            
            # ② 打牌 (discard)
            if player_idx == 0:  # Agent's turn - use the provided action
                print(f"Player 0 hand: {[str(t) for t in self.手牌]}")
                print(f"Valid actions: {self.get_valid_actions()}")
                print(f"Chosen action: {action}")
                reward -= 2 # アクションの基本罰則
                
                # Find and discard the tile
                tile_type = self._index_to_tile_type(action)
                for i, t in enumerate(player_hand):
                    if (t.何者, t.その上の数字) == tile_type:
                        discarded_tile = player_hand.pop(i)
                        break
                else:
                    raise ValueError(f"Invalid action: {action} (tile not in hand)")
            else:
                # Non-agent players: discard random tile
                discard_idx = random.randrange(len(player_hand))
                discarded_tile = player_hand.pop(discard_idx)
            
            # Add to discard pile
            self.捨て牌[player_idx].append(discarded_tile)
            
            # ③ ロン (ron) check for all other players
            for ron_player_idx in range(self.N_PLAYERS):
                if ron_player_idx != player_idx:  # Check all players except the current one
                    # Create a proper hand for ron check
                    ron_check_hand = self.全手牌[ron_player_idx].copy()
                    
                    # Add the discarded tile
                    ron_check_hand.append(discarded_tile)
                    
                    # For checking winning condition, we need to prepare a complete hand
                    # that includes tiles from exposed sets
                    winning_hand = ron_check_hand.copy()
                    
                    # Add all tiles from exposed sets (for winning calculation)
                    if self.副露[ron_player_idx]:
                        print(f"ロンチェック: {ron_player_idx}の手牌 (副露あり)")
                        # We add exposed tiles only for winning calculation
                        exposed_tiles = [tile for meld in self.副露[ron_player_idx] for tile in meld['tiles']]
                        winning_hand.extend(exposed_tiles)
                    else:
                        print(f"ロンチェック: {ron_player_idx}の手牌")
                    
                    print(f"手牌の枚数: {len(winning_hand)}")
                    
                    # Check if this creates a winning hand
                    点数, 役リスト, winning = 点数計算(winning_hand)
                    
                    if winning:
                        # Handle ron for the agent
                        if ron_player_idx == 0:
                            print(f"ロン！{役リスト}")
                            self.complete = 役リスト
                            reward += int(点数 * 1.5)  # Typical ron is worth less than tsumo in this implementation
                            done = True
                            remaining_turns = 126 - self.turn
                            assert remaining_turns >= 0
                            reward += int((75 * remaining_turns) * 0.5)  # Slightly less bonus for ron
                            self.mz_score = 16
                            self.score += reward
                            self.hand_when_complete = [str(t) for t in winning_hand]
                            return self._get_state(), reward, done, {
                                "score": self.score, 
                                "turn": self.turn,
                                "penalty_456": self.penalty_456,
                                "complete": self.complete,
                                "mz_score": self.mz_score,
                                "hand_when_complete": self.hand_when_complete
                            }
                        else:
                            # Non-agent player wins with ron
                            print(f"The other player win.")
                            done = True
                            reward -= 200  # Penalty for agent when another player wins
                            self.score += reward
                            return self._get_state(), reward, done, {
                                "score": self.score, 
                                "turn": self.turn,
                                "penalty_456": self.penalty_456,
                                "complete": self.complete,
                                "mz_score": self.mz_score,
                                "hand_when_complete": self.hand_when_complete
                            }

            pon_performed = False
            # ④ ぽん (pon) check for all other players
            for pon_player_idx in range(self.N_PLAYERS):
                if pon_player_idx != player_idx:  # Check all players except the current one
                    pon_hand = self.全手牌[pon_player_idx]
                    
                    # Count how many matching tiles the player has
                    matching_tiles = [t for t in pon_hand if 
                                    t.何者 == discarded_tile.何者 and 
                                    t.その上の数字 == discarded_tile.その上の数字]
                    
                    # Player needs at least 2 matching tiles to pon
                    if len(matching_tiles) >= 2:
                        # If the player is the agent, let them decide
                        if pon_player_idx == 0:
                            tile_idx = self._tile_to_index(discarded_tile)
                            pon_action = tile_idx + self.N_TILE_TYPES
                            valid = self.get_valid_actions()
                            if pon_action in valid:
                               chosen = self.agent.act(
                                   self._get_state(), valid
                                )
                               should_pon = (chosen == pon_action)
                            else:
                                should_pon = False

                            if should_pon:
                                print(f"Agent calls ぽん!")
                                # Remove 2 matching tiles from hand
                                for _ in range(2):
                                    for i, t in enumerate(pon_hand):
                                        if (t.何者 == discarded_tile.何者 and 
                                            t.その上の数字 == discarded_tile.その上の数字):
                                            pon_hand.pop(i)
                                            break
                                
                                self.副露[pon_player_idx].append({
                                    'type': 'pon',
                                    'tiles': matching_tiles[:2] + [discarded_tile],
                                    'from_player': player_idx
                                })
                                
                                # Set flag to indicate pon was performed
                                pon_performed = True
                                reward += 50  # Small reward for successfully calling pon
                                
                                # After pon, play continues from the player who called pon
                                self.current_player = pon_player_idx
                                break
                        else:
                            if random.random() < 0.4: 
                                print(f"Player {pon_player_idx} calls ぽん!")
                                for _ in range(2):
                                    for i, t in enumerate(pon_hand):
                                        if (t.何者 == discarded_tile.何者 and 
                                            t.その上の数字 == discarded_tile.その上の数字):
                                            pon_hand.pop(i)
                                            break
                                
                                self.副露[pon_player_idx].append({
                                    'type': 'pon',
                                    'tiles': matching_tiles[:2] + [discarded_tile],
                                    'from_player': player_idx
                                })
                                
                                pon_performed = True
                                
                                self.current_player = pon_player_idx
                                break
            
            if pon_performed:
                continue

            if player_idx == 0:
                聴牌, 何の牌 = 聴牌ですか(player_hand)
                if 聴牌:
                    print(f"聴牌:")
                    for p in 何の牌:
                        print(f"{p.何者} {p.その上の数字}")
                    reward += 500
                    reward += len(何の牌) * 50
            
                # 面子スコア calculation remains the same
                self.mz_score = 面子スコア(self.手牌)
                reward += self.mz_score * 8

            self.current_player = (self.current_player + 1) % self.N_PLAYERS
            
            if player_idx == 0:
                self.score += reward
                self.turn += 1
            
            if not self.山:
                done = True
                return self._get_state(), reward, done, {
                    "score": self.score, 
                    "turn": self.turn,
                    "penalty_456": self.penalty_456,
                    "complete": self.complete,
                    "mz_score": self.mz_score,
                    "hand_when_complete": self.hand_when_complete
                }
                
        return self._get_state(), reward, False, {
            "score": self.score, 
            "turn": self.turn,
            "penalty_456": self.penalty_456,
            "complete": self.complete,
            "mz_score": self.mz_score,
            "hand_when_complete": self.hand_when_complete
        }


# ---------------------------
# DQN
# ---------------------------
class DQNAgent:

    TARGET_UPDATE_EVERY = 2000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.memory = deque(maxlen=50_000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self._step_count = 0

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    # --------------------------------------------------
    # 行動選択
    # --------------------------------------------------
    def act(self, state: np.ndarray, valid_actions: list[int]) -> int:
        # ε‐greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(state_t)[0]
            q_valid = q_vals[valid_actions]
            best_idx = q_valid.argmax().item()
            return valid_actions[best_idx]

    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    def memorize(self, *transition):
        self.memory.append(transition)

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        hand_part = state_batch[:, :34*5]
        counts = hand_part.view(B, 34, 5).argmax(dim=2)
        
        return counts > 0

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states      = torch.as_tensor(np.array(states),      dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions,               dtype=torch.int64 , device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards,               dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones,                 dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.model(states).gather(1, actions)

        # y = r + γ max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_model(next_states)
            valid_mask = self._valid_mask_from_state(next_states)
            q_next[~valid_mask] = -1e9                       
            q_next_max = q_next.max(1, keepdim=True)[0]      
            y = rewards + self.gamma * q_next_max * (1 - dones)

        loss = self.criterion(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---------------------------
# トレーニングループ
# ---------------------------

def train_agent(episodes: int = 5000, pretrained: str | None = None, device: str = "cuda") -> DQNAgent:
    env = MahjongEnvironment()
    
    state_size = env.N_TILE_TYPES * env.STATE_BITS + env.N_TILE_TYPES * env.N_PLAYERS
    
    agent = DQNAgent(state_size, env.N_TILE_TYPES, device=device)

    if pretrained and os.path.exists(pretrained):
        agent.model.load_state_dict(torch.load(pretrained, map_location=device))
        agent.update_target_model()
        agent.epsilon = 0.005
        print(f"[INFO] 既存モデル {pretrained} をロードしました。")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            if env.current_player == 0:
                valid = env.get_valid_actions()
                action = agent.act(state, valid)
                next_state, reward, done, info = env.step(action)
                agent.memorize(state, action, reward, next_state, done)
                agent.replay(batch_size)
                state = next_state
                ep_reward += reward
            else:
                # If it's not the agent's turn, just step the environment with a dummy action
                # The environment will handle the non-agent players internally
                _, _, done, _ = env.step(-1)  # Dummy action, won't be used

        agent.decay_epsilon()
        with open("training_log_4players.csv", "a") as f:
            if ep == 0:  # Write header only once
                f.write("Episode,Score,Turn,Epsilon,Reward,Penalty456,Complete,MZScore,HandComplete\n")
            f.write(f"{ep+1},{info['score']},{info['turn']},{agent.epsilon:.3f},{ep_reward},{info['penalty_456']},"
                    f"{' '.join(str(x) for x in info['complete'])},{info['mz_score']}, {' '.join(info['hand_when_complete'])}\n")

    return agent


if __name__ == "__main__":
    trained_agent = train_agent(pretrained="model.pth")
    torch.save(trained_agent.model.state_dict(), "model.pth")
