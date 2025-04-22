import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from 牌 import 対子スコア, 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

    def __init__(self):
        self.agent_tile_count = 13
        self.sandbag_tile_count = 13 # slightly reduced difficulty
        self.reset()

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------

    def 配牌を割り当てる(self, 山: list, agent_count: int, sandbag_count: int):
        assert len(山) >= agent_count + 3 * sandbag_count, "山に十分な牌がありません"

        agent_tiles = 山[:agent_count]
        sandbag_a = 山[agent_count:agent_count + sandbag_count]
        sandbag_b = 山[agent_count + sandbag_count:agent_count + 2 * sandbag_count]
        # sandbag_c = 山[agent_count + 2 * sandbag_count:agent_count + 3 * sandbag_count]
        remaining = 山[agent_count + 2 * sandbag_count:]

        return agent_tiles, sandbag_a, sandbag_b, remaining
    

    def reset(self):
        self.手牌: list[麻雀牌]
        self.山 = 山を作成する()
        (
            self.手牌,
            self.sandbag_a_tiles,
            self.sandbag_b_tiles,
            # self.sandbag_c_tiles,
            self.山
        ) = self.配牌を割り当てる(
            self.山,
            agent_count=self.agent_tile_count,
            sandbag_count=self.sandbag_tile_count
        )
        self.current_actor = 0  # 0=agent, 1=sandbag_a, 2=sandbag_b, 3=sandbag_c
        self.discard_pile = []  # Record what tiles are discarded in current game.
        self.seat = random.randint(0, 3)  # 0=東, 1=南, 2=西, 3=北
        self.score = 0
        self.turn  = 0
        self.penalty_A = 0
        self.total_tennpai = 0
        self.mz_score = 0
        self.tuiz_score = 0
        self.pon = 0
        self.完成した役 = []
        self.hand_when_complete = []
        self.agent_after_pon = False
        return self._get_state()


    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        counter = Counter(self._tile_to_index(t) for t in self.手牌)
        for idx in range(self.N_TILE_TYPES):
            cnt = counter.get(idx, 0)
            state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
        seat = np.zeros(4, dtype=np.float32) # 東南西北
        seat[self.seat] = 1.0
        # 捨て牌の枚数（連続値）← 最大136枚なので数値として問題なし
        discard_counts = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        for tile in self.discard_pile:
            discard_counts[self._tile_to_index(tile)] += 1.0
        discard_counts /= 4.0  # 正規化（最大4枚）
        return np.concatenate([state, seat, discard_counts]) # → 170 + 4 + 34 = 208


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
        if self.current_actor == 0:
            return sorted({self._tile_to_index(t) for t in self.手牌 if not t.副露})
        else:
            hand_counts = Counter(self._tile_to_index(t) for t in self.手牌 if not t.副露)
            # print(f"hand_counts: {hand_counts}")
            # hand_counts: Counter({22: 2, 0: 1, 1: 1, 2: 1, 9: 1, 11: 1, 20: 1, 26: 1, 30: 1, 27: 1, 31: 1, 33: 1})
            # 手牌2枚以上はポンできる、3枚以上はチーできる
            chiipon_actions = []
            # pon: 同じ牌が2枚以上ある場合、ポンできる
            if len([t for t in self.手牌 if not t.副露]) > 1:
                for tile_idx, cnt in hand_counts.items():
                    if cnt > 1: 
                        chiipon_actions.append(34 + tile_idx) # action 34 - 67: pon true for specific tile
                        chiipon_actions.append(68 + tile_idx) # action 68 - 102: pon false for specific tile
            # chii: Condition not implemented
            if len([t for t in self.手牌 if not t.副露]) > 2:
                # # 萬子 2-8: NO chii
                # chiipon_actions.extend([102, 103, 104, 105, 106, 107, 108])
                # # 萬子 2-8: chii
                # chiipon_actions.extend([109, 110, 111, 112, 113, 114, 115])
                # # 筒子 2-8: NO chii
                # chiipon_actions.extend([116, 117, 118, 119, 120, 121, 122])
                # # 筒子 2-8: chii
                # chiipon_actions.extend([123, 124, 125, 126, 127, 128, 129])
                # # 索子 2-8: NO chii
                # chiipon_actions.extend([130, 131, 132, 133, 134, 135, 136])
                # # 索子 2-8: chii
                # chiipon_actions.extend([137, 138, 139, 140, 141, 142, 143])
                # Chii 判定: 中張牌(2-8)に対して (n-1, n+1) の2枚があるか確認
                suits = ["萬子", "筒子", "索子"]
                for suit_index, suit in enumerate(suits):
                    numbers = sorted([t.その上の数字 for t in self.手牌 if t.何者 == suit and not t.副露])
                    number_set = set(numbers)
                    for n in range(2, 9):  # 中張牌: 2〜8
                        if (n - 1 in number_set) and (n + 1 in number_set):
                            # NO chii action
                            chiipon_actions.append(102 + (n - 2) + suit_index * 14)
                            # chii action
                            chiipon_actions.append(109 + (n - 2) + suit_index * 14)
            return chiipon_actions


    def _agent_extra_reward(self, 手牌: list[麻雀牌]) -> int:
        reward_extra = 0
        # 聴牌の場合、大量の報酬を与える
        聴牌, 何の牌 = 聴牌ですか(self.手牌)
        if 聴牌:
            reward_extra += 800
            # print(f"聴牌:")
            # for p in 何の牌:
            #     print(f"{p.何者} {p.その上の数字}")
            self.total_tennpai += len(何の牌)
            reward_extra += len(何の牌) * 100
        
        # 面子スコア: 13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
        # 0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
        mz_score = 面子スコア(self.手牌)
        self.mz_score += mz_score * 12
        reward_extra += mz_score * 12

        # 対子スコア: 13 枚の手牌から完成対子の最大数を求めて
        # 0対子→0, 1対子→1, 2対子→2, 3対子→4, 4対子→8, 5対子→16, 6対子→32を返す。副露は数えない。
        tuiz_score = 対子スコア(self.手牌)
        self.tuiz_score += tuiz_score * 4
        reward_extra += tuiz_score * 4

        # Big penalty for having 4,5,6.
        # for t in self.手牌:
        #     if t.何者 in {"萬子", "筒子", "索子"} and t.その上の数字 in {4,5,6}:
        #         reward_extra -= 40
        #         self.penalty_A += 40

        # Big reward for having 1,2,3 and 7,8,9 and 字牌
        # for t in self.手牌:
        #     if t.何者 in {"白ちゃん", "發ちゃん", "中ちゃん"}:
        #         reward_extra += 40

        return reward_extra


    def _step_agent(self, action: int, after_pon: bool):
        reward = 0
        assert len([t for t in self.手牌 if not t.副露]) > 0, "Agent's hand is empty"
        # ① ツモ
        if not after_pon:
            self.手牌.append(self.山.pop(0))
        a, b, c = 点数計算(self.手牌, self.seat) # This func returns 点数int, 完成した役list, 和了形bool
        if c:
            print(f"ツモ！{b}")
            self.完成した役 = b
            reward += int(a)
            done = True
            remaining_turns = 140 - self.turn
            assert remaining_turns >= 0
            reward += int((300 * remaining_turns) * 0.5)
            self.score += reward 
            self.hand_when_complete = [str(t) for t in self.手牌]
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}
            
        # ② 打牌
        # reward -= max(1, int(self.turn * 0.8)) # Will greatly hinder learning process
        reward -= 2  # base penalty
        tile_type = self._index_to_tile_type(action)
        # print(tile_type) # ('筒子', 5)
        target_tile = None
        for t in [t for t in self.手牌 if not t.副露]:
            if (t.何者, t.その上の数字) == tile_type:
                target_tile = t
                break
        if target_tile is None:
            raise ValueError(f"Invalid action: {action} (tile not in hand)")
        # Remove the tile from the hand and add it to the discard pile
        assert target_tile.副露 == False, f"Exposed tile can not be discarded! {target_tile.何者}{target_tile.その上の数字}"
        self.手牌.remove(target_tile)
        self.discard_pile.append(target_tile)
    
        # Reward by how good the agent has formed the hand
        reward += self._agent_extra_reward(self.手牌)

        done = not self.山
        self.score += reward
        self.turn  += 1
        self.current_actor += 1  # Next actor is sandbag_a
        # should be a cycle, but agent is always 0
        # if done, add hand
        if done:
            self.hand_when_complete = [str(t) for t in self.手牌]
        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}


    def _step_sandbag(self, action: int, full_action_value_dict: dict):
        # The sandbag cannot do anything but randomly discard a tile,
        # and the discarded tile is then added to the discard pile.
        # during sandbag's turn, the agent can "ロン", "ポン" the tile.
        # "ロン" is automatically checked like ツモ.
        # "ロン": try add the tile to the hand and check if it is a ツモ,
        # if so, it is a "ロン".
        # "ポン": agent, if having the 2 same tiles as the discarded tile,
        # can "ポン" the tile and add it to the hand, if so, mark the 3 tiles as 副露
        # by setting 牌.副露 = True and remove the tile in pile. 
        # After ポン, current_actor is set to the agent but
        # the agent cannot draw a tile this turn.
        # To decide whether to "ポン", use the action value dict, pon
        # if 34 + tile_idx > 68 + tile_idx, otherwise do not pon.


        # Determine which sandbag is currently active
        sandbag_tiles = None
        if self.current_actor == 1:
            sandbag_tiles = self.sandbag_a_tiles
        elif self.current_actor == 2:
            sandbag_tiles = self.sandbag_b_tiles
        # elif self.current_actor == 3: # The player is removed
        #     sandbag_tiles = self.sandbag_c_tiles
        
        # Draw a tile for the sandbag if there are tiles left in the mountain
        new_tile = self.山.pop(0)
        sandbag_tiles.append(new_tile)
        done = not self.山
    
        # Sandbag randomly discards a tile
        discard_idx = random.randint(0, len(sandbag_tiles) - 1)
        discarded_tile = sandbag_tiles.pop(discard_idx)
        self.discard_pile.append(discarded_tile)

        # Check if agent can ロン (win) with this tile
        temp_hand = self.手牌.copy()
        temp_hand.append(discarded_tile)
        点数, 役, 和了形 = 点数計算(temp_hand, self.seat)
        
        if 和了形:
            # Agent can ロン
            print(f"ロン！{役}")
            self.完成した役 = 役
            reward = int(点数)
            done = True
            remaining_turns = 140 - self.turn
            assert remaining_turns >= 0
            reward += int((300 * remaining_turns) * 0.5)
            self.score += reward
            self.hand_when_complete = [str(t) for t in temp_hand]
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}
    
        if done: # 注意：これで最後はポンできない。これは一般的なルールと違う。
            self.hand_when_complete = [str(t) for t in self.手牌]
            return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}
    
        # Only if the agent has more than 2 tiles in hand, pon is possible
        if len([t for t in self.手牌 if not t.副露]) > 1:
            # Check if agent can ポン (pong)
            # Count how many of the discarded tile type the agent has
            same_tile_count = 0
            same_tiles = []
            discarded_tile_idx = self._tile_to_index(discarded_tile)
            
            for i, tile in enumerate(self.手牌):
                if (tile.何者, tile.その上の数字) == (discarded_tile.何者, discarded_tile.その上の数字) and not tile.副露:
                    same_tile_count += 1
                    same_tiles.append(i)
            
            # If agent has at least 2 of the same tile, they can pon
            if same_tile_count >= 2:
                # The agent will decide to pon or not based on the action value dictionary
                pon_action = 34 + discarded_tile_idx
                no_pon_action = 68 + discarded_tile_idx
                
                if full_action_value_dict and pon_action in full_action_value_dict and no_pon_action in full_action_value_dict:
                    if full_action_value_dict[pon_action] > full_action_value_dict[no_pon_action]:
                        # print(f"ポン！{discarded_tile.何者} {discarded_tile.その上の数字}")
                        # Agent decides to pon
                        # Mark the 2 tiles in agent's hand as 副露
                        for i in sorted(same_tiles[:2], reverse=True):
                            self.手牌[i].mark_as_exposed()
                        
                        # Add the discarded tile to agent's hand and mark it as 副露
                        discarded_tile.mark_as_exposed()
                        self.手牌.append(discarded_tile)
                        
                        # Remove the discarded tile from the discard pile
                        self.discard_pile.pop()
                        
                        # Set current actor to agent for immediate play
                        self.current_actor = 0
                        self.agent_after_pon = True
                        # Return state with after_pon=True for next step
                        return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                        "penalty_A": self.penalty_A,
                                                        "tenpai": self.total_tennpai,
                                                        "mz_score": self.mz_score,
                                                        "tuiz_score": self.tuiz_score,
                                                        "completeyaku": self.完成した役,
                                                        "hand_when_complete": self.hand_when_complete}

        # Check if agent can チー (Chii)
        # Agent can only Chii if they have two consecutive tiles that can form a sequence with the discarded tile
        # Custom rule: agent can only chii if the discarded tile is 中張牌 (2-8), and the only allowed sequence is (n-1, n, n+1)
        # Custom rule: any player can chii from any other player
        if len([t for t in self.手牌 if not t.副露]) > 2:
            can_chii = False
            chii_tiles = []
            
            if "中張牌" in discarded_tile.固有状態:
                number = discarded_tile.その上の数字
                possible_sequence = [(discarded_tile.何者, number-1), (discarded_tile.何者, number+1)] # [n-1,n,n+1]
                
                # print(f"Checking sequence: {possible_sequence}")
                # Checking sequence: [('筒子', 6), ('筒子', 7)]
                check = 0
                # check if both these tiles are in the hand
                for combo in possible_sequence:
                    # combo: ('筒子', 6) then ('筒子', 7)
                    for p in self.手牌:
                        if p.何者 == combo[0] and p.その上の数字 == combo[1] and not p.副露:
                            check += 1
                            chii_tiles.append(p)
                            break
                assert check <= 2, "No, not happening"
                if check == 2:
                    can_chii = True
                else:
                    chii_tiles.clear()
                            

                if can_chii:
                    # The agent will decide to chii or not based on the action value dictionary
                    chii_action_base = {
                        "萬子": 109,  # 萬子 2-8 chii: 109-115
                        "筒子": 123,  # 筒子 2-8 chii: 123-129
                        "索子": 137,  # 索子 2-8 chii: 137-143
                    }[discarded_tile.何者]

                    chii_action_idx = chii_action_base + (number - 2)  # 対応するchiiアクション番号
                    no_chii_action_idx = chii_action_idx - 7         # 対応するNO chiiアクション番号（chii action - 7）
                    
                    if full_action_value_dict and chii_action_idx in full_action_value_dict and no_chii_action_idx in full_action_value_dict:
                        if full_action_value_dict[chii_action_idx] > full_action_value_dict[no_chii_action_idx]:
                            # print(f"チー！{discarded_tile.何者} {discarded_tile.その上の数字}")
                            # Mark the 2 tiles in hand that is the chii_tiles to be 副露
                            ct_t_check = 0
                            for ctn in chii_tiles:
                                assert ctn.副露 == False, f"No, not happening"
                            for t in self.手牌:
                                if t.何者 == chii_tiles[0].何者 and t.その上の数字 == chii_tiles[0].その上の数字 and t.赤ドラ == chii_tiles[0].赤ドラ and t.副露 == chii_tiles[0].副露:
                                    # print(f"Found tile: {t.何者} {t.その上の数字}")
                                    t.marked_a = True
                                    ct_t_check += 1
                                    break
                            for t in self.手牌:
                                if t.何者 == chii_tiles[1].何者 and t.その上の数字 == chii_tiles[1].その上の数字 and t.赤ドラ == chii_tiles[1].赤ドラ and t.副露 == chii_tiles[1].副露:
                                    # print(f"Found tile: {t.何者} {t.その上の数字}")
                                    t.marked_a = True
                                    ct_t_check += 1
                                    break
                            assert ct_t_check == 2, "No, not happening"

                            discarded_tile.mark_as_exposed()  # Mark the discarded tile as 副露
                            for t in self.手牌:
                                if t.marked_a:
                                    t.mark_as_exposed()
                            for t in self.手牌:
                                t.marked_a = False

                            self.手牌.append(discarded_tile)
                            self.discard_pile.pop()

                            self.current_actor = 0
                            self.agent_after_pon = True  # Reuse the same flag for simplicity
                            # Return state with after_pon=True for next step
                            return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                            "penalty_A": self.penalty_A,
                                                            "tenpai": self.total_tennpai,
                                                            "mz_score": self.mz_score,
                                                            "tuiz_score": self.tuiz_score,
                                                            "completeyaku": self.完成した役,
                                                            "hand_when_complete": self.hand_when_complete}



        # Move to next actor (cycle through 0-2) prev: 0-3
        self.current_actor = (self.current_actor + 1) % 3
        self.turn += 1
        
        # Return state, no reward for agent during sandbag turns
        return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                            "penalty_A": self.penalty_A,
                                            "tenpai": self.total_tennpai,
                                            "mz_score": self.mz_score,
                                            "tuiz_score": self.tuiz_score,
                                            "completeyaku": self.完成した役,
                                            "hand_when_complete": self.hand_when_complete}


    def step(self, action: int, after_pon: bool = False, full_action_value_dict: dict = None):
        # print(self.current_actor) 012 012 012
        if not self.山 and not after_pon:
            raise Exception("Trying to call step() when deck is empty.")
        self.agent_after_pon = False
        if self.current_actor == 0:
            return self._step_agent(action, after_pon)
        else:
            return self._step_sandbag(action, full_action_value_dict)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # determines how much prioritization is used
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def __len__(self):
        return len(self.buffer)
        
    def add(self, *transition, priority=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta_by_frame())
        weights /= weights.max()
        
        self.frame += 1
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()


class EpisodicMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        
    def add_episode(self, episode, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(episode)
            self.priorities.append(reward)
        else:
            # Replace lowest priority episode if this one is better
            min_idx = np.argmin(self.priorities)
            if reward > self.priorities[min_idx]:
                self.memory[min_idx] = episode
                self.priorities[min_idx] = reward
                
    def sample(self, n=1):
        if not self.memory:
            return []
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), min(n, len(self.memory)), p=probs, replace=False)
        return [self.memory[i] for i in indices]


class NewDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # For hand representation (34 tile types)
        self.hand_encoder = nn.Sequential(
            nn.Linear(34*5, 256),  # 34 tile types with counts 0-4
            nn.ReLU()
        )
        
        # For other game state features
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size - 34*5, 128),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        # Split input into hand and other features
        hand = x[:, :170]  # First 170 features (34 tiles × 5 possible counts)
        other = x[:, 170:]  # Remaining features
        
        # Process separately
        hand_features = self.hand_encoder(hand)
        state_features = self.state_encoder(other)
        
        # Combine and produce action values
        combined = torch.cat([hand_features, state_features], dim=1)
        return self.combined(combined)


class DQNAgent:
    TARGET_UPDATE_EVERY = 2000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        # Use prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        # Add episodic memory for valuable game sequences
        self.episodic_memory = EpisodicMemory(capacity=50)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        # Use the Mahjong-specific model instead of the sequential model
        self.model = NewDQN(state_size, action_size).to(self.device)
        self.target_model = NewDQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self._step_count = 0

    # --------------------------------------------------
    # 行動選択
    # --------------------------------------------------
    def act(self, state: np.ndarray, valid_actions: list[int]):
        # ε‐greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions), dict()

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.model(state_t)[0]  # shape: [102]
            # 有効アクションの中から最大Q値のアクションを選ぶ
            q_valid = q_vals[valid_actions]
            best_idx = q_valid.argmax().item()
            # すべてのアクションに対するQ値の辞書を返す
            q_dict = {i: q_vals[i].item() for i in range(len(q_vals))}
            return valid_actions[best_idx], q_dict

    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    def memorize(self, *transition):
        # Use add method of prioritized replay buffer
        self.memory.add(*transition)
        
    def add_episode_to_memory(self, episode, reward):
        # Add complete episodes with high rewards to episodic memory
        self.episodic_memory.add_episode(episode, reward)

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        tile_state = state_batch[:, :170]  # [B, 170]
        counts = tile_state.view(B, 34, 5).argmax(dim=2)  # [B, 34]
        mask34 = counts > 0  # [B, 34]
        mask_pon = torch.ones((B, 68), dtype=torch.bool, device=state_batch.device)
        mask_chii = torch.ones((B, 42), dtype=torch.bool, device=state_batch.device)
        full_mask = torch.cat([mask34, mask_pon, mask_chii], dim=1)  # [B, 144]
        return full_mask

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
            
        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Filter out invalid actions (-1)
        valid_indices = actions.squeeze(1) != -1
        states = states[valid_indices]
        actions = actions[valid_indices]
        rewards = rewards[valid_indices]
        next_states = next_states[valid_indices]
        dones = dones[valid_indices]
        weights = weights[valid_indices]
        
        # Update indices to match filtered tensors
        valid_indices_list = valid_indices.cpu().numpy()
        valid_idx_map = [i for i, valid in enumerate(valid_indices_list) if valid]
        filtered_indices = [indices[i] for i in valid_idx_map]

        if len(states) == 0:  # Skip if no valid actions in the batch
            return

        # Q(s,a)
        q_sa = self.model(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_model(next_states)
            valid_mask = self._valid_mask_from_state(next_states)
            q_next[~valid_mask] = -1e9
            q_next_max = q_next.max(1, keepdim=True)[0]
            y = rewards + self.gamma * q_next_max * (1 - dones)

        # TD error for prioritization
        td_error = torch.abs(q_sa - y).detach().cpu().numpy()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(filtered_indices, td_error + 1e-6)  # Small constant to ensure all priorities > 0
        
        # Weighted MSE loss
        loss = (weights * (q_sa - y) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()
            
        # Occasionally sample from episodic memory to reinforce learning from successful episodes
        if self._step_count % 500 == 0 and self.episodic_memory.memory:
            self._replay_from_episodic_memory()
            
    def _replay_from_episodic_memory(self, num_episodes=1):
        # Sample successful episodes and learn from them
        sampled_episodes = self.episodic_memory.sample(num_episodes)
        for episode in sampled_episodes:
            for transition in episode:
                self.memory.add(*transition, priority=2.0)  # Higher priority for good episodes

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes: int = 800000, pretrained: str | None = None, device: str = "cuda") -> DQNAgent:
    env = MahjongEnvironment()
    agent = DQNAgent(208, env.N_TILE_TYPES * 3 + 7 * 6, device=device) # state_size: 208, action_size: 102 + 42 

    if pretrained and os.path.exists(pretrained):
        agent.model.load_state_dict(torch.load(pretrained, map_location=device))
        agent.update_target_model()
        agent.epsilon = 0.035
        print(f"[INFO] 既存モデル {pretrained} をロードしました。")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            valid: list = env.get_valid_actions()
            if valid:
                action, avd = agent.act(state, valid)
            else:
                action = -1
                avd = None
            next_state, reward, done, info = env.step(action, env.agent_after_pon, full_action_value_dict=avd)
            agent.memorize(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            ep_reward += reward
            if done:
                agent.decay_epsilon()
                if env.seat == 0:
                    envseat = "East"
                elif env.seat == 1:
                    envseat = "South"
                elif env.seat == 2:
                    envseat = "West"
                elif env.seat == 3:
                    envseat = "North"
                with open("training_logv5_0.csv", "a") as f:
                    if ep == 0:  # Write header only once
                        f.write("Episode,Seat,Score,Turn,Epsilon,penalty_A,Yaku,MZ_Score,TZ_Score,Tenpai,HandComplete\n")
                    f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},{agent.epsilon:.3f},{info['penalty_A']},{' '.join(str(x) for x in info['completeyaku'])},{info['mz_score']},{info['tuiz_score']},{info['tenpai']},{' '.join(info['hand_when_complete'])}\n")

                # モデル保存
                if (ep + 1) % 1000 == 0:
                    save_path = f"saved_model_ep{ep+1}.pth"
                    torch.save(agent.model.state_dict(), save_path)
                    print(f"[INFO] モデルを保存しました: {save_path}")
                break

    return agent


if __name__ == "__main__":
    trained_agent = train_agent(pretrained="modelv5.pth")
    torch.save(trained_agent.model.state_dict(), "modelv5.pth")
