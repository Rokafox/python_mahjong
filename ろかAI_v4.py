import collections
import csv
from itertools import permutations
import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ろか麻雀 import calculate_weighted_preference_score, nicely_print_tiles, 刻子スコア, 基礎訓練山を作成する, 対子スコア, 搭子スコア, 聴牌ですか, 順子スコア, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    SUIT_MAPPING = {"萬子": 0, "筒子": 1, "索子": 2}
    HONOR_ID_MAPPING = {
        "東風": 1, "南風": 2, "西風": 3, "北風": 4,
        "白ちゃん": 5, "發ちゃん": 6, "中ちゃん": 7
    }
    _MAX_DISCARD_GROUPS = 14
    ACTION_PASS = _MAX_DISCARD_GROUPS + 0
    ACTION_PON = _MAX_DISCARD_GROUPS + 1
    ACTION_CHI = _MAX_DISCARD_GROUPS + 2
    AGENT_HAND_SIZE = 13

    def __init__(self, is_test: bool):
        self.agent_tile_count = self.AGENT_HAND_SIZE
        self.sandbag_tile_count = self.AGENT_HAND_SIZE
        self.action_lgid_to_canonical_map: dict[int, int] = {}
        self.reset()
        # test environment does not give penalty nor hand reward
        self.is_test_environment = is_test

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------

    def 配牌を割り当てる(self, 山: list, agent_count: int, sandbag_count: int):
        assert len(山) >= agent_count + 3 * sandbag_count, "山に十分な牌がありません"

        agent_tiles = 山[:agent_count]
        sandbag_a = 山[agent_count:agent_count + sandbag_count]
        # sandbag_b = 山[agent_count + sandbag_count:agent_count + 2 * sandbag_count]
        # sandbag_c = 山[agent_count + 2 * sandbag_count:agent_count + 3 * sandbag_count]
        remaining = 山[agent_count + sandbag_count:]

        return agent_tiles, sandbag_a, remaining
    

    def reset(self):
        self.手牌: list[麻雀牌]
        self.sandbag_a_tiles: list[麻雀牌]
        self.山: list[麻雀牌] = 山を作成する()
        # self.山 = 基礎訓練山を作成する()
        (
            self.手牌,
            self.sandbag_a_tiles,
            # self.sandbag_b_tiles,
            # self.sandbag_c_tiles, # Sandbag removed
            self.山
        ) = self.配牌を割り当てる(
            self.山,
            agent_count=self.agent_tile_count,
            sandbag_count=self.sandbag_tile_count
        )
        self.current_actor = 0  # 0=agent, 1=sandbag_a, 2=sandbag_b, 3=sandbag_c
        self.discard_pile: list[麻雀牌] = []  # Record what tiles are discarded in current game.
        self.seat = random.randint(0, 3)  # 0=東, 1=南, 2=西, 3=北
        self.score = 0
        self.turn  = 0
        self.penalty_A = 0
        self.is_tenpai: bool = False
        self.total_tennpai = 0
        self.mz_score = 0
        self.tuiz_score = 0
        self.tatsu_score = 0
        self.pon = 0
        self.完成した役 = []
        self.hand_when_complete = []
        self.agent_after_pon = False

        self.action_lgid_to_canonical_map.clear()
        return self._get_state_unified(last_tile=None)


    def _get_canonical_id(self, tile: 麻雀牌) -> int:
        """Still useful for comparing tiles or other internal logic."""
        if tile is None: return -1
        category, num = tile.何者, tile.その上の数字
        if category == "萬子": return num - 1
        if category == "筒子": return 9 + num - 1
        if category == "索子": return 18 + num - 1
        if category == "東風": return 27
        if category == "南風": return 28
        if category == "西風": return 29
        if category == "北風": return 30
        if category == "白ちゃん": return 31
        if category == "發ちゃん": return 32
        if category == "中ちゃん": return 33
        return -2 # Should not happen

    def _are_tiles_identical_type(self, tile1: 麻雀牌, tile2: 麻雀牌) -> bool:
        """Checks if two tiles are of the exact same type (e.g., 1 Man == 1 Man)."""
        if tile1 is None or tile2 is None:
            return tile1 is tile2 # Both None is True, one None is False
        # Using __eq__ defined in 麻雀牌 if available and appropriate,
        # or comparing canonical IDs.
        # return tile1 == tile2 # If __eq__ is defined for type matching
        return self._get_canonical_id(tile1) == self._get_canonical_id(tile2)


    def _get_tile_base_properties(self, tile: 麻雀牌 | None) -> tuple:
        """
        Extracts base properties (excluding local group ID).
        Returns: (suit_cat, number, is_exposed, is_honor, honor_id_val)
        """
        if tile is None:
            return (-1, 0, 0, 0, 0) # suit_cat, number, exposed, is_honor, honor_id

        suit_val, num_val = tile.何者, tile.その上の数字
        is_honor_val, honor_id_val, suit_cat = 0, 0, -1

        if suit_val in self.SUIT_MAPPING:
            suit_cat = self.SUIT_MAPPING[suit_val]
        else: # Honor tiles
            suit_cat = 3 # Dedicated category for honors
            is_honor_val = 1
            honor_id_val = self.HONOR_ID_MAPPING.get(suit_val, 0)
            num_val = 0

        is_exposed_val = tile.get_exposed_status()
        return (suit_cat, num_val, is_exposed_val, is_honor_val, honor_id_val)


    def _get_state_unified(self, last_tile: 麻雀牌 | None = None) -> np.ndarray:
        actual_hand_tiles = sorted(
            [tile for tile in self.手牌 if tile is not None],
            key=lambda t: self._get_canonical_id(t)
        )

        # Calculate tile relationships for 順子 detection
        tile_relationships = [0]  # First tile has no previous tile to compare
        for i in range(1, len(actual_hand_tiles)):
            current_tile = actual_hand_tiles[i]
            prev_tile = actual_hand_tiles[i-1]
            
            # Get properties for comparison
            current_suit, current_num = current_tile.何者, current_tile.その上の数字
            prev_suit, prev_num = prev_tile.何者, prev_tile.その上の数字
            
            # Default relationship: no relation
            relationship = 0
            
            # Check if same suit/honor category
            if current_suit == prev_suit:
                if current_num == prev_num:
                    # Same tile type (helps with 刻子 detection)
                    relationship = 1
                elif abs(current_num - prev_num) == 1:
                    # Sequential tiles (helps with 順子 detection)
                    relationship = 2
            
            tile_relationships.append(relationship)
        
        # Pad relationships to match hand size
        while len(tile_relationships) < self.AGENT_HAND_SIZE:
            tile_relationships.append(0)

        hand_local_group_ids = []
        canonical_to_group_map = {}
        next_available_group_id = 0
        
        for tile in actual_hand_tiles:
            tile_cano_id = self._get_canonical_id(tile)
            if tile_cano_id not in canonical_to_group_map:
                canonical_to_group_map[tile_cano_id] = next_available_group_id
                hand_local_group_ids.append(next_available_group_id)
                next_available_group_id += 1
            else:
                hand_local_group_ids.append(canonical_to_group_map[tile_cano_id])
        
        max_local_id_in_hand = next_available_group_id -1 if next_available_group_id > 0 else -1

        hand_features_list = []
        for i in range(self.AGENT_HAND_SIZE):
            if i < len(actual_hand_tiles): # Use actual_hand_tiles here
                tile = actual_hand_tiles[i]
                local_id = hand_local_group_ids[i]
                relationship_code = tile_relationships[i]  # Add the relationship code
                base_props = self._get_tile_base_properties(tile)
                hand_features_list.extend([local_id, relationship_code, *base_props])
            else: 
                padding_local_id = -1
                relationship_code = 0 
                base_props_padding = self._get_tile_base_properties(None)
                hand_features_list.extend([padding_local_id, relationship_code, *base_props_padding])
        
        hand_tiles_state = np.array(hand_features_list, dtype=np.float32)

        # --- Last Tile Encoding ---
        last_tile_local_id = -1 # Default for padding 
        last_tile_relationship = 0  # Default relationship for last tile
        base_props_last_tile = self._get_tile_base_properties(last_tile)

        if last_tile is not None:
            last_tile_cano_id = self._get_canonical_id(last_tile)
            if last_tile_cano_id in canonical_to_group_map:
                last_tile_local_id = canonical_to_group_map[last_tile_cano_id]
            else:
                last_tile_local_id = max_local_id_in_hand + 1
            
            # Check relationship with ANY tile in the hand
            has_same_tile = False
            has_sequential_tile = False
            
            if actual_hand_tiles:
                last_suit, last_num = last_tile.何者, last_tile.その上の数字
                
                for hand_tile in actual_hand_tiles:
                    hand_suit, hand_num = hand_tile.何者, hand_tile.その上の数字
                    
                    if last_suit == hand_suit:
                        if last_num == hand_num:
                            has_same_tile = True
                        elif abs(last_num - hand_num) == 1:
                            has_sequential_tile = True
                    
                    # If we already found both relationships, we can stop checking
                    if has_same_tile and has_sequential_tile:
                        break
            
            # Encode relationship as:
            # 0: No relationship
            # 1: Same tile exists
            # 2: Sequential tile exists
            # 3: Both relationships exist
            if has_same_tile and has_sequential_tile:
                last_tile_relationship = 3
            elif has_same_tile:
                last_tile_relationship = 1
            elif has_sequential_tile:
                last_tile_relationship = 2
        
        last_tile_state_list = [last_tile_local_id, last_tile_relationship, *base_props_last_tile]
        last_tile_state = np.array(last_tile_state_list, dtype=np.float32)
        
        # --- Seat Encoding ---
        seat_state = np.zeros(4, dtype=np.float32)
        seat_state[self.seat] = 1.0
        
        # --- Current Actor Encoding (One-Hot) ---
        actor_state = np.zeros(4, dtype=np.float32)
        # Ensure current_actor index is valid (0-3)
        if 0 <= self.current_actor < 4:
            actor_state[self.current_actor] = 1.0

        # state size calculation:
        # (AGENT_HAND_SIZE) * (1 local ID + 1 relationship + 5 base props) -> 13 * 7 = 91 (for hand slots)
        # + 1 * (1 local ID + 1 relationship + 5 base props)                -> 1 * 7 = 7  (for last tile slot)
        # + 4                                                               -> 4          (for seat)
        # + 4                                                               -> 4          (for actor)
        # Total: 91 + 7 + 4 + 4 = 106

        return np.concatenate([
            hand_tiles_state,  # (AGENT_HAND_SIZE) * 7 features
            last_tile_state,   # 1 * 7 features
            seat_state,        # 4 features
            actor_state        # 4 features
        ])


    def get_valid_actions(self, last_tile: 麻雀牌 | None = None, tenpai: bool = False) -> list[int]:
        """
        Generates a list of valid action IDs.
        For agent's turn (discard): action IDs are local group IDs of discardable tiles.
        For opponent's turn (call/pass): action IDs are special constants (14, 15, 16): (pass, pon, chii)
        """
        self.action_lgid_to_canonical_map.clear() # Clear map from previous turn

        if self.current_actor == 0: # Agent's turn: Discard a tile
            discardable_tiles_in_hand = [t for t in self.手牌 if not t.副露]
            effective_hand_for_action = [t for t in self.手牌] # instead of discardable_tiles_in_hand
            if last_tile:
                effective_hand_for_action.append(last_tile)
            if not effective_hand_for_action: # Should not happen in a normal game state
                raise Exception
            effective_hand_for_action.sort(key=lambda x: (x.sort_order, x.その上の数字)) # Relies on 麻雀牌.__lt__
            # Generate local group IDs specifically for this action context
            action_specific_lgids = []
            # Temp map for THIS call, to build action_lgid_to_canonical_map and get unique lgids
            temp_cano_to_action_lgid = {} 
            next_action_lgid = 0

            for tile in effective_hand_for_action:
                cano_id = self._get_canonical_id(tile)
                if cano_id not in temp_cano_to_action_lgid:
                    assigned_lgid = next_action_lgid
                    temp_cano_to_action_lgid[cano_id] = assigned_lgid
                    # Populate the instance map for use in step()
                    self.action_lgid_to_canonical_map[assigned_lgid] = cano_id 
                    next_action_lgid += 1
                else:
                    assigned_lgid = temp_cano_to_action_lgid[cano_id]
                if not tile.副露:
                    action_specific_lgids.append(assigned_lgid)

            if tenpai and last_tile:
                last_tile_cano_id = self._get_canonical_id(last_tile)
                # Find the action LGID assigned to this last_tile's type
                lgid_for_last_tile_action = temp_cano_to_action_lgid.get(last_tile_cano_id)
                
                if lgid_for_last_tile_action is not None:
                    # Ensure the map for step() is correctly populated for this single action
                    # (already done above if last_tile was processed, but good to be explicit if we were to optimize)
                    # self.action_lgid_to_canonical_map = {lgid_for_last_tile_action: last_tile_cano_id}
                    return [lgid_for_last_tile_action]
                else:
                    raise Exception
            # Return unique, sorted local group IDs that can be discarded
            unique_valid_action_lgids = sorted(list(set(action_specific_lgids)))
            return unique_valid_action_lgids
        
        else:
            return [self.ACTION_PASS, self.ACTION_PON, self.ACTION_CHI] # 14, 15, 16


    def map_action_lgid_to_canonical_id(self, action_lgid: int) -> int | None:
        """
        Maps a local group ID (used as an action) to the canonical tile ID it represents.
        This map is populated by get_valid_actions.
        """
        return self.action_lgid_to_canonical_map.get(action_lgid)



    def _agent_extra_reward(self, 手牌: list[麻雀牌], naki_tile_pon: 麻雀牌 | None = None,
                            naki_tile_chii: 麻雀牌 | None = None) -> int:
        reward_extra = 0

        # 聴牌の場合、大量の報酬を与える
        if self.is_tenpai:
            reward_extra += 300
            self.total_tennpai += 1
        
        mz_score = int(面子スコア(self.手牌) * 8)
        self.mz_score += mz_score
        reward_extra += mz_score

        tuiz_score = int(刻子スコア(self.手牌) * 4)
        self.tuiz_score += tuiz_score
        reward_extra += tuiz_score

        # tatsu_score = int(順子スコア(self.手牌, [[1,2,3],[4,5,6],[7,8,9]]) * 100)
        # self.tatsu_score += tatsu_score
        # reward_extra += tatsu_score

        tatsu_score = int(順子スコア(self.手牌) * 4)
        self.tatsu_score += tatsu_score
        reward_extra += tatsu_score

        # tuiz_score += int(対子スコア(self.手牌) * 8)
        # self.tuiz_score += tuiz_score
        # reward_extra += tuiz_score

        ht_counter = Counter((t.何者, t.その上の数字) for t in self.手牌)

        for key, cnt in ht_counter.items():
            if key[0] == "萬子": # "萬子", "筒子", "索子"
                reward_extra += 8 * cnt
                self.penalty_A -= 8 * cnt
            else:
                reward_extra -= 8 * cnt
                self.penalty_A += 8 * cnt

        # present_categories = set()
        # for tile in 手牌:
        #     if "四風牌" in tile.固有状態:
        #         present_categories.add("四風牌")
        #     elif "三元牌" in tile.固有状態:
        #         present_categories.add("三元牌")
        #     elif tile.何者 in ["筒子", "萬子", "索子"]:
        #          present_categories.add(tile.何者)
        # 五門_counter = len(present_categories)
        # assert 1 <= 五門_counter <= 5
        # if 五門_counter == 1:
        #     reward_extra += 5
        # elif 五門_counter == 2:
        #     reward_extra += 2
        # elif 五門_counter == 3:
        #     reward_extra -= 2
        # elif 五門_counter == 4:
        #     reward_extra -= 5
        # elif 五門_counter == 5:
        #     reward_extra -= 10
        # else:
        #     raise Exception

        return reward_extra


    def _step_agent(self, actor: "DQNAgent", after_pon: bool):
        reward = 0
        assert len([t for t in self.手牌 if not t.副露]) > 0, "Agent's hand is empty"

        newly_drawn_tile_index: int = -1
        tenpai_before_draw = False

        if not after_pon:
            tenpai_before_draw, list_of_tanpai = 聴牌ですか(self.手牌, self.seat)
            self.is_tenpai = tenpai_before_draw
            newly_drawn_tile = self.山.pop(0)
            self.手牌.append(newly_drawn_tile)
        else:
            newly_drawn_tile = None

        # ツモ
        a, b, c = 点数計算(self.手牌, self.seat) # This func returns 点数int, 完成した役list, 和了形bool
        if c:
            print(f"ツモ！{b}")
            self.完成した役 = b
            reward += int(a)
            done = True
            self.score += reward 
            self.hand_when_complete = nicely_print_tiles(self.手牌)
            return self._get_state_unified(newly_drawn_tile), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "tatsu_score": self.tatsu_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}, -1
            
        # 牌を捨てる
        valid_actions: list = self.get_valid_actions(last_tile=newly_drawn_tile,
                                                     tenpai=tenpai_before_draw)
        assert len(valid_actions) > 0
        action, full_dict = actor.act(self._get_state_unified(newly_drawn_tile), valid_actions)
        assert 0 <= action <= 13
        if not self.is_test_environment:
            # reward -= 2  # base penalty
            reward -= int(self.turn * 0.5)
        tile_type = self.map_action_lgid_to_canonical_id(int(action))
        # print(tile_type) # ('筒子', 5)
        target_tile = None
        for t in [t for t in self.手牌 if not t.副露]:
            if self._get_canonical_id(t) == tile_type:
                target_tile = t
                break
        if target_tile is None:
            print(nicely_print_tiles(self.手牌))
            print(action)
            print(tile_type)
            raise ValueError(f"Invalid action: {tile_type} not in hand or cannot be discarded.")


        # Remove the tile from the hand and add it to the discard pile
        assert target_tile.副露 == False, f"Exposed tile can not be discarded! {target_tile.何者}{target_tile.その上の数字}"
        self.手牌.remove(target_tile)
        self.discard_pile.append(target_tile)

        if not self.is_test_environment:
            # Reward by how good the agent has formed the hand
            reward += self._agent_extra_reward(self.手牌)

        done = not self.山
        self.score += reward
        self.turn  += 1
        self.current_actor += 1  # Next actor is sandbag_a
        # should be a cycle, but agent is always 0
        # if done, add hand
        if done:
            self.hand_when_complete = nicely_print_tiles(self.手牌)
        return self._get_state_unified(newly_drawn_tile), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "tatsu_score": self.tatsu_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, action


    def _step_sandbag(self, actor: "DQNAgent"):
        reward = 0
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

        sandbag_tiles = self.sandbag_a_tiles
        
        # Draw tile
        new_tile = self.山.pop(0)
        sandbag_tiles.append(new_tile)
        done = not self.山
    
        # Randomly discards a tile
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
            reward += int(点数)
            done = True
            self.score += reward
            self.hand_when_complete = nicely_print_tiles(temp_hand)
            return self._get_state_unified(discarded_tile), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "tatsu_score": self.tatsu_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, -1
    
        if done: # 注意：これで最後はポンできない。これは一般的なルールと違う。
            self.hand_when_complete = nicely_print_tiles(self.手牌)
            return self._get_state_unified(discarded_tile), 0, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "tatsu_score": self.tatsu_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, -1

        agent_did_nothing = False
        pass_pon_and_chii = False

        if self.is_tenpai:
            pass_pon_and_chii = True

        # Only if the agent has more than 2 tiles in hand, pon is possible
        if not pass_pon_and_chii and len([t for t in self.手牌 if not t.副露]) > 1:
            # Check if agent can ポン (pong)
            # Count how many of the discarded tile type the agent has
            same_tile_count = 0
            same_tiles = []
            
            for i, tile in enumerate(self.手牌):
                if (tile.何者, tile.その上の数字) == (discarded_tile.何者, discarded_tile.その上の数字) and not tile.副露:
                    same_tile_count += 1
                    same_tiles.append(i)
            
            # If agent has at least 2 of the same tile, they can pon
            if same_tile_count >= 2:
                valid_actions: list = self.get_valid_actions(discarded_tile)
                valid_actions.remove(16)
                assert 14 in valid_actions and 15 in valid_actions
                action, full_dict = actor.act(self._get_state_unified(discarded_tile), valid_actions)
                if action == 15:
                    # print(f"ポン！{discarded_tile.何者} {discarded_tile.その上の数字}")
                    agent_did_nothing = False
                    # Mark the 2 tiles in agent's hand as 副露
                    for i in sorted(same_tiles[:2], reverse=True):
                        self.手牌[i].mark_as_exposed("pon")
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    discarded_tile.mark_as_exposed("pon")
                    self.手牌.append(discarded_tile)
                    
                    # Remove the discarded tile from the discard pile
                    self.discard_pile.pop()
                    
                    # Set current actor to agent for immediate play
                    self.current_actor = 0
                    self.agent_after_pon = True

                    if not self.is_test_environment:
                        reward += self._agent_extra_reward(self.手牌, discarded_tile, None)
                    self.score += reward

                    return self._get_state_unified(discarded_tile), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "tatsu_score": self.tatsu_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}, 15
                else: # action 14
                    agent_did_nothing = True

        # Check if agent can Chii
        # Agent can only Chii if they have two consecutive tiles that can form a sequence with the discarded tile
        # Custom rule: agent can only chii if the discarded tile is 中張牌 (2-8), and the only allowed sequence is (n-1, n, n+1)
        # Custom rule: any player can chii from any other player
        if not pass_pon_and_chii and len([t for t in self.手牌 if not t.副露]) > 2:
            can_chii = False
            chii_tiles = []
            
            if "中張牌" in discarded_tile.固有状態:
                dt_num = discarded_tile.その上の数字
                possible_sequence = [(discarded_tile.何者, dt_num-1), (discarded_tile.何者, dt_num+1)] # [n-1,n,n+1]
                
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
                    valid_actions: list = self.get_valid_actions(discarded_tile)
                    valid_actions.remove(15)
                    assert 14 in valid_actions and 16 in valid_actions
                    action, full_dict = actor.act(self._get_state_unified(discarded_tile), valid_actions)
                    if action == 16:
                        # print(f"チー！{discarded_tile.何者} {discarded_tile.その上の数字}")
                        agent_did_nothing = False
                        # Mark the 2 tiles in hand that is the chii_tiles to be 副露
                        ct_t_check = 0
                        for ctn in chii_tiles:
                            assert ctn.副露 == False
                        for t in self.手牌:
                            if t.何者 == chii_tiles[0].何者 and t.その上の数字 == chii_tiles[0].その上の数字 and t.赤ドラ == chii_tiles[0].赤ドラ and t.副露 == chii_tiles[0].副露:
                                t.mark_as_exposed("chii")
                                ct_t_check += 1
                                break
                        for t in self.手牌:
                            if t.何者 == chii_tiles[1].何者 and t.その上の数字 == chii_tiles[1].その上の数字 and t.赤ドラ == chii_tiles[1].赤ドラ and t.副露 == chii_tiles[1].副露:
                                t.mark_as_exposed("chii")
                                ct_t_check += 1
                                break
                        assert ct_t_check == 2, "No, not happening"

                        discarded_tile.mark_as_exposed("chii")

                        self.手牌.append(discarded_tile)
                        self.discard_pile.pop()

                        self.current_actor = 0
                        self.agent_after_pon = True  # Reuse the same flag for simplicity

                        if not self.is_test_environment:
                            reward += self._agent_extra_reward(self.手牌, None, discarded_tile)
                        self.score += reward

                        return self._get_state_unified(discarded_tile), reward, done, {"score": self.score, "turn": self.turn,
                                                        "penalty_A": self.penalty_A,
                                                        "tenpai": self.total_tennpai,
                                                        "mz_score": self.mz_score,
                                                        "tuiz_score": self.tuiz_score,
                                                        "tatsu_score": self.tatsu_score,
                                                        "completeyaku": self.完成した役,
                                                        "hand_when_complete": self.hand_when_complete}, 16
                    else: # action 14
                        agent_did_nothing = True



        # Move to next actor (cycle through 0-2) prev: 0-3
        # 2025-04-25 NEW: only 2 players left
        self.current_actor = (self.current_actor + 1) % 2
        self.turn += 1
        final_action = -1
        if agent_did_nothing or pass_pon_and_chii:
            # print("The agent did not pon or chii.")
            final_action = 14

        if not self.is_test_environment:
            reward += self._agent_extra_reward(self.手牌)
        self.score += reward

        # Return state
        return self._get_state_unified(discarded_tile), reward, done, {"score": self.score, "turn": self.turn,
                                            "penalty_A": self.penalty_A,
                                            "tenpai": self.total_tennpai,
                                            "mz_score": self.mz_score,
                                            "tuiz_score": self.tuiz_score,
                                            "tatsu_score": self.tatsu_score,
                                            "completeyaku": self.完成した役,
                                            "hand_when_complete": self.hand_when_complete}, final_action


    def step(self, actor: "DQNAgent", after_pon: bool = False):
        # print(self.current_actor) 012 012 012
        if not self.山 and not after_pon:
            raise Exception("Trying to call step() when deck is empty.")
        self.agent_after_pon = False
        if self.current_actor == 0:
            return self._step_agent(actor, after_pon)
        else:
            return self._step_sandbag(actor)


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
    def __init__(self, capacity=300):
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
        # probs = np.array(self.priorities) / sum(self.priorities)
        # Random sample
        indices = np.random.choice(len(self.memory), min(n, len(self.memory)), p=None, replace=False)
        return [self.memory[i] for i in indices]


class NewDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x):
        # Process the entire state directly without splitting
        return self.model(x)


class DQNAgent:
    TARGET_UPDATE_EVERY = 2000
    EPISODIC_MEMORY_REPLAY_FREQ = 500
    SUCCESS_REWARD_THRESHOLD = 3000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu",
                 epsilon: float = 1.0, epsilon_min : float = 0.01, epsilon_decay: float = 0.998):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        self.episodic_memory = EpisodicMemory(capacity=3333)
        self.current_episode_transitions = []

        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 1e-3

        self.model = NewDQN(state_size, action_size).to(self.device)
        self.target_model = NewDQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self._step_count = 0

    def act(self, state: np.ndarray, valid_actions: list[int]):
        # ε‐greedy
        if random.random() < self.epsilon:
            # Return a random action from valid_actions
            # If valid_actions is empty, this will raise an error, need to handle that
            if not valid_actions:
                 return -1, {} # Return an invalid action if no valid actions (shouldn't happen in Mahjong normally)
            return random.choice(valid_actions), dict()

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.model(state_t)[0]  # shape: [action_size]
            # Mask invalid actions with a very low value
            q_vals_masked = q_vals.clone()
            invalid_action_indices = [i for i in range(self.action_size) if i not in valid_actions]
            q_vals_masked[invalid_action_indices] = -float('inf') # Use -inf to ensure they are not chosen

            # Select the action with the highest Q value among valid actions
            best_action_idx = torch.argmax(q_vals_masked).item()

            # Return Q values for all actions (for logging/debugging if needed)
            q_dict = {i: q_vals[i].item() for i in range(self.action_size)}
            return best_action_idx, q_dict

    def memorize(self, *transition):
        # Add transition to the main replay buffer
        self.memory.add(*transition)
        # Also append to the current episode's transitions
        self.current_episode_transitions.append(transition)

    def add_episode_to_memory(self, episode: list, reward: float):
        # Add complete episodes with high rewards to episodic memory
        # This method is called externally after an episode finishes
        self.episodic_memory.add_episode(episode, reward)
        print("Episode memorized.")

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Generates a batch of boolean masks indicating valid actions for each state.
        Valid actions depend on the current actor encoded in the state.
        """
        B = state_batch.size(0)
        # Initialize mask to all False
        mask = torch.zeros((B, self.action_size), dtype=torch.bool, device=state_batch.device)
        
        # Define indices for action constants
        action_pass_idx = 14
        action_pon_idx = 15
        action_chi_idx = 16
        
        # Updated state slicing indices based on the new structure with relationship feature
        # hand_features_end = self.AGENT_HAND_SIZE * 7 # 13 * 7 = 91
        # last_tile_features_start = hand_features_end # 91
        # last_tile_features_end = last_tile_features_start + 7 # 98
        # seat_features_start = last_tile_features_end # 98
        # seat_features_end = seat_features_start + 4 # 102
        # actor_features_start = seat_features_end # 102
        
        # Updated actor features start (91 for hand + 7 for last_tile + 4 for seat = 102)
        actor_features_start = 102
        
        for i in range(B):
            state = state_batch[i]
            
            # Determine current actor from the one-hot encoding
            actor_one_hot = state[actor_features_start : actor_features_start + 4]
            current_actor = torch.argmax(actor_one_hot).item()
            
            if current_actor == 0:  # Agent's turn: Discard actions
                # Slice with step 7 to get local_id (now at indices 0, 7, 14, etc.)
                hand_local_ids = state[0 : 13 * 7 : 7]  
                last_tile_local_id = state[13 * 7]  # Should be index 91
                
                all_local_ids = torch.cat((hand_local_ids, last_tile_local_id.unsqueeze(0)))
                valid_local_ids = all_local_ids[all_local_ids != -1].unique().long()  # Use .long() for indexing
                valid_local_ids_within_range = valid_local_ids[valid_local_ids < 14]
                
                if valid_local_ids_within_range.numel() > 0:
                    mask[i, valid_local_ids_within_range] = True
            
            else:  # Opponent's turn: Call/Pass actions
                if action_pass_idx < self.action_size:
                    mask[i, action_pass_idx] = True
                if action_pon_idx < self.action_size:
                    mask[i, action_pon_idx] = True
                if action_chi_idx < self.action_size:
                    mask[i, action_chi_idx] = True
        
        return mask

    def replay(self, batch_size: int = 64):
        # Periodically sample from episodic memory and add to main buffer
        if self._step_count % self.EPISODIC_MEMORY_REPLAY_FREQ == 0 and len(self.episodic_memory.memory) > 0:
            self._replay_from_episodic_memory()
            # print("Successful episode replayed.")

        if len(self.memory) < batch_size:
            return

        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(batch_size)

        # Check if sampling returned anything
        if not samples:
             return

        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert to tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Filter out invalid actions (-1) if any were stored.
        # Note: It's better practice to avoid storing invalid actions in the first place
        valid_transition_mask = actions.squeeze(1) != -1
        # Apply mask to all tensors
        states = states[valid_transition_mask]
        actions = actions[valid_transition_mask]
        rewards = rewards[valid_transition_mask]
        next_states = next_states[valid_transition_mask]
        dones = dones[valid_transition_mask]
        weights = weights[valid_transition_mask]
        # Need to also filter the original indices to match the filtered tensors
        filtered_indices = np.array(indices)[valid_transition_mask.cpu().numpy()]


        if len(states) == 0:  # Skip if no valid transitions in the batch after filtering
            # print("Warning: Batch contained only invalid transitions after filtering.") # Debugging
            return

        # Q(s,a)
        q_sa = self.model(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_model(next_states) # [B', action_size] (B' <= batch_size)
            valid_mask = self._valid_mask_from_state(next_states) # [B', action_size]
            q_next[~valid_mask] = -1e9 # Mask out invalid actions
            q_next_max = q_next.max(1, keepdim=True)[0] # [B', 1]
            y = rewards + self.gamma * q_next_max * (1 - dones) # [B', 1]

        td_error = torch.abs(q_sa - y).detach().cpu().numpy() # [B', 1] -> [B',]

        self.memory.update_priorities(filtered_indices, td_error) # td_error is already abs and 1e-6 + alpha is applied inside update_priorities

        # Weighted MSE loss
        # Ensure weights match the filtered batch size
        loss = (weights * (q_sa - y.detach()) ** 2).mean() # Use y.detach() here as target is fixed

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        # Update target model periodically
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()

        # Episodic memory replay happens at the start of replay() now.

    def _replay_from_episodic_memory(self, num_episodes=1):
        # print(f"Replaying from episodic memory. Steps: {self._step_count}") # Debugging
        sampled_episodes = self.episodic_memory.sample(num_episodes)
        added_count = 0
        for episode in sampled_episodes:
            high_priority_base = 2.0
            for transition in episode:
                 # Add transition to main buffer with boosted priority
                 # The add method calculates priority^alpha
                self.memory.add(*transition, priority=high_priority_base)
                added_count += 1
        # if added_count > 0: # Debugging
        #     print(f"Added {added_count} transitions from episodic memory to main buffer.")


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes: int = 2999, name: str = "agent",
                device: str = "cuda", save_every_this_ep: int = 1000,
                save_after_this_ep: int = 900,
                e = 1.0, em = 0.01, ed = 0.998) -> DQNAgent:
    env = MahjongEnvironment(False)
    agent = DQNAgent(106, 17, device=device, epsilon=e, epsilon_min=em, epsilon_decay=ed)
    assert save_every_this_ep >= 99, "Reasonable saving interval must be no less than 99."
    log_save_path = f"./log/train_{name}.csv"

    if os.path.exists(log_save_path):
        print(f"Training Log file {log_save_path} already exists.")
        raise FileExistsError(f"Log file {log_save_path} already exists.")
        # If appending, make sure header is not written again
        #  write_header = False
    else:
        write_header = True
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_save_path), exist_ok=True)


    # Load agent if name is provided and file exists
    model_save_dir = "./DQN_agents_candidates/"
    formal_agent_save_dir = "./DQN_agents/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{name}.pth")
    formal_model_path = os.path.join(formal_agent_save_dir, f"{name}.pth")

    if name and os.path.exists(formal_model_path):
        try:
            agent.model.load_state_dict(torch.load(formal_model_path, map_location=device))
            agent.update_target_model()
            # Optionally adjust epsilon if loading a trained agent
            agent.epsilon = agent.epsilon_min # fine-tuning
            print(f"[INFO] Agent {name} loaded from {formal_model_path}.")
            # If continuing training, might want to load prior episodic memory if saved
            # (Requires implementing save/load for episodic memory)
        except Exception as e:
            print(f"Error loading model {formal_model_path}: {e}")
            print("Starting training from scratch.")
    else:
        print(f"Training new agent: {name}")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        # Clear episode transitions at the start of each episode
        agent.current_episode_transitions = []

        while True:
            next_state, reward, done, info, action_taken = env.step(agent, env.agent_after_pon)
            agent.memorize(state, action_taken, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            ep_reward += reward

            if done:
                # Episode finished
                agent.decay_epsilon()

                # Add episode to episodic memory if successful
                # if ep_reward > agent.SUCCESS_REWARD_THRESHOLD:
                # Or: Add to memory as long as it is a win.
                yaku_list: list = info.get('completeyaku', [])
                if len(yaku_list) > 0:
                    # Pass the collected transitions and the total reward
                    agent.add_episode_to_memory(agent.current_episode_transitions, ep_reward)
                    # print(f"Episode {ep+1}: Added to episodic memory with reward {ep_reward}") # Debugging

                # Clear the temporary episode storage for the next episode
                agent.current_episode_transitions = []


                # Log results
                envseat = {0: "East", 1: "South", 2: "West", 3: "North"}.get(env.seat, "Unknown")
                try:
                    with open(log_save_path, "a", encoding="utf-8") as f:
                        if write_header: # Write header only once
                            f.write("Episode,Seat,Score,Turn,Epsilon,penalty_A,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete\n")
                            write_header = False # Don't write again
                        # Ensure log values are safe to write (e.g., convert None to string)
                        log_values = [
                            ep+1,
                            envseat,
                            info.get('score', 'N/A'),
                            info.get('turn', 'N/A'),
                            f"{agent.epsilon:.3f}",
                            info.get('penalty_A', 'N/A'),
                            ' '.join(str(x) for x in info.get('completeyaku', [])), # Handle potential None/empty list
                            info.get('mz_score', 'N/A'),
                            info.get('tuiz_score', 'N/A'),
                            info.get('tatsu_score', 'N/A'),
                            info.get('tenpai', 'N/A'),
                            info.get('hand_when_complete', 'N/A')
                        ]
                        f.write(",".join(map(str, log_values)) + "\n")

                except Exception as e:
                    print(f"Error writing to log file {log_save_path}: {e}")


                if (ep + 1) % save_every_this_ep == 0 and ep > save_after_this_ep:
                    save_path = os.path.join(model_save_dir, f"{name}_{ep+1}.pth")
                    try:
                        torch.save(agent.model.state_dict(), save_path)
                        print(f"[INFO] Agent saved: {save_path}")
                    except Exception as e:
                        print(f"Error saving model to {save_path}: {e}")
                break

    final_save_path = os.path.join(model_save_dir, f"{name}_final.pth")
    try:
        torch.save(agent.model.state_dict(), final_save_path)
        print(f"[INFO] Final agent saved: {final_save_path}")
    except Exception as e:
        print(f"Error saving final model to {final_save_path}: {e}")

    return agent


def test_agent(episodes: int, model_path: str, device: str = "cpu", target_yaku: str = "None") -> tuple[float, float, int]:
    env = MahjongEnvironment(True)
    agent = DQNAgent(106, 17, device=device)
    agent.epsilon = 0
    log_save_path = f"./log/test_{model_path.split('/')[-1].split('.')[0]}.csv"
    
    # Load pre-trained model
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Model loaded.")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    if log_save_path:
        if os.path.exists(log_save_path):
            print(f"Test file {log_save_path} already exist, skip.")
            return 0, 0, 0
        with open(log_save_path, "a", encoding="utf-8") as f:
            f.write("Episode,Seat,Score,Turn,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete,AgariRate\n")
    
    # Statistics tracking
    agari = 0
    total_score = 0
    target_yaku_cnt = 0
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while True:
            # Get action from agent but don't train (no memorize or replay)
            next_state, reward, done, info, action = env.step(agent, env.agent_after_pon)
            state = next_state
            ep_reward += reward
            
            if done:
                # Convert seat number to compass direction
                seat_names = ["East", "South", "West", "North"]
                envseat = seat_names[env.seat]
                
                # Track statistics
                total_score += info['score']
                
                if 'completeyaku' in info and info['completeyaku']:
                    agari += 1
                # print(target_yaku)
                # print(info["completeyaku"])
                if 'completeyaku' in info and target_yaku in info['completeyaku']:
                    target_yaku_cnt += 1
                
                # Calculate current Agari rate
                current_agari_rate = (agari / (ep + 1)) * 100
                
                # Log results
                if log_save_path:
                    with open(log_save_path, "a", encoding="utf-8") as f:
                        f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},"
                               f"{' '.join(str(x) for x in info['completeyaku'])},"
                               f"{info['mz_score']},{info['tuiz_score']},{info['tatsu_score']},"
                               f"{info['tenpai']},{info['hand_when_complete']},{current_agari_rate:.3f}\n")
                
                # Print progress
                if (ep + 1) % 500 == 0:
                    print(f"[TEST] Episode {ep+1}/{episodes} completed. "
                         f"Avg score: {total_score/(ep+1):.2f}, "
                         f"Agari rate: {current_agari_rate:.2f}% "
                         f"Target Yaku Complete: {target_yaku_cnt} times."
                         )
                break
    
    # Final statistics
    print(f"\n[TEST COMPLETE] Total episodes: {episodes}")
    print(f"Average score: {total_score/episodes:.2f}")
    print(f"Agari rate: {agari/episodes*100:.3f}%")
    print(f"Target Yaku completed {target_yaku_cnt} times.")
    return total_score/episodes, agari/episodes*100, target_yaku_cnt


def test_all_agent_candidates(episodes: int, device: str = "cpu", target_yaku: str = "None", delete_poor: bool = True, 
                              performance_method : int = 0) -> None:
    """
    performance_method: 0: avg_score * agari_rate, 1: agari_rate, 2: target yaku complete count.
    """
    
    if target_yaku == "None" and performance_method == 2:
        raise Exception

    agent_dir = "./DQN_agents_candidates/"
    formal_agent_dir = "./DQN_agents/"
    agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".pth")]
    
    # Track best agent performance
    best_agent_file = None
    best_performance = float('-inf')  # Initialize with worst possible score
    results = []
    
    # Test each agent and record results
    for f in agent_files:
        print(f"Testing agent: {f}")
        avg_score, agari_rate, tkc = test_agent(episodes, f"{agent_dir}{f}", device, target_yaku)
        
        # Calculate a combined performance metric (you can adjust this formula)
        if performance_method == 0:
            performance = avg_score * agari_rate
        elif performance_method == 1:
            performance = agari_rate
        elif performance_method == 2:
            performance = tkc
        else:
            raise Exception
        results.append((f, avg_score, agari_rate, performance))
        
        # Update best agent if current one performs better
        if performance > best_performance:
            best_performance = performance
            best_agent_file = f
    
    if not delete_poor:
        return None

    # Keep only the best agent, delete the others
    if best_agent_file:
        print(f"\n[BEST AGENT] {best_agent_file}")
        print(f"Average score: {results[results.index((best_agent_file, *[r for r in results if r[0] == best_agent_file][0][1:]))][1]:.2f}")
        print(f"Agari rate: {results[results.index((best_agent_file, *[r for r in results if r[0] == best_agent_file][0][1:]))][2]:.3f}%")
        print(f"Performance: {results[results.index((best_agent_file, *[r for r in results if r[0] == best_agent_file][0][1:]))][3]:.3f}.")

        # Delete all agents except the best one
        for f in agent_files:
            file_path = f"{agent_dir}{f}"
            formal_file_path = f"{formal_agent_dir}{f}"
            log_file_path = f"./log/test_{f.split('.')[0]}.csv"
            
            if f != best_agent_file:
                try:
                    # Delete model file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    
                    # Delete corresponding log file
                    if os.path.exists(log_file_path):
                        os.remove(log_file_path)
                        print(f"Deleted: {log_file_path}")
                except Exception as e:
                    print(f"Error deleting files: {e}")
            else:
                try:
                    os.rename(file_path, formal_file_path)
                except Exception as e:
                    print(f"Error moving best agent: {e}")

    else:
        print("No agents were evaluated.")


def test_all_agents(episodes: int, device: str = "cpu") -> None:
    agent_dir = "./DQN_agents/"
    agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".pth")]
    for f in agent_files:
        avg_score, agari_rate, tkc = test_agent(episodes, f"{agent_dir}{f}", device, "対々和")



def train_and_test_pipeline():
    agent_name = "I"
    for ab in "A":
        train_agent(399, name=agent_name + ab, device="cuda", save_every_this_ep=100, save_after_this_ep=50,
                    e=0.5, em=0.5)
        test_all_agent_candidates(500, "cuda", target_yaku="None", performance_method=1)


if __name__ == "__main__":
    train_and_test_pipeline()
    # test_all_agent_candidates(500, "cuda", delete_poor=False)
    # test_all_agents(1000, 'cuda')
    # test_agent(episodes=5000, model_path=f"./DQN_agents/7tui_hr_a_2800.pth", device="cuda")

