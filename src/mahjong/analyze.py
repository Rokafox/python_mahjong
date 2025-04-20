import torch
import numpy as np
from collections import Counter

from ai_v2 import DQNAgent, MahjongEnvironment
from 牌 import 聴牌ですか, 面子スコア, 麻雀牌

# Assuming you have the same environment and agent classes from your code

def analyze_hand(hand_tiles, agent, env):
    """Analyze a specific Mahjong hand using the trained agent.
    
    Args:
        hand_tiles: List of tiles in the format your environment uses
        agent: Trained DQNAgent
        env: MahjongEnvironment instance
    
    Returns:
        Dictionary with analysis results
    """
    # Set the environment to have these specific tiles
    env.手牌 = hand_tiles
    
    # Get state representation
    state = env._get_state()
    
    # Get valid actions (tiles that can be discarded)
    valid_actions = env.get_valid_actions()
    
    # Get Q-values for all actions
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.model(state_tensor)[0].cpu().numpy()
    
    # Create a ranking of all valid discard options
    discard_rankings = []
    for action in valid_actions:
        tile_type = env._index_to_tile_type(action)
        discard_rankings.append({
            'action_index': action,
            'tile_type': tile_type,
            'q_value': q_values[action]
        })
    
    # Sort by Q-value (higher is better)
    discard_rankings.sort(key=lambda x: x['q_value'], reverse=True)
    
    # Check if the hand is in tenpai (ready)
    # tenpai, waiting_tiles = 聴牌ですか(hand_tiles)
    
    # # Calculate other metrics
    # mz_score = 面子スコア(hand_tiles)
    
    return {
        'discard_rankings': discard_rankings,
        # 'tenpai': tenpai,
        # 'waiting_tiles': waiting_tiles if tenpai else [],
        # 'mz_score': mz_score
    }

# Example usage
def main():
    # Load environment and agent
    env = MahjongEnvironment()
    agent = DQNAgent(env.N_TILE_TYPES * env.STATE_BITS, env.N_TILE_TYPES, device="cuda")
    
    # Load the trained model
    agent.model.load_state_dict(torch.load("一通清筒子.pth", map_location="cuda"))
    agent.model.eval()  # Set to evaluation mode
    
    # Create a sample hand (modify this with the hand you want to analyze)
    # This is just an example, update with your actual tile representation
    sample_hand = [麻雀牌("筒子", 1, False), 麻雀牌("筒子", 1, False), 麻雀牌("筒子", 1, False),
                   麻雀牌("筒子", 2, False), 麻雀牌("筒子", 3, False), 麻雀牌("筒子", 4, False),
                   麻雀牌("筒子", 3, False), 麻雀牌("筒子", 4, False), 麻雀牌("筒子", 5, False),
                   麻雀牌("筒子", 9, False), 麻雀牌("筒子", 9, False), 麻雀牌("筒子", 9, False),
                   麻雀牌("筒子", 5, False), 麻雀牌("筒子", 7, False)]
    
    sample_hand.sort(key=lambda x: (x.sort_order, x.その上の数字)) # Does not matter
    # Analyze the hand
    analysis = analyze_hand(sample_hand, agent, env)
    
    # Print the results
    print("Hand Analysis:")
    print("Hand: ", " ".join(str(t) for t in sample_hand))
    
    print("\nBest discard options (ranked by Q-value):")
    for i, option in enumerate(analysis['discard_rankings']):
        tile_info = option['tile_type']
        tile_name = f"{tile_info[0]} {tile_info[1]}" if tile_info[1] > 0 else tile_info[0]
        print(f"{i+1}. {tile_name}: {option['q_value']:.3f}")
    
    # print(f"\nTenpai (Ready): {analysis['tenpai']}")
    # if analysis['tenpai']:
    #     print("Waiting for: ", " ".join(str(t) for t in analysis['waiting_tiles']))
    
    # print(f"MZ Score (面子スコア): {analysis['mz_score']}")

if __name__ == "__main__":
    main()