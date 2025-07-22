import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class StrategyEngine:
    """Advanced poker strategy engine with GTO concepts"""
    
    def __init__(self):
        self.load_strategy_data()
        self.position_order = ['UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN', 'SB', 'BB']
        
    def load_strategy_data(self):
        """Load strategy data from JSON files"""
        try:
            # Load hand ranges
            with open('data/hand_ranges.json', 'r') as f:
                self.hand_ranges = json.load(f)
        except FileNotFoundError:
            self.hand_ranges = self.generate_default_ranges()
        
        try:
            # Load GTO strategies
            with open('data/gto_strategies.json', 'r') as f:
                self.gto_data = json.load(f)
        except FileNotFoundError:
            self.gto_data = self.generate_default_gto()
    
    def generate_default_ranges(self) -> Dict:
        """Generate default hand ranges for different positions"""
        return {
            'cash_game': {
                'UTG': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo'],
                    '3bet': ['AA', 'KK', 'QQ', 'AKs', 'AKo'],
                    'call': ['88', '77', '66', '55', 'A9s', 'A8s', 'KQs', 'KJs']
                },
                'UTG+1': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'AKo', 'AQo', 'AJo'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'],
                    'call': ['77', '66', '55', '44', 'A8s', 'A7s', 'KQs', 'KJs', 'KTs']
                },
                'MP': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'AKo', 'AQo', 'AJo', 'KQs', 'KJs'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AQs', 'AKo'],
                    'call': ['66', '55', '44', '33', 'A7s', 'A6s', 'KTs', 'K9s', 'QJs', 'QTs']
                },
                'CO': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'K9s', 'KQo', 'KJo', 'QJs', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo'],
                    'call': ['55', '44', '33', '22', 'A9o', 'A8o', 'K8s', 'K7s', 'Q8s', 'J8s', '87s', '76s']
                },
                'BTN': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'KQo', 'KJo', 'KTo', 'K9o', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'QJo', 'QTo', 'Q9o', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'JTo', 'J9o', 'T9s', 'T8s', 'T7s', 'T6s', 'T9o', '98s', '97s', '96s', '87s', '86s', '76s', '75s', '65s'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'K9s'],
                    'call': ['77', '66', '55', '44', '33', '22', 'A4o', 'A3o', 'A2o', 'K8o', 'K7o', 'Q8o', 'J8o', 'T8o']
                },
                'SB': {
                    'open': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s', 'QJo', 'QTo', 'Q9o', 'Q8o', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s', 'JTo', 'J9o', 'J8o', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s', 'T9o', 'T8o', '98s', '97s', '96s', '95s', '94s', '93s', '92s', '98o', '87s', '86s', '85s', '84s', '83s', '82s', '87o', '76s', '75s', '74s', '73s', '72s', '76o', '65s', '64s', '63s', '62s', '54s', '53s', '52s', '43s', '42s', '32s'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'KQo', 'KJo'],
                    'call': ['66', '55', '44', '33', '22', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'K7o', 'K6o', 'Q7o', 'J7o', 'T7o']
                },
                'BB': {
                    'defend': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s', 'QJo', 'QTo', 'Q9o', 'Q8o', 'Q7o', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s', 'JTo', 'J9o', 'J8o', 'J7o', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s', 'T9o', 'T8o', 'T7o', '98s', '97s', '96s', '95s', '94s', '93s', '92s', '98o', '97o', '87s', '86s', '85s', '84s', '83s', '82s', '87o', '86o', '76s', '75s', '74s', '73s', '72s', '76o', '75o', '65s', '64s', '63s', '62s', '65o', '64o', '54s', '53s', '52s', '54o', '53o', '43s', '42s', '43o', '32s', '32o'],
                    '3bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'K9s', 'KQo']
                }
            },
            'tournament': {
                'early': {
                    'UTG': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AQs', 'AKo'],
                    'CO': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo', 'KQs'],
                    'BTN': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'AKo', 'AQo', 'AJo', 'KQs', 'KJs', 'QJs']
                },
                'middle': {
                    'UTG': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'AKo', 'AQo'],
                    'CO': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'AKo', 'AQo', 'AJo', 'KQs', 'KJs'],
                    'BTN': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs']
                },
                'late': {
                    'UTG': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'KQs', 'KJs', 'KTs', 'K9s', 'QJs', 'QTs', 'JTs', 'T9s'],
                    'CO': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'KQo', 'KJo', 'QJs', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s'],
                    'BTN': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'QJo', 'QTo', 'Q9o', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'JTo', 'J9o', 'T9s', 'T8s', 'T7s', 'T6s', 'T9o', '98s', '97s', '96s', '87s', '86s', '76s', '75s', '65s']
                }
            }
        }
    
    def generate_default_gto(self) -> Dict:
        """Generate default GTO strategy data"""
        return {
            'preflop': {
                'open_sizing': {
                    'early_position': 2.5,
                    'middle_position': 2.5,
                    'late_position': 2.0,
                    'small_blind': 3.0
                },
                '3bet_sizing': {
                    'in_position': 3.0,
                    'out_of_position': 3.5
                },
                '4bet_sizing': {
                    'for_value': 2.2,
                    'as_bluff': 2.5
                }
            },
            'postflop': {
                'c_bet_frequency': {
                    'dry_boards': 0.75,
                    'wet_boards': 0.65,
                    'paired_boards': 0.80
                },
                'c_bet_sizing': {
                    'small': 0.33,
                    'medium': 0.66,
                    'large': 1.0
                },
                'check_raise_frequency': {
                    'in_position': 0.12,
                    'out_of_position': 0.15
                }
            },
            'bet_sizing': {
                'value_bet': {
                    'thin': 0.5,
                    'medium': 0.75,
                    'thick': 1.0
                },
                'bluff': {
                    'small': 0.33,
                    'large': 1.5
                }
            }
        }
    
    def get_optimal_action(self, game_state: Dict) -> Dict:
        """Get optimal action recommendation for given game state"""
        position = game_state.get('position', 'BTN')
        hole_cards = game_state.get('hole_cards', ['As', 'Kh'])
        game_type = game_state.get('game_type', 'cash')
        stack_size = game_state.get('stack_size', 100)
        opponents = game_state.get('opponents', 5)
        
        # Convert hole cards to range format
        hand_str = self.cards_to_range_format(hole_cards)
        
        # Get position-based strategy
        if game_type == 'tournament':
            stage = self.get_tournament_stage(game_state)
            position_range = self.hand_ranges.get('tournament', {}).get(stage, {}).get(position, [])
        else:
            position_range = self.hand_ranges.get('cash_game', {}).get(position, {}).get('open', [])
        
        # Basic action determination
        action = self.determine_basic_action(hand_str, position_range, game_state)
        
        # Calculate sizing
        sizing = self.calculate_optimal_sizing(action, game_state)
        
        # Get equity and EV calculations
        equity_analysis = self.calculate_action_equity(game_state, action)
        
        return {
            'action': action,
            'sizing': sizing,
            'confidence': self.calculate_confidence(game_state, action),
            'reasoning': self.generate_reasoning(game_state, action),
            'equity': equity_analysis,
            'alternative_actions': self.get_alternative_actions(game_state)
        }
    
    def cards_to_range_format(self, cards: List[str]) -> str:
        """Convert card list to range format (e.g., ['As', 'Kh'] -> 'AKo')"""
        if len(cards) != 2:
            return 'unknown'
        
        card1_rank = cards[0][0]
        card1_suit = cards[0][1]
        card2_rank = cards[1][0]
        card2_suit = cards[1][1]
        
        # Sort ranks by value
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        if card1_rank == card2_rank:
            return f"{card1_rank}{card2_rank}"
        
        # Order by rank value
        if rank_values[card1_rank] > rank_values[card2_rank]:
            high_rank, low_rank = card1_rank, card2_rank
            high_suit, low_suit = card1_suit, card2_suit
        else:
            high_rank, low_rank = card2_rank, card1_rank
            high_suit, low_suit = card2_suit, card1_suit
        
        suited = 's' if card1_suit == card2_suit else 'o'
        return f"{high_rank}{low_rank}{suited}"
    
    def determine_basic_action(self, hand_str: str, position_range: List[str], game_state: Dict) -> str:
        """Determine basic action (fold, call, raise) based on hand and position"""
        if hand_str in position_range:
            # Consider stack sizes and opponents
            stack_size = game_state.get('stack_size', 100)
            opponents = game_state.get('opponents', 5)
            
            # Adjust for stack size
            if stack_size < 20:  # Short stack
                if self.is_premium_hand(hand_str):
                    return 'all_in'
                elif self.is_strong_hand(hand_str):
                    return 'raise'
                else:
                    return 'fold'
            elif stack_size > 200:  # Deep stack
                if self.is_premium_hand(hand_str):
                    return 'raise'
                elif self.is_speculative_hand(hand_str):
                    return 'call'
                else:
                    return 'raise'
            else:  # Standard stack
                return 'raise'
        else:
            return 'fold'
    
    def is_premium_hand(self, hand_str: str) -> bool:
        """Check if hand is premium"""
        premium = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']
        return hand_str in premium
    
    def is_strong_hand(self, hand_str: str) -> bool:
        """Check if hand is strong"""
        strong = ['TT', '99', '88', 'AQs', 'AJs', 'ATs', 'AQo', 'AJo', 'KQs', 'KJs']
        return hand_str in strong
    
    def is_speculative_hand(self, hand_str: str) -> bool:
        """Check if hand is speculative (good for deep stack play)"""
        speculative = ['77', '66', '55', '44', '33', '22', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                      'K9s', 'K8s', 'K7s', 'Q9s', 'Q8s', 'J9s', 'J8s', 'T9s', 'T8s', '98s', '97s', '87s', '86s',
                      '76s', '75s', '65s', '64s', '54s', '53s', '43s']
        return hand_str in speculative
    
    def calculate_optimal_sizing(self, action: str, game_state: Dict) -> float:
        """Calculate optimal bet sizing"""
        position = game_state.get('position', 'BTN')
        stack_size = game_state.get('stack_size', 100)
        big_blind = game_state.get('big_blind', 1)
        
        if action == 'raise':
            if position in ['UTG', 'UTG+1', 'MP']:
                return 2.5 * big_blind
            elif position in ['MP+1', 'CO']:
                return 2.5 * big_blind
            elif position == 'BTN':
                return 2.0 * big_blind
            elif position == 'SB':
                return 3.0 * big_blind
            else:
                return 2.5 * big_blind
        
        elif action == 'all_in':
            return stack_size
        
        elif action == 'call':
            return big_blind  # Assuming facing a standard open
        
        else:  # fold
            return 0
    
    def calculate_action_equity(self, game_state: Dict, action: str) -> Dict:
        """Calculate equity for the recommended action"""
        # Simplified equity calculation
        hole_cards = game_state.get('hole_cards', ['As', 'Kh'])
        position = game_state.get('position', 'BTN')
        opponents = game_state.get('opponents', 5)
        
        # Base equity estimation
        hand_str = self.cards_to_range_format(hole_cards)
        
        if self.is_premium_hand(hand_str):
            base_equity = 0.65
        elif self.is_strong_hand(hand_str):
            base_equity = 0.55
        elif self.is_speculative_hand(hand_str):
            base_equity = 0.45
        else:
            base_equity = 0.35
        
        # Adjust for position
        position_adjustment = {
            'UTG': -0.05, 'UTG+1': -0.03, 'MP': -0.02, 'MP+1': 0,
            'CO': 0.02, 'BTN': 0.05, 'SB': -0.03, 'BB': 0
        }
        
        adjusted_equity = base_equity + position_adjustment.get(position, 0)
        
        # Adjust for number of opponents
        opponent_penalty = (opponents - 1) * 0.02
        final_equity = max(0.1, adjusted_equity - opponent_penalty)
        
        return {
            'raw_equity': base_equity,
            'position_adjusted': adjusted_equity,
            'final_equity': final_equity,
            'fold_equity': self.estimate_fold_equity(game_state, action)
        }
    
    def estimate_fold_equity(self, game_state: Dict, action: str) -> float:
        """Estimate fold equity for aggressive actions"""
        if action not in ['raise', 'all_in']:
            return 0.0
        
        position = game_state.get('position', 'BTN')
        opponents = game_state.get('opponents', 5)
        
        # Base fold equity by position
        base_fold_equity = {
            'UTG': 0.20, 'UTG+1': 0.25, 'MP': 0.30, 'MP+1': 0.35,
            'CO': 0.40, 'BTN': 0.50, 'SB': 0.35, 'BB': 0.25
        }.get(position, 0.30)
        
        # Adjust for opponents remaining
        adjusted_fold_equity = base_fold_equity * (1 - (opponents - 1) * 0.1)
        
        return max(0.1, min(0.8, adjusted_fold_equity))
    
    def calculate_confidence(self, game_state: Dict, action: str) -> float:
        """Calculate confidence level for the recommendation"""
        hole_cards = game_state.get('hole_cards', ['As', 'Kh'])
        position = game_state.get('position', 'BTN')
        stack_size = game_state.get('stack_size', 100)
        
        hand_str = self.cards_to_range_format(hole_cards)
        
        # Base confidence
        if self.is_premium_hand(hand_str):
            base_confidence = 0.95
        elif self.is_strong_hand(hand_str):
            base_confidence = 0.85
        elif self.is_speculative_hand(hand_str):
            base_confidence = 0.70
        else:
            base_confidence = 0.60
        
        # Adjust for position clarity
        position_clarity = {
            'UTG': 0.90, 'UTG+1': 0.85, 'MP': 0.80, 'MP+1': 0.85,
            'CO': 0.90, 'BTN': 0.95, 'SB': 0.75, 'BB': 0.70
        }.get(position, 0.80)
        
        # Adjust for stack size complexity
        if 50 <= stack_size <= 150:
            stack_clarity = 1.0
        elif 20 <= stack_size < 50 or 150 < stack_size <= 300:
            stack_clarity = 0.85
        else:
            stack_clarity = 0.70
        
        final_confidence = base_confidence * position_clarity * stack_clarity
        return min(0.99, max(0.50, final_confidence))
    
    def generate_reasoning(self, game_state: Dict, action: str) -> str:
        """Generate human-readable reasoning for the action"""
        hole_cards = game_state.get('hole_cards', ['As', 'Kh'])
        position = game_state.get('position', 'BTN')
        stack_size = game_state.get('stack_size', 100)
        opponents = game_state.get('opponents', 5)
        
        hand_str = self.cards_to_range_format(hole_cards)
        
        reasoning_parts = []
        
        # Hand strength reasoning
        if self.is_premium_hand(hand_str):
            reasoning_parts.append(f"{hand_str} is a premium hand")
        elif self.is_strong_hand(hand_str):
            reasoning_parts.append(f"{hand_str} is a strong hand")
        elif self.is_speculative_hand(hand_str):
            reasoning_parts.append(f"{hand_str} has good implied odds potential")
        else:
            reasoning_parts.append(f"{hand_str} is below opening range")
        
        # Position reasoning
        if position in ['BTN', 'CO']:
            reasoning_parts.append("good position allows for wider range")
        elif position in ['UTG', 'UTG+1']:
            reasoning_parts.append("early position requires tighter range")
        elif position in ['SB', 'BB']:
            reasoning_parts.append("blind position has positional disadvantage")
        
        # Stack size reasoning
        if stack_size < 20:
            reasoning_parts.append("short stack favors all-in or fold strategy")
        elif stack_size > 200:
            reasoning_parts.append("deep stacks favor speculative hands with implied odds")
        
        # Opponent count reasoning
        if opponents > 6:
            reasoning_parts.append("many opponents require tighter range")
        elif opponents < 4:
            reasoning_parts.append("fewer opponents allow for wider range")
        
        return "; ".join(reasoning_parts)
    
    def get_alternative_actions(self, game_state: Dict) -> List[Dict]:
        """Get alternative actions with their pros/cons"""
        hole_cards = game_state.get('hole_cards', ['As', 'Kh'])
        hand_str = self.cards_to_range_format(hole_cards)
        
        alternatives = []
        
        if self.is_premium_hand(hand_str):
            alternatives.extend([
                {"action": "raise", "pros": "Build pot with strong hand", "cons": "May fold out worse hands"},
                {"action": "call", "pros": "Disguise hand strength", "cons": "Doesn't build pot"},
                {"action": "3bet", "pros": "Maximum value", "cons": "May fold entire field"}
            ])
        elif self.is_strong_hand(hand_str):
            alternatives.extend([
                {"action": "raise", "pros": "Good hand deserves action", "cons": "Vulnerable to 3bets"},
                {"action": "call", "pros": "Control pot size", "cons": "Miss value"},
                {"action": "fold", "pros": "Avoid difficult spots", "cons": "Too tight"}
            ])
        else:
            alternatives.extend([
                {"action": "fold", "pros": "Avoid marginal spot", "cons": "Miss potential"},
                {"action": "call", "pros": "See cheap flop", "cons": "Poor hand strength"},
                {"action": "bluff_raise", "pros": "Fold equity", "cons": "High risk"}
            ])
        
        return alternatives
    
    def get_range_analysis(self, position: str, action: str, game_type: str) -> Dict:
        """Get detailed range analysis for position and action"""
        if game_type == 'tournament':
            # Use middle stage as default for tournaments
            ranges = self.hand_ranges.get('tournament', {}).get('middle', {})
        else:
            ranges = self.hand_ranges.get('cash_game', {})
        
        position_data = ranges.get(position, {})
        action_range = position_data.get(action, [])
        
        # Categorize hands
        pairs = [h for h in action_range if len(h) == 2 and h[0] == h[1]]
        suited = [h for h in action_range if len(h) == 3 and h[2] == 's']
        offsuit = [h for h in action_range if len(h) == 3 and h[2] == 'o']
        
        # Calculate range statistics
        total_combos = len(action_range) * 6  # Rough estimate
        range_percentage = (total_combos / 1326) * 100  # 1326 total combos
        
        return {
            'position': position,
            'action': action,
            'game_type': game_type,
            'range': action_range,
            'pairs': pairs,
            'suited': suited,
            'offsuit': offsuit,
            'total_hands': len(action_range),
            'estimated_combos': total_combos,
            'range_percentage': round(range_percentage, 1),
            'tightness': self.calculate_range_tightness(action_range),
            'description': self.describe_range(action_range)
        }
    
    def calculate_range_tightness(self, hand_range: List[str]) -> str:
        """Calculate how tight/loose a range is"""
        premium_count = sum(1 for h in hand_range if self.is_premium_hand(h))
        strong_count = sum(1 for h in hand_range if self.is_strong_hand(h))
        total = len(hand_range)
        
        if total == 0:
            return "unknown"
        
        premium_ratio = premium_count / total
        strong_ratio = (premium_count + strong_count) / total
        
        if premium_ratio > 0.8:
            return "extremely tight"
        elif strong_ratio > 0.7:
            return "tight"
        elif strong_ratio > 0.4:
            return "balanced"
        elif strong_ratio > 0.2:
            return "loose"
        else:
            return "very loose"
    
    def describe_range(self, hand_range: List[str]) -> str:
        """Generate description of the range"""
        if not hand_range:
            return "No hands in range"
        
        pairs = [h for h in hand_range if len(h) == 2]
        suited = [h for h in hand_range if len(h) == 3 and h[2] == 's']
        offsuit = [h for h in hand_range if len(h) == 3 and h[2] == 'o']
        
        descriptions = []
        
        if pairs:
            if 'AA' in pairs and 'KK' in pairs:
                descriptions.append("premium pairs")
            elif any(p in pairs for p in ['TT', '99', '88']):
                descriptions.append("medium-strong pairs")
            if any(p in pairs for p in ['77', '66', '55', '44', '33', '22']):
                descriptions.append("small pairs")
        
        if suited:
            if any(h.startswith('A') for h in suited):
                descriptions.append("suited aces")
            if any(h.startswith('K') for h in suited):
                descriptions.append("suited kings")
            if any(h in suited for h in ['QJs', 'JTs', 'T9s', '98s']):
                descriptions.append("suited connectors")
        
        if offsuit:
            if any(h.startswith('A') for h in offsuit):
                descriptions.append("offsuit aces")
            if any(h.startswith('K') for h in offsuit):
                descriptions.append("offsuit broadway")
        
        return ", ".join(descriptions) if descriptions else "mixed range"
    
    def get_tournament_strategy(self, tournament_state: Dict) -> Dict:
        """Get tournament-specific strategy recommendations"""
        stage = tournament_state.get('stage', 'early')
        stack_size = tournament_state.get('stack_size', 50)
        avg_stack = tournament_state.get('avg_stack', 50)
        players_left = tournament_state.get('players_left', 100)
        blind_level = tournament_state.get('blind_level', 1)
        bubble_factor = tournament_state.get('bubble_factor', 1.0)
        
        # Calculate effective stack in big blinds
        effective_stack = stack_size / blind_level
        stack_ratio = stack_size / avg_stack
        
        # Determine strategy adjustments
        strategy_adjustments = {
            'range_tightness': self.calculate_tournament_tightness(stage, effective_stack, bubble_factor),
            'aggression_level': self.calculate_tournament_aggression(stack_ratio, stage),
            'stack_preservation': self.calculate_stack_preservation(effective_stack, bubble_factor),
            'bubble_considerations': self.get_bubble_strategy(bubble_factor, players_left)
        }
        
        # Get specific recommendations
        recommendations = self.generate_tournament_recommendations(tournament_state, strategy_adjustments)
        
        return {
            'stage': stage,
            'effective_stack': effective_stack,
            'stack_ratio': stack_ratio,
            'strategy_adjustments': strategy_adjustments,
            'recommendations': recommendations,
            'key_focuses': self.get_tournament_key_focuses(stage, effective_stack, bubble_factor)
        }
    
    def calculate_tournament_tightness(self, stage: str, effective_stack: float, bubble_factor: float) -> str:
        """Calculate how tight to play in tournament"""
        if bubble_factor > 1.5:  # Near bubble
            return "tight"
        elif stage == 'early' and effective_stack > 100:
            return "normal"
        elif effective_stack < 20:
            return "loose_aggressive"
        elif stage == 'late':
            return "loose"
        else:
            return "normal"
    
    def calculate_tournament_aggression(self, stack_ratio: float, stage: str) -> str:
        """Calculate aggression level for tournament"""
        if stack_ratio > 2.0:  # Big stack
            return "high"
        elif stack_ratio < 0.5:  # Short stack
            return "selective_aggression"
        elif stage == 'late':
            return "high"
        else:
            return "medium"
    
    def calculate_stack_preservation(self, effective_stack: float, bubble_factor: float) -> float:
        """Calculate how much to focus on stack preservation"""
        preservation_score = 0.5  # Base
        
        if effective_stack < 15:
            preservation_score -= 0.3  # Must take risks
        elif effective_stack > 100:
            preservation_score += 0.2  # Can afford to be careful
        
        if bubble_factor > 1.5:
            preservation_score += 0.3  # Bubble play
        
        return max(0.0, min(1.0, preservation_score))
    
    def get_bubble_strategy(self, bubble_factor: float, players_left: int) -> Dict:
        """Get bubble-specific strategy"""
        if bubble_factor < 1.2:
            return {"status": "not_bubble", "adjustments": "normal play"}
        elif bubble_factor < 1.5:
            return {"status": "approaching_bubble", "adjustments": "slightly tighter"}
        else:
            return {
                "status": "bubble_play",
                "adjustments": "exploit short stacks, avoid medium stacks",
                "key_principle": "pressure short stacks, avoid big stacks"
            }
    
    def generate_tournament_recommendations(self, tournament_state: Dict, adjustments: Dict) -> List[str]:
        """Generate specific tournament recommendations"""
        recommendations = []
        
        stage = tournament_state.get('stage', 'early')
        effective_stack = tournament_state.get('stack_size', 50) / tournament_state.get('blind_level', 1)
        
        if stage == 'early':
            recommendations.extend([
                "Play tight-aggressive with premium hands",
                "Avoid marginal spots without good odds",
                "Build stack gradually with strong hands"
            ])
        elif stage == 'middle':
            recommendations.extend([
                "Increase aggression with antes in play",
                "Steal blinds from late position",
                "3bet light against loose openers"
            ])
        else:  # late
            recommendations.extend([
                "Maximum aggression with fold equity",
                "Target shorter stacks for pressure",
                "Use ICM considerations in decisions"
            ])
        
        if effective_stack < 15:
            recommendations.append("Consider push/fold strategy")
        elif effective_stack > 100:
            recommendations.append("Play speculative hands for implied odds")
        
        return recommendations
    
    def get_tournament_key_focuses(self, stage: str, effective_stack: float, bubble_factor: float) -> List[str]:
        """Get key focuses for tournament stage"""
        focuses = []
        
        if bubble_factor > 1.5:
            focuses.append("ICM considerations")
            focuses.append("Stack preservation")
        
        if effective_stack < 20:
            focuses.append("Push/fold decisions")
            focuses.append("All-in timing")
        
        if stage == 'late':
            focuses.append("Blind stealing")
            focuses.append("Short stack pressure")
        
        focuses.append("Position awareness")
        focuses.append("Stack size management")
        
        return focuses
    
    def get_gto_strategy(self, situation: Dict) -> Dict:
        """Get GTO strategy for specific situation"""
        street = situation.get('street', 'preflop')
        action = situation.get('action', 'open')
        position = situation.get('position', 'BTN')
        bet_size = situation.get('bet_size', 2.5)
        pot_size = situation.get('pot_size', 1.5)
        
        gto_strategy = {
            'situation': situation,
            'optimal_frequencies': self.get_optimal_frequencies(situation),
            'sizing_recommendations': self.get_gto_sizing(situation),
            'range_construction': self.get_gto_range_construction(situation),
            'exploitative_adjustments': self.get_exploitative_adjustments(situation)
        }
        
        return gto_strategy
    
    def get_optimal_frequencies(self, situation: Dict) -> Dict:
        """Get optimal action frequencies for GTO play"""
        street = situation.get('street', 'preflop')
        action = situation.get('action', 'open')
        
        if street == 'preflop':
            if action == 'open':
                return {
                    'value_hands': 0.70,
                    'bluff_hands': 0.30,
                    'total_frequency': 0.25  # 25% of hands
                }
            elif action == '3bet':
                return {
                    'value_hands': 0.65,
                    'bluff_hands': 0.35,
                    'total_frequency': 0.08  # 8% of hands
                }
        else:  # postflop
            if action == 'c_bet':
                return {
                    'value_hands': 0.60,
                    'bluff_hands': 0.40,
                    'total_frequency': 0.70
                }
            elif action == 'check_raise':
                return {
                    'value_hands': 0.75,
                    'bluff_hands': 0.25,
                    'total_frequency': 0.15
                }
        
        return {'value_hands': 0.70, 'bluff_hands': 0.30, 'total_frequency': 0.50}
    
    def get_gto_sizing(self, situation: Dict) -> Dict:
        """Get GTO sizing recommendations"""
        street = situation.get('street', 'preflop')
        pot_size = situation.get('pot_size', 1.5)
        
        if street == 'preflop':
            return {
                'open_size': 2.5,
                '3bet_size': 3.0,
                '4bet_size': 2.2,
                'reasoning': 'Standard preflop sizing'
            }
        else:
            return {
                'small_bet': 0.33 * pot_size,
                'medium_bet': 0.66 * pot_size,
                'large_bet': 1.0 * pot_size,
                'overbet': 1.5 * pot_size,
                'recommended': 0.66 * pot_size,
                'reasoning': 'Balanced postflop sizing'
            }
    
    def get_gto_range_construction(self, situation: Dict) -> Dict:
        """Get GTO range construction principles"""
        street = situation.get('street', 'preflop')
        action = situation.get('action', 'open')
        
        return {
            'value_range': 'Top tier hands for value',
            'bluff_range': 'Hands with blockers and equity',
            'balance_principle': 'Mix strong and weak hands',
            'frequency_targets': self.get_optimal_frequencies(situation),
            'key_concepts': [
                'Balanced ranges prevent exploitation',
                'Mix value and bluffs at optimal ratio',
                'Consider blockers and equity',
                'Adjust based on opponent tendencies'
            ]
        }
    
    def get_exploitative_adjustments(self, situation: Dict) -> List[str]:
        """Get exploitative adjustments to GTO strategy"""
        return [
            "Against tight players: increase bluff frequency",
            "Against loose players: tighten value range",
            "Against aggressive players: trap more with strong hands",
            "Against passive players: bet thinner for value",
            "Adjust sizing based on opponent's calling frequency",
            "Exploit obvious imbalances in opponent's strategy"
        ]
    
    def get_tournament_stage(self, game_state: Dict) -> str:
        """Determine tournament stage based on game state"""
        tournament_stage = game_state.get('tournament_stage', 'early')
        stack_size = game_state.get('stack_size', 100)
        big_blind = game_state.get('big_blind', 1)
        
        effective_stack = stack_size / big_blind
        
        if effective_stack > 100:
            return 'early'
        elif effective_stack > 30:
            return 'middle'
        else:
            return 'late'
    
    def review_session(self, session_data: Dict) -> Dict:
        """Review and analyze a poker session"""
        hands_played = session_data.get('hands_played', 100)
        profit_loss = session_data.get('profit_loss', 0)
        decisions = session_data.get('decisions', [])
        game_type = session_data.get('game_type', 'cash')
        stake_level = session_data.get('stake_level', 100)
        
        # Calculate session statistics
        bb_per_100 = (profit_loss / stake_level) * 100 * (100 / hands_played) if hands_played > 0 else 0
        
        # Analyze decision quality
        decision_analysis = self.analyze_decisions(decisions)
        
        # Generate recommendations
        recommendations = self.generate_session_recommendations(session_data, decision_analysis)
        
        return {
            'session_stats': {
                'hands_played': hands_played,
                'profit_loss': profit_loss,
                'bb_per_100': round(bb_per_100, 2),
                'hourly_rate': self.calculate_hourly_rate(session_data)
            },
            'decision_analysis': decision_analysis,
            'recommendations': recommendations,
            'areas_for_improvement': self.identify_improvement_areas(decision_analysis),
            'session_rating': self.rate_session(bb_per_100, decision_analysis)
        }
    
    def analyze_decisions(self, decisions: List[Dict]) -> Dict:
        """Analyze the quality of decisions made"""
        if not decisions:
            return {'quality': 'insufficient_data', 'score': 0}
        
        total_score = 0
        categories = {'preflop': 0, 'postflop': 0, 'sizing': 0, 'timing': 0}
        
        for decision in decisions:
            # Simplified decision scoring
            decision_score = decision.get('ev_impact', 0)
            total_score += decision_score
            
            category = decision.get('category', 'general')
            if category in categories:
                categories[category] += decision_score
        
        avg_score = total_score / len(decisions) if decisions else 0
        
        return {
            'overall_score': round(avg_score, 2),
            'category_breakdown': categories,
            'decision_count': len(decisions),
            'quality': self.categorize_decision_quality(avg_score)
        }
    
    def categorize_decision_quality(self, score: float) -> str:
        """Categorize decision quality based on score"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'average'
        elif score >= 0.2:
            return 'below_average'
        else:
            return 'poor'
    
    def generate_session_recommendations(self, session_data: Dict, decision_analysis: Dict) -> List[str]:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        profit_loss = session_data.get('profit_loss', 0)
        quality = decision_analysis.get('quality', 'average')
        
        if profit_loss < 0:
            recommendations.append("Review hand histories for major losses")
            recommendations.append("Consider taking a break if on tilt")
        
        if quality in ['poor', 'below_average']:
            recommendations.append("Focus on fundamental strategy review")
            recommendations.append("Consider working with a coach")
        
        recommendations.extend([
            "Continue tracking session data",
            "Review spots where you felt uncertain",
            "Practice difficult scenarios in study time"
        ])
        
        return recommendations
    
    def identify_improvement_areas(self, decision_analysis: Dict) -> List[str]:
        """Identify specific areas for improvement"""
        categories = decision_analysis.get('category_breakdown', {})
        improvement_areas = []
        
        for category, score in categories.items():
            if score < 0.5:  # Below average performance
                improvement_areas.append(f"{category} decision making")
        
        if not improvement_areas:
            improvement_areas.append("Continue current study routine")
        
        return improvement_areas
    
    def calculate_hourly_rate(self, session_data: Dict) -> float:
        """Calculate hourly rate from session"""
        profit_loss = session_data.get('profit_loss', 0)
        hands_played = session_data.get('hands_played', 100)
        
        # Assume 30 hands per hour for live, 80 for online
        hands_per_hour = 80  # Default to online
        hours_played = hands_played / hands_per_hour
        
        return round(profit_loss / hours_played, 2) if hours_played > 0 else 0
    
    def rate_session(self, bb_per_100: float, decision_analysis: Dict) -> str:
        """Rate overall session performance"""
        quality = decision_analysis.get('quality', 'average')
        
        if bb_per_100 > 10 and quality in ['excellent', 'good']:
            return 'A+'
        elif bb_per_100 > 5 and quality in ['good', 'average']:
            return 'A'
        elif bb_per_100 > 0 and quality == 'average':
            return 'B'
        elif bb_per_100 > -5:
            return 'C'
        else:
            return 'D'