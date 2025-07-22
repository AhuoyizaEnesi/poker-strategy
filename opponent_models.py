import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class PlayerStats:
    """Player statistics for modeling"""
    vpip: float = 20.0  # Voluntarily Put $ In Pot
    pfr: float = 15.0   # Pre-Flop Raise
    aggression: float = 2.5  # Aggression Factor
    c_bet: float = 75.0  # Continuation Bet %
    fold_to_3bet: float = 65.0  # Fold to 3-bet %
    steal_attempt: float = 30.0  # Steal Attempt %
    fold_to_steal: float = 80.0  # Fold to Steal %
    check_raise: float = 8.0  # Check-Raise %
    wtsd: float = 25.0  # Went To Showdown %
    w_sd: float = 50.0  # Won $ at Showdown %

class PlayerType:
    """Different player archetypes"""
    
    TIGHT_AGGRESSIVE = PlayerStats(
        vpip=18, pfr=15, aggression=3.0, c_bet=80, fold_to_3bet=60,
        steal_attempt=25, fold_to_steal=85, check_raise=6, wtsd=22, w_sd=55
    )
    
    LOOSE_AGGRESSIVE = PlayerStats(
        vpip=35, pfr=25, aggression=3.5, c_bet=85, fold_to_3bet=45,
        steal_attempt=45, fold_to_steal=70, check_raise=12, wtsd=30, w_sd=48
    )
    
    TIGHT_PASSIVE = PlayerStats(
        vpip=15, pfr=8, aggression=1.5, c_bet=65, fold_to_3bet=75,
        steal_attempt=15, fold_to_steal=90, check_raise=4, wtsd=20, w_sd=52
    )
    
    LOOSE_PASSIVE = PlayerStats(
        vpip=45, pfr=12, aggression=1.8, c_bet=60, fold_to_3bet=70,
        steal_attempt=20, fold_to_steal=75, check_raise=5, wtsd=35, w_sd=45
    )
    
    MANIAC = PlayerStats(
        vpip=60, pfr=40, aggression=5.0, c_bet=90, fold_to_3bet=30,
        steal_attempt=60, fold_to_steal=50, check_raise=15, wtsd=40, w_sd=42
    )
    
    ROCK = PlayerStats(
        vpip=8, pfr=6, aggression=2.0, c_bet=70, fold_to_3bet=80,
        steal_attempt=10, fold_to_steal=95, check_raise=3, wtsd=15, w_sd=58
    )
    
    UNKNOWN = PlayerStats()  # Default/average stats

class OpponentModeler:
    """Models opponent behavior and tendencies"""
    
    def __init__(self):
        self.player_types = {
            'tight_aggressive': PlayerType.TIGHT_AGGRESSIVE,
            'loose_aggressive': PlayerType.LOOSE_AGGRESSIVE,
            'tight_passive': PlayerType.TIGHT_PASSIVE,
            'loose_passive': PlayerType.LOOSE_PASSIVE,
            'maniac': PlayerType.MANIAC,
            'rock': PlayerType.ROCK,
            'unknown': PlayerType.UNKNOWN
        }
    
    def analyze_player_type(self, player_data: Dict) -> Dict:
        """Analyze player type based on statistics"""
        vpip = player_data.get('vpip', 20)
        pfr = player_data.get('pfr', 15)
        aggression = player_data.get('aggression', 2.5)
        c_bet = player_data.get('c_bet', 75)
        fold_to_3bet = player_data.get('fold_to_3bet', 65)
        
        # Classify player type
        player_type = self.classify_player_type(vpip, pfr, aggression)
        
        # Get detailed analysis
        analysis = self.get_detailed_analysis(player_data, player_type)
        
        # Generate exploitative strategy
        exploit_strategy = self.generate_exploit_strategy(player_type, player_data)
        
        return {
            'player_type': player_type,
            'classification_confidence': self.calculate_classification_confidence(player_data),
            'key_stats': {
                'vpip': vpip,
                'pfr': pfr,
                'aggression': aggression,
                'pfr_gap': vpip - pfr
            },
            'tendencies': analysis,
            'exploit_strategy': exploit_strategy,
            'range_estimates': self.estimate_ranges(player_type),
            'situational_adjustments': self.get_situational_adjustments(player_type)
        }
    
    def classify_player_type(self, vpip: float, pfr: float, aggression: float) -> str:
        """Classify player type based on key statistics"""
        pfr_gap = vpip - pfr
        
        # Determine tightness/looseness
        if vpip < 18:
            tightness = 'tight'
        elif vpip > 30:
            tightness = 'loose'
        else:
            tightness = 'balanced'
        
        # Determine aggression level
        if aggression > 3.0 and pfr > 18:
            aggression_level = 'aggressive'
        elif aggression < 2.0 or pfr_gap > 15:
            aggression_level = 'passive'
        else:
            aggression_level = 'balanced'
        
        # Special cases
        if vpip > 50 and aggression > 4.0:
            return 'maniac'
        elif vpip < 12 and pfr < 8:
            return 'rock'
        elif tightness == 'balanced' and aggression_level == 'balanced':
            return 'unknown'
        
        return f"{tightness}_{aggression_level}"
    
    def calculate_classification_confidence(self, player_data: Dict) -> float:
        """Calculate confidence in player type classification"""
        sample_size = player_data.get('hands_observed', 100)
        
        # Base confidence on sample size
        if sample_size < 50:
            base_confidence = 0.3
        elif sample_size < 100:
            base_confidence = 0.6
        elif sample_size < 200:
            base_confidence = 0.8
        else:
            base_confidence = 0.95
        
        # Adjust based on stat consistency
        vpip = player_data.get('vpip', 20)
        pfr = player_data.get('pfr', 15)
        aggression = player_data.get('aggression', 2.5)
        
        # Check for statistical consistency
        consistency_bonus = 0
        if 0 <= pfr <= vpip:  # PFR should never exceed VPIP
            consistency_bonus += 0.1
        if vpip < 50 and aggression < 6:  # Reasonable ranges
            consistency_bonus += 0.1
        
        return min(0.95, base_confidence + consistency_bonus)
    
    def get_detailed_analysis(self, player_data: Dict, player_type: str) -> Dict:
        """Get detailed behavioral analysis"""
        tendencies = {
            'preflop': self.analyze_preflop_tendencies(player_data, player_type),
            'postflop': self.analyze_postflop_tendencies(player_data, player_type),
            'bluffing': self.analyze_bluffing_tendencies(player_data, player_type),
            'value_betting': self.analyze_value_betting(player_data, player_type),
            'positional': self.analyze_positional_play(player_data, player_type)
        }
        
        return tendencies
    
    def analyze_preflop_tendencies(self, player_data: Dict, player_type: str) -> Dict:
        """Analyze preflop tendencies"""
        vpip = player_data.get('vpip', 20)
        pfr = player_data.get('pfr', 15)
        fold_to_3bet = player_data.get('fold_to_3bet', 65)
        steal_attempt = player_data.get('steal_attempt', 30)
        
        return {
            'range_width': 'wide' if vpip > 25 else 'tight' if vpip < 18 else 'standard',
            'aggression_level': 'high' if pfr > 18 else 'low' if pfr < 12 else 'medium',
            'limping_frequency': 'high' if (vpip - pfr) > 15 else 'low',
            '3bet_defense': 'strong' if fold_to_3bet < 60 else 'weak' if fold_to_3bet > 70 else 'standard',
            'steal_frequency': 'high' if steal_attempt > 35 else 'low' if steal_attempt < 25 else 'standard',
            'key_exploits': self.get_preflop_exploits(vpip, pfr, fold_to_3bet)
        }
    
    def get_preflop_exploits(self, vpip: float, pfr: float, fold_to_3bet: float) -> List[str]:
        """Get preflop exploitation strategies"""
        exploits = []
        
        if vpip > 30:
            exploits.append("3bet light for value - they play too many hands")
        if pfr < 10:
            exploits.append("Steal blinds more frequently - low aggression")
        if fold_to_3bet > 75:
            exploits.append("3bet bluff frequently - high fold rate")
        if (vpip - pfr) > 20:
            exploits.append("Raise limpers for isolation")
        
        return exploits
    
    def analyze_postflop_tendencies(self, player_data: Dict, player_type: str) -> Dict:
        """Analyze postflop tendencies"""
        c_bet = player_data.get('c_bet', 75)
        check_raise = player_data.get('check_raise', 8)
        aggression = player_data.get('aggression', 2.5)
        wtsd = player_data.get('wtsd', 25)
        
        return {
            'c_bet_frequency': 'high' if c_bet > 80 else 'low' if c_bet < 65 else 'standard',
            'check_raise_tendency': 'high' if check_raise > 12 else 'low' if check_raise < 6 else 'medium',
            'overall_aggression': 'high' if aggression > 3.0 else 'low' if aggression < 2.0 else 'balanced',
            'showdown_tendency': 'high' if wtsd > 28 else 'low' if wtsd < 22 else 'average',
            'likely_betting_patterns': self.predict_betting_patterns(c_bet, aggression, check_raise)
        }
    
    def predict_betting_patterns(self, c_bet: float, aggression: float, check_raise: float) -> List[str]:
        """Predict likely betting patterns"""
        patterns = []
        
        if c_bet > 80:
            patterns.append("High c-bet frequency on all board textures")
        if aggression > 3.5:
            patterns.append("Multiple barrel bluffs")
        if check_raise > 10:
            patterns.append("Frequent check-raises with draws and strong hands")
        if aggression < 2.0:
            patterns.append("Mostly check-call lines with marginal hands")
        
        return patterns
    
    def analyze_bluffing_tendencies(self, player_data: Dict, player_type: str) -> Dict:
        """Analyze bluffing behavior"""
        aggression = player_data.get('aggression', 2.5)
        w_sd = player_data.get('w_sd', 50)
        c_bet = player_data.get('c_bet', 75)
        
        # Estimate bluff frequency based on aggression and showdown success
        bluff_frequency = 'high' if aggression > 3.0 and w_sd < 48 else 'low' if aggression < 2.0 else 'medium'
        
        return {
            'frequency': bluff_frequency,
            'likely_spots': self.identify_bluff_spots(player_type, aggression),
            'sizing_tells': self.get_bluff_sizing_patterns(player_type),
            'timing_tells': self.get_bluff_timing_patterns(player_type),
            'defense_strategy': self.get_bluff_defense_strategy(bluff_frequency)
        }
    
    def identify_bluff_spots(self, player_type: str, aggression: float) -> List[str]:
        """Identify likely bluffing spots"""
        spots = []
        
        if 'aggressive' in player_type:
            spots.extend(['missed c-bets', 'river bluffs', 'squeeze plays'])
        if 'loose' in player_type:
            spots.extend(['draw bluffs', 'position bluffs'])
        if player_type == 'maniac':
            spots.extend(['any two cards', 'constant pressure'])
        
        return spots if spots else ['standard bluff spots']
    
    def get_bluff_sizing_patterns(self, player_type: str) -> List[str]:
        """Get bluff sizing patterns for player type"""
        if 'aggressive' in player_type:
            return ['Large sizing with bluffs', 'Overbet bluffs']
        elif 'passive' in player_type:
            return ['Small sizing when bluffing', 'Minimum bet bluffs']
        else:
            return ['Standard sizing']
    
    def get_bluff_timing_patterns(self, player_type: str) -> List[str]:
        """Get bluff timing patterns"""
        if player_type == 'maniac':
            return ['Instant aggressive action', 'No hesitation']
        elif 'tight' in player_type:
            return ['Longer tank times when bluffing', 'Uncomfortable body language']
        else:
            return ['Standard timing']
    
    def get_bluff_defense_strategy(self, bluff_frequency: str) -> List[str]:
        """Get strategy to defend against bluffs"""
        if bluff_frequency == 'high':
            return ['Call down lighter', 'Look up with bluff catchers', 'Avoid folding medium strength']
        elif bluff_frequency == 'low':
            return ['Fold to aggression', 'Need strong hands to continue', 'Respect their bets']
        else:
            return ['Balanced defense', 'Hand-dependent decisions']
    
    def analyze_value_betting(self, player_data: Dict, player_type: str) -> Dict:
        """Analyze value betting tendencies"""
        w_sd = player_data.get('w_sd', 50)
        aggression = player_data.get('aggression', 2.5)
        
        return {
            'thickness': 'thin' if w_sd > 52 else 'thick' if w_sd < 48 else 'balanced',
            'sizing_strategy': self.get_value_sizing_strategy(player_type),
            'likely_value_hands': self.estimate_value_range(player_type),
            'exploitation_method': self.get_value_exploitation(player_type, w_sd)
        }
    
    def get_value_sizing_strategy(self, player_type: str) -> str:
        """Get value betting sizing strategy"""
        if 'aggressive' in player_type:
            return 'Large sizing for maximum value'
        elif 'passive' in player_type:
            return 'Small sizing to induce calls'
        else:
            return 'Balanced sizing approach'
    
    def estimate_value_range(self, player_type: str) -> List[str]:
        """Estimate value betting range"""
        if 'tight' in player_type:
            return ['Top pair good kicker+', 'Strong two pair+', 'Sets and better']
        elif 'loose' in player_type:
            return ['Any pair', 'Weak top pair', 'Marginal hands']
        else:
            return ['Top pair decent kicker', 'Two pair', 'Strong hands']
    
    def get_value_exploitation(self, player_type: str, w_sd: float) -> List[str]:
        """Get value betting exploitation strategies"""
        exploits = []
        
        if 'passive' in player_type:
            exploits.append('Value bet thinly - they call too much')
        if w_sd < 45:
            exploits.append('Avoid bluff catchers - they value bet too wide')
        if 'tight' in player_type and w_sd > 55:
            exploits.append('Give action to their value bets - very strong range')
        
        return exploits if exploits else ['Standard value betting approach']
    
    def analyze_positional_play(self, player_data: Dict, player_type: str) -> Dict:
        """Analyze positional awareness"""
        steal_attempt = player_data.get('steal_attempt', 30)
        fold_to_steal = player_data.get('fold_to_steal', 80)
        
        return {
            'positional_awareness': self.assess_positional_awareness(player_type, steal_attempt),
            'button_play': self.analyze_button_play(steal_attempt),
            'blind_defense': self.analyze_blind_defense(fold_to_steal),
            'exploitation_strategy': self.get_positional_exploits(steal_attempt, fold_to_steal)
        }
    
    def assess_positional_awareness(self, player_type: str, steal_attempt: float) -> str:
        """Assess positional awareness level"""
        if 'tight' in player_type and steal_attempt < 20:
            return 'Poor - too tight in position'
        elif steal_attempt > 40:
            return 'Good - takes advantage of position'
        else:
            return 'Average - standard positional play'
    
    def analyze_button_play(self, steal_attempt: float) -> str:
        """Analyze button play style"""
        if steal_attempt > 45:
            return 'Very aggressive - wide stealing range'
        elif steal_attempt < 25:
            return 'Too tight - missing value'
        else:
            return 'Balanced approach'
    
    def analyze_blind_defense(self, fold_to_steal: float) -> str:
        """Analyze blind defense tendencies"""
        if fold_to_steal > 85:
            return 'Over-folding - can be exploited'
        elif fold_to_steal < 70:
            return 'Over-defending - calling too light'
        else:
            return 'Balanced defense'
    
    def get_positional_exploits(self, steal_attempt: float, fold_to_steal: float) -> List[str]:
        """Get positional exploitation strategies"""
        exploits = []
        
        if steal_attempt < 25:
            exploits.append('Steal more frequently - they are too tight')
        if fold_to_steal > 85:
            exploits.append('Increase steal frequency - high fold rate')
        if fold_to_steal < 70:
            exploits.append('Tighten stealing range - they defend too much')
        
        return exploits if exploits else ['Standard positional play']
    
    def generate_exploit_strategy(self, player_type: str, player_data: Dict) -> Dict:
        """Generate comprehensive exploitation strategy"""
        
        exploit_strategy = {
            'primary_exploits': self.get_primary_exploits(player_type),
            'betting_adjustments': self.get_betting_adjustments(player_type, player_data),
            'hand_selection': self.get_hand_selection_adjustments(player_type),
            'sizing_adjustments': self.get_sizing_adjustments(player_type),
            'position_specific': self.get_position_specific_adjustments(player_type),
            'session_notes': self.generate_session_notes(player_type, player_data)
        }
        
        return exploit_strategy
    
    def get_primary_exploits(self, player_type: str) -> List[str]:
        """Get primary exploitation strategies"""
        exploits = {
            'tight_aggressive': [
                'Play more speculative hands for implied odds',
                'Avoid bluffing - they fold appropriately',
                'Value bet thinly - they pay off strong hands'
            ],
            'loose_aggressive': [
                'Tighten up and let them bluff into you',
                'Call down lighter with bluff catchers',
                '3bet for value more frequently'
            ],
            'tight_passive': [
                'Steal blinds frequently',
                'Bet for value with marginal hands',
                'Avoid bluffing - they call with weak hands'
            ],
            'loose_passive': [
                'Value bet extremely thin',
                'Avoid bluffing completely',
                'Build pots with strong hands'
            ],
            'maniac': [
                'Play extremely tight and let them hang themselves',
                'Call with strong hands and let them bluff',
                'Avoid getting involved without premium hands'
            ],
            'rock': [
                'Steal constantly',
                'Fold to any aggression',
                'Only give action with very strong hands'
            ]
        }
        
        return exploits.get(player_type, ['Balanced approach until more data'])
    
    def get_betting_adjustments(self, player_type: str, player_data: Dict) -> Dict:
        """Get betting adjustments for player type"""
        fold_to_3bet = player_data.get('fold_to_3bet', 65)
        c_bet = player_data.get('c_bet', 75)
        
        adjustments = {}
        
        if 'passive' in player_type:
            adjustments['value_bet_frequency'] = 'increase'
            adjustments['bluff_frequency'] = 'decrease'
        elif 'aggressive' in player_type:
            adjustments['value_bet_frequency'] = 'standard'
            adjustments['bluff_frequency'] = 'decrease'
        
        if fold_to_3bet > 70:
            adjustments['3bet_bluff_frequency'] = 'increase'
        elif fold_to_3bet < 50:
            adjustments['3bet_bluff_frequency'] = 'decrease'
        
        return adjustments
    
    def get_hand_selection_adjustments(self, player_type: str) -> Dict:
        """Get hand selection adjustments"""
        if 'tight' in player_type:
            return {
                'speculative_hands': 'increase - good implied odds',
                'marginal_hands': 'decrease - they have strong ranges',
                'bluffing_hands': 'decrease - they fold appropriately'
            }
        elif 'loose' in player_type:
            return {
                'value_hands': 'increase - they pay off',
                'bluff_catchers': 'increase - they bluff more',
                'marginal_hands': 'decrease - avoid tough spots'
            }
        else:
            return {'approach': 'balanced until more data'}
    
    def get_sizing_adjustments(self, player_type: str) -> Dict:
        """Get sizing adjustments for different player types"""
        sizing = {}
        
        if 'passive' in player_type:
            sizing['value_bets'] = 'larger - they call anyway'
            sizing['bluffs'] = 'smaller - save money'
        elif 'aggressive' in player_type:
            sizing['value_bets'] = 'standard - they might raise'
            sizing['bluffs'] = 'avoid - they call/raise light'
        
        return sizing
    
    def get_position_specific_adjustments(self, player_type: str) -> Dict:
        """Get position-specific adjustments"""
        return {
            'in_position': self.get_ip_adjustments(player_type),
            'out_of_position': self.get_oop_adjustments(player_type),
            'blind_vs_blind': self.get_bvb_adjustments(player_type)
        }
    
    def get_ip_adjustments(self, player_type: str) -> List[str]:
        """Get in-position adjustments"""
        if 'passive' in player_type:
            return ['Bet more thin for value', 'Control pot sizes']
        elif 'aggressive' in player_type:
            return ['Be more defensive', 'Let them lead into you']
        else:
            return ['Standard positional play']
    
    def get_oop_adjustments(self, player_type: str) -> List[str]:
        """Get out-of-position adjustments"""
        if 'aggressive' in player_type:
            return ['Check-call more', 'Avoid leading into them']
        elif 'passive' in player_type:
            return ['Lead more for value', 'Take control of betting']
        else:
            return ['Balanced approach']
    
    def get_bvb_adjustments(self, player_type: str) -> List[str]:
        """Get blind vs blind adjustments"""
        if 'tight' in player_type:
            return ['Increase stealing frequency', 'Defend tighter']
        elif 'loose' in player_type:
            return ['Steal tighter', 'Defend wider with position']
        else:
            return ['Standard blind vs blind play']
    
    def generate_session_notes(self, player_type: str, player_data: Dict) -> List[str]:
        """Generate session notes for the player"""
        notes = [f"Player type: {player_type.replace('_', ' ').title()}"]
        
        vpip = player_data.get('vpip', 20)
        pfr = player_data.get('pfr', 15)
        aggression = player_data.get('aggression', 2.5)
        
        notes.append(f"VPIP: {vpip}%, PFR: {pfr}%, Aggression: {aggression}")
        
        # Add key exploits
        primary_exploits = self.get_primary_exploits(player_type)
        if primary_exploits:
            notes.append(f"Key exploit: {primary_exploits[0]}")
        
        # Add warning signs
        if player_type == 'maniac':
            notes.append("WARNING: Extremely aggressive - avoid marginal spots")
        elif 'tight_aggressive' in player_type:
            notes.append("CAUTION: Strong player - respect their aggression")
        
        return notes
    
    def estimate_ranges(self, player_type: str) -> Dict:
        """Estimate hand ranges for different actions"""
        ranges = {
            'tight_aggressive': {
                'open_utg': '99+, AJs+, KQs, AQo+',
                'open_btn': '22+, A2s+, K8s+, Q9s+, J9s+, T8s+, 98s, 87s, ATo+, KJo+, QJo',
                '3bet': 'JJ+, AKs, AKo',
                'c_bet_bluff': 'A high, gutshots, flush draws'
            },
            'loose_aggressive': {
                'open_utg': '77+, A9s+, KTs+, QJs, AJo+, KQo',
                'open_btn': '22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 95s+, 85s+, 75s+, 65s, A2o+, K8o+, Q9o+, J9o+, T9o',
                '3bet': '99+, A9s+, KQs, A5s-A2s, AJo+, KQo',
                'c_bet_bluff': 'Any two cards, draws, overcards'
            },
            'tight_passive': {
                'open_utg': 'TT+, AQs+, AKo',
                'open_btn': '88+, A9s+, KTs+, QJs, ATo+, KQo',
                '3bet': 'QQ+, AKs, AKo',
                'value_bet': 'Top pair good kicker+'
            },
            'loose_passive': {
                'open_utg': '66+, A8s+, K9s+, QTs+, ATo+, KJo+',
                'open_btn': '22+, A2s+, K6s+, Q8s+, J8s+, T7s+, 97s+, 87s, 76s, A2o+, K9o+, Q9o+, J9o+',
                '3bet': 'JJ+, AQs+, AKo',
                'calling_range': 'Any pair, any ace, suited connectors'
            }
        }
        
        return ranges.get(player_type, {
            'open_utg': 'Standard tight range',
            'open_btn': 'Standard wide range',
            '3bet': 'Balanced 3bet range'
        })
    
    def get_situational_adjustments(self, player_type: str) -> Dict:
        """Get situational adjustments for different scenarios"""
        return {
            'short_stack': self.get_short_stack_adjustments(player_type),
            'deep_stack': self.get_deep_stack_adjustments(player_type),
            'heads_up': self.get_heads_up_adjustments(player_type),
            'multi_way': self.get_multiway_adjustments(player_type),
            'tournament': self.get_tournament_adjustments(player_type)
        }
    
    def get_short_stack_adjustments(self, player_type: str) -> List[str]:
        """Get short stack adjustments"""
        if 'tight' in player_type:
            return ['They play even tighter', 'Pressure with any two cards']
        elif 'loose' in player_type:
            return ['They go all-in light', 'Call with strong hands only']
        else:
            return ['Standard short stack play']
    
    def get_deep_stack_adjustments(self, player_type: str) -> List[str]:
        """Get deep stack adjustments"""
        if 'passive' in player_type:
            return ['More calling, less folding', 'Value bet extremely thin']
        elif 'aggressive' in player_type:
            return ['More complex lines', 'Be more careful with bluffs']
        else:
            return ['Standard deep stack play']
    
    def get_heads_up_adjustments(self, player_type: str) -> List[str]:
        """Get heads-up adjustments"""
        if 'tight' in player_type:
            return ['They struggle heads-up', 'Pressure constantly']
        elif player_type == 'maniac':
            return ['Play extremely tight', 'Let them bluff off chips']
        else:
            return ['Balanced heads-up play']
    
    def get_multiway_adjustments(self, player_type: str) -> List[str]:
        """Get multiway pot adjustments"""
        if 'loose' in player_type:
            return ['They call light multiway', 'Value bet thinner']
        elif 'tight' in player_type:
            return ['They fold too much', 'Bluff more in multiway pots']
        else:
            return ['Standard multiway play']
    
    def get_tournament_adjustments(self, player_type: str) -> List[str]:
        """Get tournament-specific adjustments"""
        if 'tight' in player_type:
            return ['Extra tight near bubble', 'Pressure with stack size']
        elif 'loose' in player_type:
            return ['Gambling tendency increases', 'Avoid coin flips with them']
        else:
            return ['Standard tournament adjustments']
    
    def analyze_opponents(self, game_state: Dict, simulation_results: Dict) -> Dict:
        """Analyze multiple opponents in current game state"""
        opponents = game_state.get('opponents', 5)
        position = game_state.get('position', 'BTN')
        
        # Generate opponent profiles for the table
        opponent_profiles = []
        for i in range(opponents):
            # Simulate different opponent types
            opponent_type = random.choice(['tight_aggressive', 'loose_aggressive', 'tight_passive', 'loose_passive', 'unknown'])
            
            profile = {
                'seat': i + 1,
                'type': opponent_type,
                'estimated_stats': self.player_types[opponent_type],
                'relative_position': self.get_relative_position(position, i, opponents),
                'threat_level': self.assess_threat_level(opponent_type),
                'key_adjustments': self.get_primary_exploits(opponent_type)[:2]
            }
            opponent_profiles.append(profile)
        
        return {
            'table_dynamics': self.analyze_table_dynamics(opponent_profiles),
            'opponent_profiles': opponent_profiles,
            'overall_strategy': self.get_table_strategy(opponent_profiles, game_state),
            'position_analysis': self.analyze_position_dynamics(opponent_profiles, position)
        }
    
    def get_relative_position(self, hero_position: str, opponent_index: int, total_opponents: int) -> str:
        """Get relative position of opponent to hero"""
        positions = ['UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN', 'SB', 'BB']
        hero_idx = positions.index(hero_position) if hero_position in positions else 5
        
        opponent_idx = (hero_idx + opponent_index + 1) % len(positions)
        relative_pos = (opponent_idx - hero_idx) % len(positions)
        
        if relative_pos <= 2:
            return 'before_hero'
        elif relative_pos >= 6:
            return 'after_hero'
        else:
            return 'across_table'
    
    def assess_threat_level(self, opponent_type: str) -> str:
        """Assess threat level of opponent type"""
        threat_levels = {
            'tight_aggressive': 'high',
            'loose_aggressive': 'medium',
            'tight_passive': 'low',
            'loose_passive': 'low',
            'maniac': 'high',
            'rock': 'very_low',
            'unknown': 'medium'
        }
        return threat_levels.get(opponent_type, 'medium')
    
    def analyze_table_dynamics(self, opponent_profiles: List[Dict]) -> Dict:
        """Analyze overall table dynamics"""
        threat_levels = [profile['threat_level'] for profile in opponent_profiles]
        opponent_types = [profile['type'] for profile in opponent_profiles]
        
        high_threat_count = sum(1 for level in threat_levels if level in ['high', 'very_high'])
        aggressive_count = sum(1 for opp_type in opponent_types if 'aggressive' in opp_type)
        passive_count = sum(1 for opp_type in opponent_types if 'passive' in opp_type)
        
        if high_threat_count >= 3:
            table_type = 'tough'
        elif aggressive_count >= 4:
            table_type = 'aggressive'
        elif passive_count >= 4:
            table_type = 'passive'
        else:
            table_type = 'mixed'
        
        return {
            'table_type': table_type,
            'aggression_level': 'high' if aggressive_count > passive_count else 'low',
            'difficulty': 'high' if high_threat_count >= 2 else 'medium' if high_threat_count == 1 else 'low',
            'recommended_approach': self.get_table_approach(table_type)
        }
    
    def get_table_approach(self, table_type: str) -> str:
        """Get recommended approach for table type"""
        approaches = {
            'tough': 'Play tight and pick spots carefully',
            'aggressive': 'Play tighter and let them battle each other',
            'passive': 'Value bet thin and avoid bluffing',
            'mixed': 'Adjust to individual opponents'
        }
        return approaches.get(table_type, 'Balanced approach')
    
    def get_table_strategy(self, opponent_profiles: List[Dict], game_state: Dict) -> List[str]:
        """Get overall table strategy"""
        strategy = []
        
        # Analyze table composition
        aggressive_opponents = sum(1 for p in opponent_profiles if 'aggressive' in p['type'])
        passive_opponents = sum(1 for p in opponent_profiles if 'passive' in p['type'])
        
        if aggressive_opponents > passive_opponents:
            strategy.append("Table is aggressive - play tighter and more defensively")
            strategy.append("Look for spots to trap with strong hands")
        elif passive_opponents > aggressive_opponents:
            strategy.append("Table is passive - value bet thinner and avoid bluffing")
            strategy.append("Build pots with strong hands")
        else:
            strategy.append("Mixed table - adjust to individual opponents")
        
        # Position-based strategy
        position = game_state.get('position', 'BTN')
        if position in ['BTN', 'CO']:
            strategy.append("Use position advantage to control pot sizes")
        elif position in ['SB', 'BB']:
            strategy.append("Play tighter due to positional disadvantage")
        
        return strategy
    
    def analyze_position_dynamics(self, opponent_profiles: List[Dict], hero_position: str) -> Dict:
        """Analyze positional dynamics at the table"""
        before_hero = [p for p in opponent_profiles if p['relative_position'] == 'before_hero']
        after_hero = [p for p in opponent_profiles if p['relative_position'] == 'after_hero']