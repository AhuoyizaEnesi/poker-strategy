import itertools
import random
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Optional

class Card:
    """Represents a playing card"""
    
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
    
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.value = self.RANK_VALUES[rank]
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))

class Deck:
    """Standard 52-card deck"""
    
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        """Reset deck to full 52 cards"""
        self.cards = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                self.cards.append(Card(rank, suit))
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck"""
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int = 1) -> List[Card]:
        """Deal specified number of cards"""
        if len(self.cards) < num_cards:
            raise ValueError("Not enough cards in deck")
        
        dealt = []
        for _ in range(num_cards):
            dealt.append(self.cards.pop())
        return dealt
    
    def remove_cards(self, cards_to_remove: List[str]):
        """Remove specific cards from deck (for known cards)"""
        cards_to_remove_objs = [self.parse_card(card) for card in cards_to_remove]
        self.cards = [card for card in self.cards if card not in cards_to_remove_objs]
    
    @staticmethod
    def parse_card(card_str: str) -> Card:
        """Parse card string like 'As' into Card object"""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card format: {card_str}")
        rank, suit = card_str[0], card_str[1].lower()
        return Card(rank, suit)

class HandEvaluator:
    """Evaluates poker hand strength"""
    
    HAND_RANKINGS = {
        'high_card': 1,
        'pair': 2,
        'two_pair': 3,
        'three_of_a_kind': 4,
        'straight': 5,
        'flush': 6,
        'full_house': 7,
        'four_of_a_kind': 8,
        'straight_flush': 9,
        'royal_flush': 10
    }
    
    def evaluate_hand(self, cards: List[str]) -> Dict:
        """Evaluate poker hand and return ranking info"""
        card_objects = [Deck.parse_card(card) for card in cards]
        
        if len(card_objects) < 5:
            raise ValueError("Need at least 5 cards to evaluate")
        
        # Get best 5-card combination
        best_hand = self.get_best_five_card_hand(card_objects)
        
        # Evaluate the hand
        hand_type, strength = self.classify_hand(best_hand)
        
        return {
            'hand_type': hand_type,
            'strength': strength,
            'ranking': self.HAND_RANKINGS[hand_type],
            'cards': [str(card) for card in best_hand],
            'description': self.get_hand_description(hand_type, best_hand)
        }
    
    def get_best_five_card_hand(self, cards: List[Card]) -> List[Card]:
        """Get best 5-card hand from available cards"""
        if len(cards) == 5:
            return cards
        
        best_hand = None
        best_strength = 0
        
        # Try all 5-card combinations
        for combo in itertools.combinations(cards, 5):
            hand_type, strength = self.classify_hand(list(combo))
            total_strength = self.HAND_RANKINGS[hand_type] * 1000000 + strength
            
            if total_strength > best_strength:
                best_strength = total_strength
                best_hand = list(combo)
        
        return best_hand
    
    def classify_hand(self, cards: List[Card]) -> Tuple[str, int]:
        """Classify hand type and calculate strength"""
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)
        
        is_flush = len(set(suits)) == 1
        is_straight = self.is_straight(ranks)
        
        # Sort by count, then by rank value
        sorted_ranks = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Check for each hand type
        if is_straight and is_flush:
            if min(ranks) == 8:  # T-A straight
                return 'royal_flush', max(ranks)
            return 'straight_flush', max(ranks)
        
        elif sorted_ranks[0][1] == 4:
            return 'four_of_a_kind', sorted_ranks[0][0] * 100 + sorted_ranks[1][0]
        
        elif sorted_ranks[0][1] == 3 and sorted_ranks[1][1] == 2:
            return 'full_house', sorted_ranks[0][0] * 100 + sorted_ranks[1][0]
        
        elif is_flush:
            return 'flush', sum(rank * (13 ** i) for i, rank in enumerate(sorted(ranks, reverse=True)))
        
        elif is_straight:
            return 'straight', max(ranks)
        
        elif sorted_ranks[0][1] == 3:
            kickers = [rank for rank, count in sorted_ranks[1:]]
            return 'three_of_a_kind', sorted_ranks[0][0] * 10000 + sum(k * (13 ** i) for i, k in enumerate(kickers))
        
        elif sorted_ranks[0][1] == 2 and sorted_ranks[1][1] == 2:
            pairs = [rank for rank, count in sorted_ranks[:2]]
            kicker = sorted_ranks[2][0]
            return 'two_pair', max(pairs) * 1000 + min(pairs) * 100 + kicker
        
        elif sorted_ranks[0][1] == 2:
            kickers = [rank for rank, count in sorted_ranks[1:]]
            return 'pair', sorted_ranks[0][0] * 10000 + sum(k * (13 ** i) for i, k in enumerate(kickers))
        
        else:
            return 'high_card', sum(rank * (13 ** i) for i, rank in enumerate(sorted(ranks, reverse=True)))
    
    def is_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks))
        
        # Check for A-5 straight (wheel)
        if sorted_ranks == [0, 1, 2, 3, 12]:  # 2,3,4,5,A
            return True
        
        # Check for regular straight
        if len(sorted_ranks) == 5:
            return sorted_ranks[-1] - sorted_ranks[0] == 4
        
        return False
    
    def get_hand_description(self, hand_type: str, cards: List[Card]) -> str:
        """Get human-readable description of hand"""
        ranks = [card.rank for card in cards]
        rank_counts = Counter(ranks)
        
        if hand_type == 'royal_flush':
            return "Royal Flush"
        elif hand_type == 'straight_flush':
            return f"Straight Flush, {max(ranks)} high"
        elif hand_type == 'four_of_a_kind':
            quad_rank = max(rank_counts, key=rank_counts.get)
            return f"Four of a Kind, {quad_rank}s"
        elif hand_type == 'full_house':
            sorted_counts = sorted(rank_counts.items(), key=lambda x: x[1], reverse=True)
            return f"Full House, {sorted_counts[0][0]}s over {sorted_counts[1][0]}s"
        elif hand_type == 'flush':
            return f"Flush, {max(ranks)} high"
        elif hand_type == 'straight':
            return f"Straight, {max(ranks)} high"
        elif hand_type == 'three_of_a_kind':
            trip_rank = max(rank_counts, key=rank_counts.get)
            return f"Three of a Kind, {trip_rank}s"
        elif hand_type == 'two_pair':
            pairs = [rank for rank, count in rank_counts.items() if count == 2]
            return f"Two Pair, {max(pairs)}s and {min(pairs)}s"
        elif hand_type == 'pair':
            pair_rank = max(rank_counts, key=rank_counts.get)
            return f"Pair of {pair_rank}s"
        else:
            return f"{max(ranks)} high"

class PokerEngine:
    """Main poker engine for game logic and calculations"""
    
    def __init__(self):
        self.evaluator = HandEvaluator()
    
    def calculate_equity(self, hero_hand: List[str], villain_range: str, 
                        board: List[str] = None, iterations: int = 10000) -> Dict:
        """Calculate hand equity against a range"""
        if board is None:
            board = []
        
        wins = 0
        ties = 0
        total = 0
        
        # Parse hero hand
        hero_cards = [Deck.parse_card(card) for card in hero_hand]
        board_cards = [Deck.parse_card(card) for card in board] if board else []
        
        for _ in range(iterations):
            deck = Deck()
            
            # Remove known cards
            deck.remove_cards(hero_hand + board)
            
            # Deal remaining board cards
            remaining_board = 5 - len(board)
            if remaining_board > 0:
                new_board = deck.deal(remaining_board)
                full_board = board_cards + new_board
            else:
                full_board = board_cards
            
            # Deal villain hand based on range
            villain_cards = self.deal_from_range(deck, villain_range)
            
            # Evaluate both hands
            hero_full = hero_cards + full_board
            villain_full = villain_cards + full_board
            
            hero_eval = self.evaluator.evaluate_hand([str(c) for c in hero_full])
            villain_eval = self.evaluator.evaluate_hand([str(c) for c in villain_full])
            
            # Compare hands
            hero_strength = hero_eval['ranking'] * 1000000 + hero_eval['strength']
            villain_strength = villain_eval['ranking'] * 1000000 + villain_eval['strength']
            
            if hero_strength > villain_strength:
                wins += 1
            elif hero_strength == villain_strength:
                ties += 1
            
            total += 1
        
        equity = (wins + ties * 0.5) / total * 100
        
        return {
            'equity': equity,
            'wins': wins,
            'ties': ties,
            'total': total,
            'win_percentage': wins / total * 100,
            'tie_percentage': ties / total * 100
        }
    
    def deal_from_range(self, deck: Deck, range_type: str) -> List[Card]:
        """Deal cards based on specified range"""
        if range_type == 'random':
            return deck.deal(2)
        elif range_type == 'tight':
            # Premium hands only
            premium_hands = self.get_premium_range()
            return self.deal_specific_range(deck, premium_hands)
        elif range_type == 'loose':
            # Wide range
            loose_hands = self.get_loose_range()
            return self.deal_specific_range(deck, loose_hands)
        else:
            return deck.deal(2)
    
    def get_premium_range(self) -> List[str]:
        """Get premium hand range"""
        return [
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99',
            'AKs', 'AQs', 'AJs', 'ATs',
            'AKo', 'AQo', 'AJo',
            'KQs', 'KJs', 'KTs',
            'KQo', 'KJo',
            'QJs', 'QTs',
            'QJo',
            'JTs'
        ]
    
    def get_loose_range(self) -> List[str]:
        """Get loose/wide range"""
        hands = []
        
        # All pairs
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        for rank in ranks:
            hands.append(f'{rank}{rank}')
        
        # Suited connectors and one-gappers
        for i in range(len(ranks) - 1):
            for j in range(i + 1, min(i + 3, len(ranks))):
                hands.append(f'{ranks[i]}{ranks[j]}s')
        
        # Broadway cards
        broadway = ['A', 'K', 'Q', 'J', 'T']
        for i in range(len(broadway)):
            for j in range(i + 1, len(broadway)):
                hands.append(f'{broadway[i]}{broadway[j]}o')
        
        return hands
    
    def deal_specific_range(self, deck: Deck, hand_range: List[str]) -> List[Card]:
        """Deal from specific range"""
        available_hands = []
        
        for hand in hand_range:
            if len(hand) == 2:  # Pair
                rank = hand[0]
                for suit1 in Card.SUITS:
                    for suit2 in Card.SUITS:
                        if suit1 != suit2:
                            card1 = Card(rank, suit1)
                            card2 = Card(rank, suit2)
                            if card1 in deck.cards and card2 in deck.cards:
                                available_hands.append([card1, card2])
            
            elif len(hand) == 3:  # Suited or offsuit
                rank1, rank2, suited = hand[0], hand[1], hand[2]
                
                if suited == 's':  # Suited
                    for suit in Card.SUITS:
                        card1 = Card(rank1, suit)
                        card2 = Card(rank2, suit)
                        if card1 in deck.cards and card2 in deck.cards:
                            available_hands.append([card1, card2])
                
                else:  # Offsuit
                    for suit1 in Card.SUITS:
                        for suit2 in Card.SUITS:
                            if suit1 != suit2:
                                card1 = Card(rank1, suit1)
                                card2 = Card(rank2, suit2)
                                if card1 in deck.cards and card2 in deck.cards:
                                    available_hands.append([card1, card2])
        
        if available_hands:
            chosen_hand = random.choice(available_hands)
            # Remove from deck
            deck.cards.remove(chosen_hand[0])
            deck.cards.remove(chosen_hand[1])
            return chosen_hand
        else:
            # Fallback to random
            return deck.deal(2)
    
    def get_preflop_strength(self, hand: List[str]) -> Dict:
        """Get preflop hand strength rating"""
        if len(hand) != 2:
            raise ValueError("Preflop analysis requires exactly 2 cards")
        
        card1 = Deck.parse_card(hand[0])
        card2 = Deck.parse_card(hand[1])
        
        # Determine hand type
        if card1.rank == card2.rank:
            hand_type = 'pair'
            strength = card1.value
        elif card1.suit == card2.suit:
            hand_type = 'suited'
            strength = max(card1.value, card2.value) * 13 + min(card1.value, card2.value)
        else:
            hand_type = 'offsuit'
            strength = max(card1.value, card2.value) * 13 + min(card1.value, card2.value)
        
        # Calculate percentile
        if hand_type == 'pair':
            percentile = (13 - card1.value) / 13 * 15  # Top 15% for pairs
        elif hand_type == 'suited':
            percentile = 100 - (strength / 169 * 100)
        else:
            percentile = 100 - (strength / 169 * 100) - 20  # Penalty for offsuit
        
        percentile = max(0, min(100, percentile))
        
        return {
            'hand_type': hand_type,
            'strength': strength,
            'percentile': percentile,
            'playable': percentile >= 70,
            'premium': percentile >= 90
        }