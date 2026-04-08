"""Rule-based classifier for soccer events."""
from typing import Dict, Optional


class EventClassifier:
    """Classifies soccer events into predefined categories using rule-based logic."""

    @staticmethod
    def classify(event_data: Dict) -> Optional[str]:
        """
        Classify an event based on its data.
        
        Args:
            event_data: Dictionary containing event information from API
            
        Returns:
            Event type string: 'goal', 'shot_on_target', 'yellow_card', 'substitution', or None
        """
        # Normalize event data keys (different APIs may use different field names)
        event_type = event_data.get("type", event_data.get("eventType", "")).lower()
        event_detail = event_data.get("detail", event_data.get("eventDetail", "")).lower()
        comment = event_data.get("comment", event_data.get("description", "")).lower()
        
        # Combine all text fields for pattern matching
        combined_text = f"{event_type} {event_detail} {comment}".lower()
        
        # Rule 1: Goal classification
        if EventClassifier._is_goal(event_type, event_detail, combined_text):
            return "goal"
        
        # Rule 2: Shot on target classification
        if EventClassifier._is_shot_on_target(event_type, event_detail, combined_text):
            return "shot_on_target"
        
        # Rule 3: Yellow card classification
        if EventClassifier._is_yellow_card(event_type, event_detail, combined_text):
            return "yellow_card"
        
        # Rule 4: Substitution classification
        if EventClassifier._is_substitution(event_type, event_detail, combined_text):
            return "substitution"
        
        return None

    @staticmethod
    def _is_goal(event_type: str, event_detail: str, combined_text: str) -> bool:
        """Check if event is a goal."""
        goal_indicators = [
            "goal",
            "scored",
            "gól",
            "but",
            "score"
        ]
        
        # Check if type or detail explicitly indicates goal
        if event_type in ["goal", "score"]:
            return True
        
        if event_detail in ["goal", "normal goal", "penalty goal", "own goal"]:
            return True
        
        # Check combined text for goal-related keywords
        for indicator in goal_indicators:
            if indicator in combined_text and "own goal" not in combined_text:
                # Additional check: make sure it's not about "goal kick" or similar
                if "goal kick" not in combined_text and "goal attempt" not in combined_text:
                    return True
        
        return False

    @staticmethod
    def _is_shot_on_target(event_type: str, event_detail: str, combined_text: str) -> bool:
        """Check if event is a shot on target."""
        shot_indicators = [
            "shot",
            "attempt",
            "on target",
            "shot on goal",
            "on goal"
        ]
        
        # Check if type indicates shot
        if event_type in ["shot", "attempt"]:
            # Must be on target (not blocked or off target)
            if "on target" in combined_text or "on goal" in combined_text:
                return True
            if "blocked" not in combined_text and "off target" not in combined_text:
                # If it's a shot type but not explicitly off target, might be on target
                if event_detail in ["on target", "on goal"]:
                    return True
        
        # Check combined text
        if "shot on target" in combined_text or "shot on goal" in combined_text:
            return True
        
        return False

    @staticmethod
    def _is_yellow_card(event_type: str, event_detail: str, combined_text: str) -> bool:
        """Check if event is a yellow card."""
        card_indicators = [
            "yellow card",
            "yellow",
            "caution",
            "booking"
        ]
        
        # Check if type or detail indicates yellow card
        if event_type in ["card", "yellow card", "caution"]:
            if "yellow" in combined_text or "yellow" in event_detail:
                return True
        
        if event_detail in ["yellow card", "yellow"]:
            return True
        
        # Check combined text
        for indicator in card_indicators:
            if indicator in combined_text:
                # Make sure it's not a red card
                if "red card" not in combined_text and "red" not in event_detail:
                    return True
        
        return False

    @staticmethod
    def _is_substitution(event_type: str, event_detail: str, combined_text: str) -> bool:
        """Check if event is a substitution."""
        substitution_indicators = [
            "substitution",
            "substitute",
            "sub",
            "replacement",
            "player change"
        ]
        
        # Check if type or detail indicates substitution
        if event_type in ["substitution", "sub", "substitute"]:
            return True
        
        if event_detail in ["substitution", "substitute"]:
            return True
        
        # Check combined text
        for indicator in substitution_indicators:
            if indicator in combined_text:
                return True
        
        return False

    @staticmethod
    def extract_event_details(event_data: Dict) -> Dict:
        """
        Extract relevant details from event data for storage.
        
        Args:
            event_data: Raw event data from API
            
        Returns:
            Dictionary with extracted fields
        """
        return {
            "event_id": str(event_data.get("id", event_data.get("eventId", ""))),
            "player_name": event_data.get("player", {}).get("name") if isinstance(event_data.get("player"), dict) else event_data.get("playerName"),
            "team_name": event_data.get("team", {}).get("name") if isinstance(event_data.get("team"), dict) else event_data.get("teamName"),
            "minute": event_data.get("minute", event_data.get("time", {}).get("elapsed") if isinstance(event_data.get("time"), dict) else None),
            "description": event_data.get("comment", event_data.get("description", "")),
            "raw_data": event_data
        }
