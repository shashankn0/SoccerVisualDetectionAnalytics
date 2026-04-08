# api client for fetching live soccer events
import requests
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# api configuration - using a free soccer api (you can replace with your preferred api)
# Example: API-Football, Football-Data.org, etc.
API_BASE_URL = os.getenv(
    "SOCCER_API_BASE_URL",
    "https://api.football-data.org/v4"
)
API_KEY = os.getenv("SOCCER_API_KEY", "")


class SoccerAPIClient:
    """client for fetching live soccer events from external api."""

    def __init__(self, base_url: str = None, api_key: str = None):
        """
        initialize api client.
        
        args:
            base_url: base url for the soccer api
            api_key: api key for authentication
        """
        self.base_url = base_url or API_BASE_URL
        self.api_key = api_key or API_KEY
        self.headers = {}
        
        if self.api_key:
            self.headers["X-Auth-Token"] = self.api_key

    def get_match_events(self, match_id: int) -> List[Dict]:
        """
        fetch events for a specific match.
        
        args:
            match_id: the id of the match
            
        returns:
            list of event dictionaries from the api
        """
        try:
            url = f"{self.base_url}/matches/{match_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            match_data = response.json()
            
            # extract events from match data
            # structure may vary by api - adjust based on your api's response format
            events = match_data.get("events", [])
            
            # if api doesn't return events directly, try alternative endpoints
            if not events:
                events_url = f"{self.base_url}/matches/{match_id}/events"
                events_response = requests.get(events_url, headers=self.headers, timeout=10)
                if events_response.status_code == 200:
                    events_data = events_response.json()
                    events = events_data.get("events", events_data.get("data", []))
            
            return events or []
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching match events: {e}")
            return []

    def get_live_matches(self) -> List[Dict]:
        """
        fetch list of live/ongoing matches.
        
        returns:
            list of match dictionaries
        """
        try:
            url = f"{self.base_url}/matches"
            params = {"status": "LIVE"}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            matches = data.get("matches", data.get("data", []))
            return matches or []
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching live matches: {e}")
            return []

    def get_match_by_id(self, match_id: int) -> Optional[Dict]:
        """
        fetch match details by id.
        
        args:
            match_id: the id of the match
            
        returns:
            match dictionary or none if not found
        """
        try:
            url = f"{self.base_url}/matches/{match_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching match {match_id}: {e}")
            return None
