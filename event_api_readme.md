# Soccer Event Interpreter API

A FastAPI backend application for interpreting and classifying live soccer match events.

## Features

- **Live Event Polling**: Automatically polls soccer API every 10 seconds for new events
- **Event Classification**: Rule-based classifier for:
  - Goals
  - Shots on target
  - Yellow cards
  - Substitutions
- **MySQL Storage**: Stores classified events with full details
- **REST API**: Exposes endpoints to query match events

## Project Structure

```
SoccerVisualDetectionAnalytics/
...
sports_interpreter_api.py    # FastAPI application with endpoints and polling logic
api_client.py                # Client for fetching events from soccer API
classifier.py                # Rule-based event classifier
database.py                  # Database configuration and session management
models.py                    # SQLAlchemy models (Event table)
.env.example                 # Environment variables template
SPORTS_INTERPRETER_README.md # This file
```

## Setup

### Prerequisites

- Python 3.8+
- MySQL 5.7+ or 8.0+
- Soccer API access (e.g., API-Football, Football-Data.org)

### Installation

1. **Set up MySQL database**:
   ```bash
   # Using MySQL client:
   mysql -u root -p
   CREATE DATABASE soccer_events;
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set:
   - `DATABASE_URL`: Your MySQL connection string (e.g. mysql+pymysql://user:password@localhost:3306/soccer_events)
   - `SOCCER_API_BASE_URL`: Your soccer API base URL
   - `SOCCER_API_KEY`: Your API key (if required)

## Usage

### Start the Sports Interpreter API server

```bash
python sports_interpreter_api.py
```

The API will be available at `http://localhost:8001`

### API Endpoints

#### Get Match Events
```bash
GET /matches/{match_id}/events
```

Returns all classified events for a specific match.

**Example:**
```bash
curl http://localhost:8001/matches/12345/events
```

#### Start Polling
```bash
POST /polling/start?match_id=12345
```

Starts polling for events. If `match_id` is provided, polls that specific match. Otherwise, polls all live matches.

**Example:**
```bash
curl -X POST http://localhost:8001/polling/start?match_id=12345
```

#### Stop Polling
```bash
POST /polling/stop
```

Stops the polling process.

#### Get Polling Status
```bash
GET /polling/status
```

Returns the current polling status.

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## How It Works

1. **Polling**: The application polls the soccer API every 10 seconds for new events
2. **Classification**: Each event is analyzed using rule-based logic to determine its type
3. **Storage**: Classified events are stored in MySQL with full details
4. **Retrieval**: Events can be queried via the REST API endpoint

## Event Classification Rules

The classifier uses pattern matching on event data:

- **Goal**: Detects keywords like "goal", "scored", checks for goal-related event types
- **Shot on Target**: Identifies shots that are on target (not blocked or off target)
- **Yellow Card**: Detects yellow card events, cautions, bookings
- **Substitution**: Identifies player substitutions and replacements

## Database Schema

The `Event` table stores:
- `id`: Primary key
- `match_id`: Match identifier
- `event_id`: Unique event ID from API
- `event_type`: Classified type (goal, shot_on_target, yellow_card, substitution)
- `player_name`: Player involved
- `team_name`: Team name
- `minute`: Match minute
- `description`: Event description
- `raw_data`: Original API response (JSON)
- `created_at`: Timestamp

## Notes

- The API client is configured to work with Football-Data.org API by default, but can be adapted to other APIs
- Events are only stored if they match one of the classified types
- Duplicate events (same `event_id` for a match) are automatically skipped
- The polling runs in a background thread and can be started/stopped via API

## Development

To modify the classifier rules, edit `classifier.py`. To change the API client behavior, edit `api_client.py`.

## Integration with Soccer Visual Detection Analytics

This sports interpreter API can be used alongside the main soccer visual detection system:
- Run the main video analysis API on port 8000
- Run the sports interpreter API on port 8001
- Both systems can complement each other for comprehensive soccer analytics

## License

MIT
