# unified soccer analytics system

a complete soccer analytics platform that combines video detection with live event interpretation.

## features

- **video analysis**: detect players, ball, and field elements in soccer videos
- **live event polling**: automatically fetch and classify soccer match events
- **event classification**: rule-based classifier for goals, shots, cards, and substitutions
- **unified api**: single endpoint for both video and event data
- **database storage**: mysql backend for persistent event storage

## quick start

### 1. setup environment

```bash
# install dependencies
pip install -r requirements.txt

# setup mysql database
mysql -u root -p
create database soccer_events;
```

### 2. configure environment

```bash
cp .env.example .env
# edit .env with your database and api settings
```

### 3. start the unified api

```bash
python unified_soccer_api.py
```

the api will be available at `http://localhost:8000`

## api endpoints

### video analysis

- `post /upload_video` - upload video file
- `post /analyze_video/{filename}` - analyze uploaded video

### event management

- `get /matches/{match_id}/events` - get events for a match
- `post /add_events` - add events from video analysis
- `post /polling/start` - start live event polling
- `post /polling/stop` - stop polling
- `get /polling/status` - check polling status

## workflow example

1. **upload video**: `post /upload_video` with your soccer video
2. **analyze video**: `post /analyze_video/your_video.mp4` to detect objects
3. **add events**: use `post /add_events` to store detected events
4. **live polling**: start `post /polling/start` for real-time event updates

## file structure

```
soccer_visual_detection_analytics/
├── unified_soccer_api.py      # main unified api
├── video_detector.py          # video detection logic
├── detection_extractor.py     # data extraction from videos
├── event_api.py              # event interpretation api
├── api_client.py             # external soccer api client
├── classifier.py             # event classification rules
├── database.py               # database configuration
├── models.py                 # database models
├── requirements.txt           # python dependencies
├── .env.example             # environment template
└── uploaded_videos/          # video upload directory
```

## model files

make sure you have the following model files in the `models/` directory:
- `player.pt` - player detection model
- `ball.pt` - ball detection model  
- `field.pt` - field detection model

## database schema

the `events` table stores:
- match information
- event type (goal, shot, card, substitution)
- player and team details
- timestamps and descriptions

## development

to modify detection models, edit `video_detector.py`.
to change event classification rules, edit `classifier.py`.
to adjust api behavior, edit `unified_soccer_api.py`.

## license

mit
