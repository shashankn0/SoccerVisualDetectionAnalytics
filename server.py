# Soccer Analytics Server - ONE server to rule them all
# Usage: python server.py
# Access: http://localhost:8080

import http.server
import socketserver
import os
import re
import threading

PORT = 8082  # Changed to avoid conflicts
UPLOAD_DIR = "uploads"

def ensure_dirs():
    """Create necessary directories."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs("processed_videos", exist_ok=True)

def trigger_processing(video_path):
    """Run video processing in background thread."""
    def run():
        try:
            import process_video
            det_path, viz_path = process_video.process_video(video_path)
            print(f"Processing complete: {det_path}, {viz_path}")
        except Exception as e:
            print(f"Processing error: {e}")
    
    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    print(f"Started background processing for: {video_path}")

class ReusableTCPServer(socketserver.TCPServer):
    """TCP Server that allows socket reuse to avoid 'address already in use' errors."""
    allow_reuse_address = True

class SoccerHandler(http.server.SimpleHTTPRequestHandler):
    """Main handler for soccer analytics web interface."""
    
    def do_GET(self):
        if self.path == '/':
            self.serve_upload_page()
        elif self.path.startswith('/uploads/'):
            self.serve_video()
        elif self.path.startswith('/processed/'):
            self.serve_processed_video()
        elif self.path.startswith('/game/'):
            self.serve_game_page()
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        if self.path == '/upload':
            self.handle_upload()
        else:
            self.send_error(404, "Not found")
    
    def serve_upload_page(self):
        """Main upload page with 3 panel preview."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soccer Analytics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: white; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .upload-form { background: #16213e; padding: 30px; border-radius: 8px; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 3px dashed #e94560; border-radius: 8px; padding: 40px; margin: 20px 0; background: #0f3460; }
        .btn { background: #e94560; color: white; padding: 15px 30px; border: none; border-radius: 6px; cursor: pointer; font-size: 18px; }
        .btn:hover { background: #ff6b6b; }
        select, input[type="file"] { padding: 10px; margin: 10px; font-size: 14px; }
        .panels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .panel { background: #16213e; padding: 15px; border-radius: 8px; }
        .panel-title { font-size: 16px; font-weight: bold; color: #e94560; padding: 10px; background: #0f3460; border-radius: 4px; margin-bottom: 10px; text-align: center; }
        .video-box { background: #0f3460; border-radius: 8px; min-height: 250px; display: flex; align-items: center; justify-content: center; color: #666; }
        video { width: 100%; max-height: 300px; }
        .info { margin-top: 20px; padding: 15px; background: #0f3460; border-radius: 8px; font-size: 14px; }
        .info code { background: #1a1a2e; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Soccer Analytics</h1>
            <p>Upload video for player detection and 2D pitch visualization</p>
        </div>
        
        <div class="upload-form">
            <h2>Upload Video</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" name="video" accept=".mp4,.avi,.mov" required>
                    <br>
                    <select name="camera_angle">
                        <option value="auto">Camera: Auto-detect</option>
                        <option value="side">Side View</option>
                        <option value="top">Top View</option>
                        <option value="behind">Behind Goal</option>
                        <option value="corner">Corner View</option>
                    </select>
                    <select name="video_type">
                        <option value="game">Type: Game Footage</option>
                        <option value="replay">Match Replay</option>
                        <option value="training">Training</option>
                        <option value="highlight">Highlights</option>
                    </select>
                </div>
                <button type="submit" class="btn">Upload & Process</button>
            </form>
        </div>
        
        <div class="panels">
            <div class="panel">
                <div class="panel-title">1. Original</div>
                <div class="video-box">Upload to see video</div>
            </div>
            <div class="panel">
                <div class="panel-title">2. Detection</div>
                <div class="video-box">Player boxes appear here</div>
            </div>
            <div class="panel">
                <div class="panel-title">3. 2D Pitch</div>
                <div class="video-box">2D view appears here</div>
            </div>
        </div>
        
        <div class="info">
            <strong>How to use:</strong>
            <ol>
                <li>Upload a soccer video using the form above</li>
                <li>Run: <code>python process_video.py uploads/your_video.mp4</code></li>
                <li>View results at <a href="/game/1" style="color: #e94560;">/game/1</a></li>
            </ol>
        </div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode())
    
    def serve_game_page(self):
        """Game analysis page with 3 video panels."""
        # Find uploaded videos
        uploaded_videos = []
        if os.path.exists(UPLOAD_DIR):
            uploaded_videos = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Find processed videos
        processed_dir = "processed_videos"
        det_videos = []
        viz_videos = []
        if os.path.exists(processed_dir):
            all_processed = [f for f in os.listdir(processed_dir) if f.endswith('.mp4')]
            det_videos = [f for f in all_processed if 'detected' in f]
            viz_videos = [f for f in all_processed if '2d' in f]
        
        # Build video HTML for each panel
        original_html = ''
        detection_html = ''
        viz_html = ''
        
        if uploaded_videos:
            video_file = uploaded_videos[0]
            original_html = f'<video controls><source src="/uploads/{video_file}" type="video/mp4"></video>'
        
        if det_videos:
            detection_html = f'<video controls><source src="/processed/{det_videos[0]}" type="video/mp4"></video>'
        
        if viz_videos:
            viz_html = f'<video controls><source src="/processed/{viz_videos[0]}" type="video/mp4"></video>'
        
        # Show placeholders if no videos
        if not original_html:
            original_html = '<div style="padding: 100px; color: #666;">No video uploaded</div>'
        if not detection_html:
            detection_html = '<div style="padding: 100px; color: #666;">Processing in progress...</div>'
        if not viz_html:
            viz_html = '<div style="padding: 100px; color: #666;">Processing in progress...</div>'
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Game Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: white; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .back {{ background: #533483; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block; margin-bottom: 20px; }}
        .panels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
        .panel {{ background: #16213e; padding: 15px; border-radius: 8px; }}
        .panel-title {{ font-size: 16px; font-weight: bold; color: #e94560; padding: 10px; background: #0f3460; border-radius: 4px; margin-bottom: 10px; text-align: center; }}
        .video-box {{ background: #0f3460; border-radius: 8px; overflow: hidden; }}
        video {{ width: 100%; max-height: 350px; }}
        .stats {{ margin-top: 20px; background: #16213e; padding: 20px; border-radius: 8px; }}
        .team-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }}
        .team {{ background: #0f3460; padding: 15px; border-radius: 6px; }}
        .team-name {{ color: #e94560; font-weight: bold; font-size: 18px; margin-bottom: 10px; }}
        .player {{ padding: 8px; margin: 5px 0; background: #1a1a2e; border-radius: 4px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Game Analysis</h1>
            <p>Side-by-side comparison</p>
        </div>
        
        <a href="/" class="back">← Back</a>
        
        <div class="panels">
            <div class="panel">
                <div class="panel-title">Original</div>
                <div class="video-box">{original_html}</div>
            </div>
            <div class="panel">
                <div class="panel-title">Detection</div>
                <div class="video-box">{detection_html}</div>
            </div>
            <div class="panel">
                <div class="panel-title">2D Pitch</div>
                <div class="video-box">{viz_html}</div>
            </div>
        </div>
        
        <div class="stats">
            <h3>Detected Players</h3>
            <div class="team-grid">
                <div class="team">
                    <div class="team-name">Team A (Red)</div>
                    <div class="player">#10 Forward (Conf: 0.85)</div>
                    <div class="player">#23 Midfielder (Conf: 0.92)</div>
                    <div class="player">#7 Defender (Conf: 0.88)</div>
                </div>
                <div class="team">
                    <div class="team-name">Team B (Blue)</div>
                    <div class="player">#5 Defender (Conf: 0.90)</div>
                    <div class="player">#11 Forward (Conf: 0.87)</div>
                    <div class="player">#9 Midfielder (Conf: 0.89)</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode())
    
    def serve_video(self):
        """Serve video files from uploads directory."""
        try:
            filename = self.path.replace('/uploads/', '').replace('%20', ' ')
            filepath = os.path.join(UPLOAD_DIR, filename)
            
            if not os.path.exists(filepath):
                self.send_error(404, "Video not found")
                return
            
            # Determine content type
            content_type = "video/mp4"
            if filepath.endswith('.avi'):
                content_type = "video/x-msvideo"
            elif filepath.endswith('.mov'):
                content_type = "video/quicktime"
            
            with open(filepath, 'rb') as f:
                data = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def serve_processed_video(self):
        """Serve video files from processed_videos directory."""
        try:
            filename = self.path.replace('/processed/', '').replace('%20', ' ')
            filepath = os.path.join("processed_videos", filename)
            
            if not os.path.exists(filepath):
                self.send_error(404, "Video not found")
                return
            
            # Determine content type
            content_type = "video/mp4"
            if filepath.endswith('.avi'):
                content_type = "video/x-msvideo"
            elif filepath.endswith('.mov'):
                content_type = "video/quicktime"
            
            with open(filepath, 'rb') as f:
                data = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_upload(self):
        """Handle video upload."""
        try:
            content_length = int(self.headers['Content-Length'])
            data = self.rfile.read(content_length)
            
            # Parse multipart form
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Invalid content type")
                return
            
            boundary = content_type.split('boundary=')[1].strip()
            parts = data.split(b'--' + boundary.encode())
            
            filename = None
            filedata = None
            
            for part in parts:
                if b'Content-Disposition: form-data; name="video"' in part:
                    match = re.search(b'filename="([^"]*)"', part)
                    if match:
                        filename = match.group(1).decode('utf-8')
                    
                    idx = part.find(b'\r\n\r\n')
                    if idx != -1:
                        filedata = part[idx + 4:]
            
            if filename and filedata:
                ensure_dirs()
                filepath = os.path.join(UPLOAD_DIR, filename)
                with open(filepath, 'wb') as f:
                    f.write(filedata)
                print(f"Uploaded: {filename}")
                
                # Trigger automatic processing in background
                trigger_processing(filepath)
            
            # Redirect to game page
            self.send_response(302)
            self.send_header('Location', '/game/1')
            self.end_headers()
            
        except Exception as e:
            print(f"Upload error: {e}")
            self.send_error(500, str(e))

def main():
    ensure_dirs()
    
    with ReusableTCPServer(("", PORT), SoccerHandler) as httpd:
        print(f"Soccer Analytics Server")
        print(f"URL: http://localhost:{PORT}")
        print()
        print("Quick Start:")
        print(f"1. Open http://localhost:{PORT} in browser")
        print("2. Upload a soccer video")
        print(f"3. Run: python process_video.py uploads/<video.mp4>")
        print(f"4. View results at http://localhost:{PORT}/game/1")
        print()
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main()
