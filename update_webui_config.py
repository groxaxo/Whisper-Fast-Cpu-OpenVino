import sqlite3
import json

db_path = "/home/op/.local/share/pipx/venvs/open-webui/lib/python3.11/site-packages/open_webui/data/webui.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, data FROM config LIMIT 1")
row = cursor.fetchone()

if row:
    config_id, config_json_str = row
    config = json.loads(config_json_str)
    
    # Update STT
    if "audio" in config and "stt" in config["audio"]:
        if "openai" in config["audio"]["stt"]:
            config["audio"]["stt"]["openai"]["api_base_url"] = "http://localhost:8000/v1"
            print(f"Updated STT API URL to http://localhost:8000/v1")
            
    # Update TTS
    if "audio" in config and "tts" in config["audio"]:
        if "openai" in config["audio"]["tts"]:
            config["audio"]["tts"]["openai"]["api_base_url"] = "http://localhost:8880/v1"
            print(f"Updated TTS API URL to http://localhost:8880/v1")
            
    new_config_json_str = json.dumps(config)
    cursor.execute("UPDATE config SET data = ? WHERE id = ?", (new_config_json_str, config_id))
    conn.commit()
    print("Configuration updated in database.")
else:
    print("No config found in database.")

conn.close()
