from fastapi import FastAPI, Request
import uvicorn
import json
import sys
import ollama
import os
from dotenv import load_dotenv
from send_message import send_message
from downlonad_image import download_image 

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get environment variables with fallback values
line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
port = int(os.getenv("PORT", 8000))
host = os.getenv("HOST", "0.0.0.0")
debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "scb10x/typhoon2.1-gemma3-12b:latest")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Configure Ollama client (if needed)
if ollama_host:
    ollama.host = ollama_host

header = {
    "Authorization": f"Bearer {line_token}",
    "Content-Type": "application/json",
}


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI model service!"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/config")
async def get_config():
    """Return the current configuration (excluding sensitive values)"""
    return {
        "server": {
            "host": host,
            "port": port,
            "debug": debug
        },
        "ollama": {
            "default_model": ollama_model,
            "host": ollama_host
        }
    }


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.body()
        data = data.decode("utf-8")
        data = json.loads(data)
        events = data.get("events", [])
        
        if not events or not isinstance(events, list):
            return {"message": "200 OK", "data": []}
        
        event = events[0]
        message = event.get('message', {})
        message_type = message.get('type')
        user_id = event.get('source', {}).get('userId')
        
        if message_type == "image":
            image_id = message.get('id')
            if image_id:
                download_image(image_id, header)
                send_message(line_token, user_id, "Image received and downloaded.")
        elif message_type == "text":
            text = message.get('text', '')
            print(f"Received text: {text}")
            prompt = text[1:]
            print(f"Prompt: {prompt}")
            try:
                response = ollama.chat(model='scb10x/typhoon2.1-gemma3-12b', messages=[{"role": "user", "content": prompt}])
                response_text = response.get('message', {}).get('content', "No response generated")
                print(f"Response: {response_text}")
                send_message(line_token, user_id, response_text)
            except Exception as e:
                print(f"Error with Ollama: {str(e)}")
                send_message(line_token, user_id, f"Error processing your request: {str(e)}")

        print(json.dumps(events[0], indent=2))
        sys.stdout.flush()
        
        return {"message": "200 OK", "data": events}
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        return {"message": "Error", "error": str(e)}


def main():
    print("Starting FastAPI server with hot reload...")
    print(f"Server config: host={host}, port={port}, debug={debug}")
    uvicorn.run("main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
