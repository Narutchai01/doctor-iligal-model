from fastapi import FastAPI, Request
import uvicorn
import json
import sys
import ollama
import os
from dotenv import load_dotenv
from send_message import send_message
from downlonad_image import download_image
from predict_acne import predict_acne
from supabase import create_client, Client
from pymongo import MongoClient
from pydantic import BaseModel
from bson.objectid import ObjectId

# MongoDB Configuration
mongo_client = MongoClient(
    os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client[os.getenv("MONGODB_DB", "mydatabase")]

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

mongo_client = MongoClient(
    os.getenv("MONGODB_URI", "mongodb://localhost:27017"))

db = mongo_client['mydatabase']  # Replace with your database name
collection = db['mycollection']  # Replace with your collection name


class Item(BaseModel):
    image: str
    user_id: str
    description: str


# Get environment variables with fallback values
line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
port = int(os.getenv("PORT", 8000))
host = os.getenv("HOST", "0.0.0.0")
debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL",
                         "scb10x/typhoon2.1-gemma3-12b")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

supabase: Client = create_client(supabase_url, supabase_key)

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
                fileName = download_image(image_id, header)
                result_pred = predict_acne(fileName)
                acne_prompt = f"Based on this acne analysis result: {result_pred}, please analyze the severity level of the acne (mild, moderate, or severe) and provide health advice and recommendations for treating acne in Thai language. Include skincare tips, lifestyle changes, severity assessment, and when to consult a dermatologist. Please respond in Thai."

                with open(fileName, 'rb') as image_file:
                    response_image = supabase.storage.from_("acne").upload(
                        f"images/{image_id}.jpg", image_file, {"contentType": "image/jpeg"})
                if hasattr(response_image, 'error') and response_image.error:
                    print(
                        f"Error uploading image to Supabase: {response_image.error}")
                    send_message(line_token, user_id,
                                 "ไม่สามารถอัพโหลดรูปภาพได้")
                    return {"message": "200 OK", "data": events}

                # Call Ollama model with the acne prompt
                text_response = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a dermatologist who can assess acne severity and provide treatment recommendations. Analyze the acne condition and provide professional medical advice including severity assessment."},
                        {"role": "user", "content": acne_prompt}
                    ]
                )

                image_url = supabase.storage.from_(
                    "acne").get_public_url(f"images/{image_id}.jpg")

                data = {
                    "image": image_url,
                    "user_id": user_id,
                    "description": text_response.get('message', {}).get('content', "")
                }

                collection.insert_one(data)
                response_text = text_response.get('message', {}).get(
                    'content', "ไม่สามารถสร้างคำแนะนำด้านสุขภาพได้")
                send_message(line_token, user_id, response_text)

        elif message_type == "text":
            text = message.get('text', '')
            print(f"Received text: {text}")
            prompt = text[1:]
            print(f"Prompt: {prompt}")
            try:
                response = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a doctor who can consult about facial health, skincare, and dermatological conditions. Provide professional medical advice and treatment recommendations."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.get('message', {}).get(
                    'content', "No response generated")
                print(f"Response: {response_text}")
                send_message(line_token, user_id, response_text)
            except Exception as e:
                print(f"Error with Ollama: {str(e)}")
                send_message(line_token, user_id,
                             f"Error processing your request: {str(e)}")

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
