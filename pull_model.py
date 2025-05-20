#!/usr/bin/env python
import ollama

print("Pulling llama3 model from Ollama...")
try:
    # Set Ollama host if needed
    # ollama.host = "http://localhost:11434"  # Uncomment if needed
    
    # Pull the model
    print("Starting to pull llama3 model...")
    result = ollama.pull("llama3")
    
    # Print status
    print(f"Model successfully pulled: llama3")
    print("Model information:")
    print(f"Digest: {result.get('digest', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
    
except Exception as e:
    print(f"Error pulling model: {str(e)}")
