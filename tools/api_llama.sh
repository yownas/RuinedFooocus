#!/bin/sh
# Small example how to use the gradio api to access llama in RuinedFooocus

USER=${1:-"Please, tell me a joke about generative AI."}

DATA='{
  "data": [
    "You are an witty ai. Please help the user as best as you can.",
    "'$USER'"
  ]
}'

EVENT_ID=$(curl -sX POST http://localhost:7860/gradio_api/call/llama -s -H "Content-Type: application/json" -d "$DATA" | awk -F'"' '{print $4}')
curl -sN http://localhost:7860/gradio_api/call/llama/$EVENT_ID

