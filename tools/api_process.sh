#!/bin/sh
# Small example how to use the gradio api to generate images with RuinedFooocus

DATA='{"data": ["Danny DeVito as gordon freeman from half life, wearing mech armor"]}'

EVENT_ID=$(curl -sX POST http://localhost:7860/gradio_api/call/process -s -H "Content-Type: application/json" -d "$DATA" | awk -F'"' '{print $4}')
curl -sN http://localhost:7860/gradio_api/call/process/$EVENT_ID | grep '^data:' | sed 's/^data: \["//;s/"\]$//' | base64 -d

