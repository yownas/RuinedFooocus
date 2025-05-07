#!/bin/sh
# Small example how to use the gradio api to generate images with RuinedFooocus

BASEURL="http://localhost:7860/gradio_api"

DATA='{"data": ["Danny DeVito as gordon freeman from half life, wearing mech armor"]}'

EVENT_ID=$(curl -sX POST ${BASEURL}/call/prompt2url -s -H "Content-Type: application/json" -d "$DATA" | awk -F'"' '{print $4}')

sleep 1 # Give RF some time to get things started

# Get the last "data" and base64 decode it.
IMGURL=$(curl -sN ${BASEURL}/call/prompt2url/$EVENT_ID | grep '^data:' | tail -1 | sed 's/^data: \["//;s/"\]$//')

echo $IMGURL
