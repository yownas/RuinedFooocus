#!/bin/sh
# Small example how to use the gradio api to generate images with RuinedFooocus

BASEURL="http://localhost:7860/gradio_api"
ENDPOINT="generate_image"

DATA='{"data": ["Danny DeVito as gordon freeman from half life, wearing mech armor"]}'

EVENT_ID=$(curl -sX POST ${BASEURL}/call/$ENDPOINT -s -H "Content-Type: application/json" -d "$DATA" | awk -F'"' '{print $4}')

sleep 1 # Give RF some time to get things started

## Get the last "data" and base64 decode it.

RESULT=$(curl -sN ${BASEURL}/call/$ENDPOINT/$EVENT_ID | grep '^data:' | tail -1 | sed 's/^data: \["//;s/"\]$//')
IMGURL=$(echo $RESULT | sed 's/^.*\"url\": \"//;s/\", .*$//')

echo $IMGURL
