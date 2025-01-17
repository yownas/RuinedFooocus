#!/bin/sh
# Small example how to use the gradio api to search images in RuinedFooocus

BASEURL="http://localhost:7860/gradio_api"

DATA='{"data": [""]}'

EVENT_ID=$(curl -sX POST ${BASEURL}/call/search -s -H "Content-Type: application/json" -d "$DATA" | awk -F'"' '{print $4}')

sleep 1 # Give RF some time to get things started

# Get the results
RESULT=$(curl -sN ${BASEURL}/call/search/$EVENT_ID | grep '^data:' | tail -1 | sed 's/^data: \[\[//;s/\]\]$//;s/, /\n/g')
for FILE in $RESULT; do
  IMGURL=$BASEURL/file/$(echo $FILE | sed 's/^"//;s/"$//')
  echo $IMGURL
done
