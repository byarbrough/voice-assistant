#!/bin/bash

# This shell script starts a Docker container
# It then waits for a GPIO rising edge on the Jetson Orin Nano Developer Kit
# On that event it makes a GET request to the container, which is then printed to the terminal.

ASSISTANT_IMAGE=voice_assistant
HOST="http://127.0.0.1:8000"

echo "Application starting"

# Check if desired image exists; if not, build it
if sudo docker image inspect $ASSISTANT_IMAGE > /dev/null 2>&1; then
    echo $ASSISTANT_IMAGE image found!
else
    echo $ASSISTANT_IMAGE not found. Building now!
    sudo docker buildx build . -t $ASSISTANT_IMAGE
fi

echo "Running docker container"
sudo docker run -it --rm --device=/dev/snd \
    --runtime=nvidia --ipc=host -p 8000:8000 \
    -v huggingface:/huggingface/ \
    --name $ASSISTANT_IMAGE $ASSISTANT_IMAGE

echo "Sleeping for 5 seconds to allow startup"
sleep 5

echo "Testing if app is up"
while ! curl -s "$HOST/ping"; do
    echo "Sleep again"
    sleep 1
done

echo "Success!"

# GPIO loop
echo "GPIO detection running..."
while true; do
    gpiomon -r -n 1 gpiochip0 105 | while read line; do
        echo "event $line"
        curl $HOST/weather
    done
done
