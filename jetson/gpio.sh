#!/bin/bash

# This shell script starts a Docker container
# It then waits for a GPIO rising edge on the Jetson Orin Nano Developer Kit
# On that event it makes a GET request to the container, which is then printed to the terminal.

echo "Application starting"
