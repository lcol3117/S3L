#!/bin/bash

cnh || alias cnh='bash clearnohup.sh'
echo "" > nohup.out || echo "Unable to clear nohup.out"
