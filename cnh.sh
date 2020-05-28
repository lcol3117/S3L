#!/bin/bash

targetfile='nohup.out'
cat $targetfile && echo "" > $targetfile || echo "Unable to clear" $targetfile
