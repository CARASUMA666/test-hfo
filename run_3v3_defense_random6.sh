#!/bin/bash

/home/tanghongyao/HFO/HFO/bin/HFO --offense-npcs=3 --defense-agents=2 --defense-npcs=1 --trials 20000 --headless --no-logging --port 6677 &
sleep 5

# If wanting to test below with different python versions, add -x to avoid
# the #!/usr/bin/env python initial line.

python /home/tanghongyao/HFO/codes/3v3_random6.py --port 6677 --agent-num 2 --seed 333 &> logs/3v3_defense_random6_seed333.txt &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
