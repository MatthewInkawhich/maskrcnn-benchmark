#!/bin/bash

kill $(ps aux | grep "test_net.py" | grep -v grep | awk '{print $2}')
