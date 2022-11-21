#!/bin/bash

# reset redis running on docker container
# Usage: ./reset_redis.sh

docker exec -it sentence-similarity-cache redis-cli flushall