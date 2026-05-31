#!/bin/bash
while true; do
    if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/ 2>/dev/null | grep -q "200"; then
        cd /home/z/my-project && npm run dev > /dev/null 2>&1 &
    fi
    sleep 10
done
