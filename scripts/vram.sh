#!/bin/bash
watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf \"Used: %d MiB / %d MiB (%.1f%%)\\n\", $1, $2, ($1/$2)*100}'"
