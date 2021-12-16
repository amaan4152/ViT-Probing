#!/bin/bash

EXT=".png"
SUB="exp"
EXP_NUMS=$(basename -s "$ext" ../python/plots/* | grep -i "exp" | grep -o -E '[0-9]+')
MAX=$(echo "${EXP_NUMS[@]}" | sort -nr | head -n1)
NEW_EXP=$((${MAX} + 1))

PLT_NAME="vit_exp${NEW_EXP}.png"
if [[ $1 == "vit" ]]; then
    PLT_NAME="pconv_exp${NEW_EXP}.png"
fi

touch "$PLT_NAME"




