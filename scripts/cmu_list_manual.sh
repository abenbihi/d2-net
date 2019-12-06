#!/bin/sh

MACHINE=1
if [ "$MACHINE" -eq 0 ]; then
  export WS_DIR=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  export WS_DIR=/home/gpu_user/assia/ws/
else
  echo "Get your MTF MACHINE macro correct. Bye !"
  exit 1
fi

img_dir="$WS_DIR"datasets/Extended-CMU-Seasons-Undistorted/

fn_list=list/cmu/list2.txt

prefix=$(echo "$fn_list" | cut -d'/' -f3 | cut -d'.' -f1)

echo "$prefix"
mkdir -p res/"$prefix"/img_match_raw/
mkdir -p res/"$prefix"/img_match_F/
mkdir -p res/"$prefix"/match_F/
mkdir -p res/"$prefix"/match_raw/
mkdir -p res/"$prefix"/img_kp/

python3 -m play.cmu.match \
  --img_dir "$img_dir" \
  --fn_list "$fn_list" \
  --show_extra
