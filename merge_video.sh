#!/bin/bash

# inputs:
# 1. path to baseline video
# 2. path to CLNeRF video
# 3. path to UB video
# 4. path to output video 
# 5. baseline name
# 6. root_dir of the data folder (to extract task ids)
# 7. task_number

export ROOT_DIR=dataset/WOT

# breville
# MEIL
# scene_name=breville
# python merge_video.py \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_0/video/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/EWC/${scene_name}_0/video/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_10/video/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/${scene_name}_0_MEIL/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/UB/colmap_ngpa_CLNerf_render/breville_0_UB/rgb.mp4" \
# 	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/comparisons_UB.mp4" \
# 	'NT' \
# 	'EWC' \
# 	'ER' \
# 	'MEIL-NeRF' \
# 	${ROOT_DIR}/${scene_name} \
# 	5 \
# 	1


scene_name=mac
python merge_video.py \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_0/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/EWC/${scene_name}_0/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/WOT/NT_ER/${scene_name}_10/video/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/MEIL/colmap_ngpa_CLNerf_render/${scene_name}_0_MEIL/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/UB/colmap_ngpa_CLNerf_render/${scene_name}_0_UB/rgb.mp4" \
	"/export/work/zcai/WorkSpace/CLNeRF/CLNeRF/results/video_demo/colmap_ngpa_CLNerf_render/${scene_name}_10_CLNeRF/comparisons_UB.mp4" \
	'NT' \
	'EWC' \
	'ER' \
	'MEIL-NeRF' \
	${ROOT_DIR}/${scene_name} \
	6 \
	1
