#!/bin/bash
# This scripts takes all files in the TIMIT database and copies them without repeating to another folder
search_dir=TRAIN
results_dir=results
mkdir "$results_dir"

for d in $(find "$search_dir/" -maxdepth 2 -type d)
do
	for file in "$d"/*
	do
		if [ ! -e "$results_dir/${file##*/}" ]
		then
			cp "$file" "$results_dir/"
		    	echo "$0: '${file}'"
		fi
	done
done
