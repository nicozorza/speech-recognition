#!/bin/bash
# This scripts takes all files in the TIMIT database and copies them to another folder
search_dir=TRAIN
results_dir=results
mkdir "$results_dir"

for d in $(find "$search_dir/" -maxdepth 2 -type d)
do
	for file in "$d"/*
	
	do
		extension="${file##*.}"
		filename="${file%.*}"
		folder="$(cut -d'/' -f3 <<<${filename})"
		
		cp "$file" "${filename}_${folder}.${extension}"
		mv "${filename}_${folder}.${extension}" "$results_dir/"
		echo "$0: '${file}'"
	done
done
