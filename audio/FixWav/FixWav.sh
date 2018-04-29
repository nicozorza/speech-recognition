#!/bin/bash
# This scripts allows to convert the Sphere NIST WAV format used in the TIMIT dataset to normal WAV format

search_dir=wav
this_dir=FixWav

cd ..
for file in "$search_dir"/*
do
  	echo "$file"
	./"$this_dir"/sph2pipe -t : -f rif "$file" out.wav
	mv out.wav "$file"
done
