#!/bin/bash

for file in ./wav/*.wav
	do
		filename=$(basename $file)
		python app.py $filename wav2wav
		echo $filename
	done
