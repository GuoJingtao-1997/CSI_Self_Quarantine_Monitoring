#!/bin/sh

Fs="$(echo "1 / $1" | bc -l)"

if [ -z "$Fs" ]
then
	Fs=0.02
fi

sudo ping -I wlan0 -f -i "$Fs" $2
