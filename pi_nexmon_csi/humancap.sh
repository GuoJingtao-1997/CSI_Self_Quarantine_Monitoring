#!/bin/sh

cd ~/bussit

if [ ! -d "$1" ];then
	mkdir $1
else
	echo "folder $1 is existed"
fi

for j in $(seq $2 $3)
do
	echo $(sudo tcpdump -i wlan0 dst port 5500 -vv -w $1/$j.pcap -c $4);
	echo Num of Files: $j
done
echo "finish capture $1"
cd
