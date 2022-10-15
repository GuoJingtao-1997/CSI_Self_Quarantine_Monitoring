#!/bin/bash
###
 # @Author: Guo Jingtao
 # @Date: 2022-04-02 18:53:40
 # @LastEditTime: 2022-07-29 17:35:03
 # @LastEditors: Guo Jingtao
 # @Description: 
 # @FilePath: /undefined/Users/guojingtao/Documents/pi_config_files/human_actcap.sh
 # 
### 

cd $(pwd)/$1

if [ ! -d "$2" ];then
	mkdir $2
	else
	echo "folder $2 is existed"
fi

for j in $(seq $3 $4)
do
	echo $(sudo tcpdump -i wlan0 dst port 5500 -vv -w $2/$j.pcap -c $5);
	read -n 1 -p "Do you want to continue [Y/N]? " answer
	printf "\n"  #wrap text
	case $answer in
		Y | y)	echo "fine, continue on..."
				continue;;
		N | n | *)	echo "OK, goodbye"
					break 2;;
	esac
	echo "finish capture in folder $2"
done

cd
