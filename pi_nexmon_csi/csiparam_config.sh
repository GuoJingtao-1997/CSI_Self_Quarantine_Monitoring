#!/bin/sh

#generate makecsiparams
cd nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams
csiparam=$(mcp -c $1/$2 -C 1 -N 1)
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -s500 -b -l34 -v$csiparam
sudo iw dev wlan0 interface add mon0 type monitor
sudo ip link set mon0 up
cd
