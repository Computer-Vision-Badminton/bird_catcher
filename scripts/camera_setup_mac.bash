pip3 install opencv-python
xcode-select --install 
git clone https://github.com/jtfrey/uvc-util
cd uvc-util/src
clang -o uvc-util -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m
sudo cp uvc-util /usr/local/bin/
uvc-util --list-devices
uvc-util -I 0 -c  