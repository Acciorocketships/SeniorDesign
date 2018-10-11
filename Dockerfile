#FROM python:3.7.0-stretch
FROM ubuntu:16.04

MAINTAINER Edward Atter <atter@seas.upenn.edu>

RUN apt-get update -y
RUN apt install cmake python-sphinx libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-date-time-dev libboost-iostreams-dev libeigen3-dev libflann-dev libvtk6-dev libqhull-dev libopenni-dev libqt5opengl5-dev libqt4-opengl-dev libusb-1.0-0-dev freeglut3-dev libxmu-dev libxi-dev libvtk6-qt-dev doxygen doxygen-latex -y
RUN apt install dh-exec -y
RUN apt install build-essential devscripts -y
RUN apt-get install libproj-dev -y
RUN dget -u https://launchpad.net/ubuntu/+archive/primary/+files/pcl_1.7.2-14ubuntu1.16.04.1.dsc
RUN cd pcl-1.7.2 && dpkg-buildpackage -r -uc -b
RUN pwd && ls -l
# RUN dpkg -i pcl_*.deb
RUN apt-get install libboost-all-dev -y
RUN dpkg -i libpcl*.deb
RUN dpkg -i pcl-tools*
RUN apt-get install python-pip -y
RUN pip install --upgrade pip
RUN pip install cython==0.25.2
RUN pip install numpy

RUN apt-get install git -y
RUN cd /tmp && git clone https://github.com/strawlab/python-pcl.git
RUN cd /tmp/python-pcl && python setup.py build_ext -i
RUN cd /tmp/python-pcl && python setup.py install

ENTRYPOINT ["/bin/bash"]
#RUN cd pcl-1.7.2 && python setup.py build_ext -i
#RUN cd pcl-1.7.2 && python setup.py install

