FROM cybered/python-pcl

RUN apt-get update
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow 
RUN pip3 install numpy 
RUN pip3 install scipy 
RUN pip3 install opencv-python 
RUN pip3 install pillow 
RUN pip3 install matplotlib 
RUN pip3 install h5py 
RUN pip3 install keras 
RUN pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 