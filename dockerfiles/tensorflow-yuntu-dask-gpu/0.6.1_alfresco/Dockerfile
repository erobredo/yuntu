# dummy line to trigger build
FROM tensorflow/tensorflow:2.3.0rc2-gpu-jupyter
USER root
ENV TIMEZONE America/Mexico_City
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV DEB_BUILD_DEPS="sudo nano less git wget curl python3-dev python3-pip python3-setuptools"
ENV DEB_PACKAGES=""
ENV PIP_PACKAGES_COMMON="pytest python-dotenv==0.19.2"
ENV JUPYTERLAB="jupyter-client==5.3.4 jupyterlab==2.3.2 qtconsole==5.2.1 nbconvert==5.6.1 jupyter-console==6.4.0 ipykernel==6.6.0 MarkupSafe==2.0.1"
ENV PIP_PACKAGES_KALE="kubeflow-kale==0.6.1"
RUN apt-get update && export $DEBIAN_FRONTEND && \
    echo $TIMEZONE > /etc/timezone && apt-get install -y tzdata
RUN apt-get update && apt-get install -y $DEB_BUILD_DEPS $DEB_PACKAGES && pip install --upgrade pip  

# Install nodejs deps
RUN curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash - && apt-get install -y nodejs

RUN pip3 install $JUPYTERLAB

# Assign jupyter password
RUN jupyter notebook --generate-config && \
    sed -i "s/#c.NotebookApp.password = .*/c.NotebookApp.password = u'sha1:9361dd216701:df4736638013cd671a09ef12a671195e8fc61505'/" ~/.jupyter/jupyter_notebook_config.py && \
    sed -i "s/#c.NotebookApp.certfile = .*/c.NotebookApp.certfile = u'\/shared_volume\/certs\/cert.pem'/" ~/.jupyter/jupyter_notebook_config.py && \
    sed -i "s/#c.NotebookApp.keyfile = .*/c.NotebookApp.keyfile = u'\/shared_volume\/certs\/privkey.key'/" ~/.jupyter/jupyter_notebook_config.py
# Install dask-cuda
RUN python3 -m pip install dask-cuda==0.20.0a210416
# Install yuntu from git with deps
RUN mkdir -p /extra_code
RUN cd /extra_code
ADD . /extra_code/yuntu-private
RUN cd /extra_code/yuntu-private/src && \
    python3 setup.py install

RUN python3 -m pip install sidekit guano timezonefinder s3fs "tensorflow_probability<=0.11.1"    
#from statement installs 2.1.0, next line upgrade to specific version
RUN pip3 install --upgrade tensorflow-gpu==2.3.0

RUN pip install $PIP_PACKAGES_COMMON

#kale installation
RUN pip install $PIP_PACKAGES_KALE
RUN jupyter labextension install kubeflow-kale-labextension@0.6.1
    
VOLUME ["/shared_volume"]
ENV NB_PREFIX audio_dgpi
WORKDIR /
ENTRYPOINT ["/usr/local/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.allow_origin='*'", "--LabApp.base_url=audio_dgpi"]    
