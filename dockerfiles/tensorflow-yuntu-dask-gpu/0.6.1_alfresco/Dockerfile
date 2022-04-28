#dummy line to trigger build
FROM sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1_dev
ENV PIP_PACKAGES_COMMON="python-dotenv==0.19.2"
RUN pip install $PIP_PACKAGES_COMMON

# Install yuntu from git with deps
RUN mkdir -p /extra_code
RUN cd /extra_code
ADD . /extra_code/yuntu-private
RUN cd /extra_code/yuntu-private/src && \
    python3 setup.py install

VOLUME ["/shared_volume"]
ENV NB_PREFIX audio_dgpi
WORKDIR /
ENTRYPOINT ["/usr/local/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.allow_origin='*'", "--LabApp.base_url=audio_dgpi"]    
