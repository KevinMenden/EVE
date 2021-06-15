FROM continuumio/miniconda3

COPY environment.yml /

RUN conda env create -f /environment.yml && conda clean -a

ENV PATH /opt/conda/envs/protein_env/bin:$PATH

# Install EVE
RUN mkdir eve_package
ADD EVE/ eve_package/EVE/
COPY setup.py eve_package/
COPY README.md eve_package/
COPY LICENSE eve_package/

RUN pip install -e eve_package

