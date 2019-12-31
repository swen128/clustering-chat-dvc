FROM python:3.7.4-slim

WORKDIR /workdir

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV IPA_VER=00401

RUN apt-get update && \
    apt-get install -y wget unzip && \
    wget "https://ipafont.ipa.go.jp/IPAexfont/ipaexg${IPA_VER}.zip" && \
    unzip ipaexg${IPA_VER}.zip && \
    apt-get purge -y wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    cp ipaexg${IPA_VER}/ipaexg.ttf /usr/local/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/ipaexg.ttf && \
    echo "font.family : IPAexGothic" >> /usr/local/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc && \
    rm -rf font.zip ipaexg${IPA_VER}
