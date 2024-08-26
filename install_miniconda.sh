set -e
if [ ! -f /opt/miniconda-installer.sh ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
fi
bash /opt/miniconda-installer.sh -b -u -p /opt/miniconda3
/opt/miniconda3/bin/conda init bash
source ~/.bashrc