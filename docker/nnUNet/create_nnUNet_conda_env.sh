source /opt/conda/bin/activate
conda init bash
conda config --set auto_activate_base true
conda create -n nnUNet python=3.8 -y

echo 'conda activate nnUNet' >> /home/$user/.bashrc