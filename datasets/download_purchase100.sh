url="https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
wget ${url}
tar -xvzf dataset_purchase.tgz
python preprocess_purchase100.py
rm dataset_purchase.tgz
rm purchase100.txt
