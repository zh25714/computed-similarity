#!/bin/bash
set -e

echo "Downloading SICK dataset"
mkdir -p data/sick/
cd data/sick/
wget -q -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip
unzip -q -o sick_train.zip
wget -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip
unzip -q -o sick_trial.zip
wget -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip
unzip -q -o sick_test_annotated.zip
rm *.zip readme.txt
cd ../../

echo "Downloading Stanford parser and tagger"

echo "Downloading GLOVE"
