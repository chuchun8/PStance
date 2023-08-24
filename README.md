# PStance

ACL 2021 (Findings) paper: P-Stance: A Large Dataset for Stance Detection in Political Domain.

#Please request the dataset at yli300@uic.edu.
The dataset is available at [here](https://drive.google.com/drive/folders/1so8lY1XKpnhUtTvb15edEz6aeHt7CSuh?usp=sharing).

## Abstract

Stance detection determines whether the author of a text is in favor of, against or neutral to a specific target and provides valuable insights into important events such as presidential election. However, progress on stance detection has been hampered by the absence of large annotated datasets. In this paper, we present P-Stance, a large stance detection dataset in the political domain, which contains 21,574 labeled tweets. We provide a detailed description of the newly created dataset and develop deep learning models on it. Our best model achieves a macro-average F1-score of 80.53%, which we improve further by using semi-supervised learning. Moreover, our P-Stance dataset can facilitate research in the fields of cross-domain stance detection such as cross-target stance detection where a classifier is adapted from a different but related target.

## Run

BERTweet is used as our baseline for in-target stance detection and cross-target stance detection in this paper. First, configure the environment:
```
$ pip install -r requirements.txt
```
Then run
```
cd source/
python train_model.py \
    --input_target <target name> \
    --model_select <BERTweet or BERT> \
    --train_mode <unified or adhoc> \
    --lr <learning rate> \
    --batch_size 32 \
    --epochs 3 \
```
`input_target` can take one of the following targets [`trump`, `biden`, `bernie`] in adhoc setting and take [`all`] in unified setting.

Or run jupyter notebook example `pstance_run.ipynb`

## Contact Info

Please contact Yingjie Li at yli300@uic.edu with any questions.


 
