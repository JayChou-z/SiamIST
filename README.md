# SiamIST
SiamIST: Infrared small target tracking based on an improved SiamRPN
### 1.Performance
<div align=center><img width="400" height="300" src="https://github.com/JayChou-z/image_container/blob/main/Precision.png"/><img width="400" height="300" src="https://github.com/JayChou-z/image_container/blob/main/success.png"/></div>


## 2.Environment
- python=3.9  
- torch=1.12.1  
- cuda=11.3  
- torchvision=0.13.1


## 3.Datasets

Note: The test set we use consists of the last 18 weak and small single-target sequences from the 22 sequences, and the training set is also selected from these single-target video sequences.
* [train dataset](https://www.scidb.cn/en/detail?dataSetId=808025946870251520&version=V2)
* [test dataset](https://www.scidb.cn/en/detail?dataSetId=720626420933459968&version=V1)
## 4.Model
Download our pretrained model:
[model](https://pan.baidu.com/s/1WDNzGo_Zo4mlZqzjwUsW7A?pwd=jayz) code: jayz

## 5.Results
We provide tracking results for comparison: In the code directory \siamist\bin\results\22seqs22\SiamRPN.

## 6.Acknowledgement
The code is implemented based on[SiamRPN](https://github.com/HonglinChu/SiamTrackers/tree/master/SiamRPN/SiamRPN). We would like to express our sincere thanks to the contributors.

## 7.Citation
```
@article{qian2023siamist,
  title={SiamIST: Infrared small target tracking based on an improved SiamRPN},
  author={Qian, Kun and Zhang, Shou-jin and Ma, Hong-yu and Sun, Wen-jun},
  journal={Infrared Physics \& Technology},
  volume={134},
  pages={104920},
  year={2023},
  publisher={Elsevier}
}
```
