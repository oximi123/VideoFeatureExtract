你需要安装如下的包来运行这些代码
python:
		opencv
		pytorch 1.2以上

文档中的根据关键帧提取的restnet代码位于ImageNNExtract.py中
根据SFIT等普通方法提取的代码位于NormalExtract.py中
以上两者所用的数据集均为video/ 其中包含了video.mp4的关键帧
根据三维卷积的代码位于VideoNNExtract.py中
用到的数据集为video.mpy


2020.6.5 完成了video retrieval
query为单个视频的numpy  refer为多个视频的numpy
最终的到重复的部位与对应视频

运行video_retrieval即可