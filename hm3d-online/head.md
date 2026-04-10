hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlm.py
测试> 那么，我现在有一个想法，设计一个VLM模块，将当前环视的RGB进行多帧拼接，12分成3组，每4个拼接成一个图片，把这几张图片喂给VLM让他查询当前的环视是否具有目标描述的物体或者相应特征，如果有的话，强制进行target 查询。

