# 二次元TTS
目前支持的角色 <br>
```
0 - 绫地宁宁
1 - 因幡巡
2 - 户隐憧子
```

# 用法
在使用之前请先安装espeak <br>
下载模型放入models目录
```
cd vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
pip install -r requirements.txt
python gen_audio.py --text you text --sid 0 --model models/*.pth
```

# 模型下载地址
[ACGTTS](https://huggingface.co/chinoll/ACGTTS)

# TODO
- [ ] 微调代码
- [ ] 训练效率改进
- [ ] 更简单的UI
- [ ] 公开数据集