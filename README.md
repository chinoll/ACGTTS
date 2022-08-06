# 二次元TTS
目前支持的角色 <br>
```
0 - 绫地宁宁
1 - 因幡巡
2 - 户隐憧子
```

# 用法
在使用之前请先安装espeak
去下载模型放入models目录
```
pip install -r requirements.txt
python gen_audio.py --text you text --sid 0 --model models/*.pth
```

# 模型下载地址
[huggingface](https://huggingface.co/chinoll/ACGTTS)

# 微调
TODO