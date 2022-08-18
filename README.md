# 二次元TTS
目前支持的角色 <br>
```
0 - 绫地宁宁
1 - 因幡巡
2 - 户隐憧子
3 - 明月栞那
4 - 四季夏目
5 - 墨染希
6 - 火打谷爱衣
7 - 汐山凉音
8 - 喵爷
9 - 奶糖
10 - 兔牙
11 - 小青
12 - 一龙(男)
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
- [x] 中文发音(发音模糊)
# 预计支持的角色
- [ ] 墨小菊
- [ ] 文芷
- [ ] 朝武芳乃
- [ ] 丛雨
- [ ] 常陆茉子
- [ ] 三司绫濑
- [ ] 在原七海
- [ ] 式部茉优
- [ ] 二条院羽月