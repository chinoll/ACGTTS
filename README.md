# 二次元TTS
目前支持的角色 <br>
```
0 - 绫地宁宁
1 - 户隐憧子
2 - 因幡巡
3 - 明月栞那
4 - 四季夏目
5 - 墨染希
6 - 火打谷爱衣
7 - 汐山凉音
8 - 中文注入声线
9 - 二条院羽月
10 - 在原七海
11 - 式部茉优
12 - 三司绫濑
13 - 壬生千咲
14 - 朝武芳乃
15 - 常陆茉子
16 - 丛雨
17 - 蕾娜·列支敦瑙尔
18 - 鞍马小春
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
- [x] 中文发音(方言味浓重)
- [ ] 标准普通话发音支持
# 预计支持的角色
- [ ] ~~墨小菊~~
- [ ] ~~文芷~~
- [x] 朝武芳乃
- [x] 丛雨
- [x] 常陆茉子
- [x] 三司绫濑
- [x] 在原七海
- [x] 式部茉优
- [x] 二条院羽月
- [ ] ATRI