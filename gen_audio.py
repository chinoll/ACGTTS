import torch
import vits.commons as commons
import vits.utils as utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from scipy.io.wavfile import write, read
import argparse
import vits.text as text
import os
import requests
import math
import tqdm
from pykakasi import kakasi
import numpy as np
from vits.mel_processing import spectrogram_torch
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
jpurl = 'https://huggingface.co/chinoll/ACGTTS/resolve/main/new/acg_base/jp_1196000.pth'
zhurl = 'https://huggingface.co/chinoll/ACGTTS/resolve/main/new/acg_base/zh_323000.pth'
def download_model(model_url: str, model_name:str):
    if not os.path.exists('models'):
        os.makedirs('models')
    if os.path.exists(f'models/{model_name}'):
        return f'models/{model_name}'

    response=requests.get(model_url,stream=True)
    data_size=math.ceil(int(response.headers['Content-Length'])/1024)
    with open(f'models/{model_name}','wb') as f:
        for data in tqdm.tqdm(iterable=response.iter_content(1024),total=data_size,desc=f'正在下载{model_name}',unit='KB'):
            f.write(data)
    return f'models/{model_name}'

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config", default="./configs/acg_base.json", type=str, help="Config file")
    parse.add_argument("-t", "--text", type=str, help="Text")
    parse.add_argument("-m", "--model", default=None, type=str, help="model path")
    parse.add_argument("-o", "--output", default="test.wav", type=str, help="output wav file path")
    parse.add_argument("--sid",default=0, type=int, help="SID of the character")
    parse.add_argument("--cuda",default=False, action="store_true", help="use cuda")
    parse.add_argument("--zh",default=False, action="store_true", help="use chinese language")
    parse.add_argument("--old",default=False, action="store_true", help="use old model")
    args = parse.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    net_g = net_g.cuda() if args.cuda else net_g
    _ = net_g.eval()
    if args.model == None:
        model_jp_path = download_model(jpurl,jpurl.split("/")[-1])
        if args.zh:
            model_zh_path = download_model(zhurl,zhurl.split("/")[-1])
    else:
        model_path = args.model
        model_zh_path = 'zh_323000.pth'

    net_g.load_state_dict(torch.load(model_jp_path,map_location='cpu'))
    if args.old:
        cleanned_text = text._clean_text(' '.join([i for i in list(args.text) if i != ' ']), ['transliteration_cleaners'])
    else:
        if args.zh:
            zh_netg = SynthesizerTrn(
                    len(symbols),
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    n_speakers=hps.data.n_speakers,
                    **hps.model)
            zh_netg.load_state_dict(torch.load(model_zh_path,map_location='cpu'))
            zh_netg = zh_netg.cuda() if args.cuda else zh_netg

            cleanned_text = args.text
        else:
            result = kakasi.convert(args.text)
            cleanned_text = ' '.join([i['kana'] for i in result])
        text._clean_text(cleanned_text, ['transliteration_cleaners'])

    stn_tst = get_text(cleanned_text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0) if args.cuda else stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        x_tst_lengths = x_tst_lengths.cuda() if args.cuda else x_tst_lengths
        sid = torch.LongTensor([args.sid])
        sid = sid.cuda() if args.cuda else sid
        if args.zh:
            csid = torch.LongTensor([5])
            csid = csid.cuda() if args.cuda else csid
            audio = zh_netg.infer(x_tst, x_tst_lengths, sid=csid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.5)[0][0,0]
            audio_norm = audio.unsqueeze(0)
            spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            spec_lengths = torch.LongTensor([spec.size(1)])
            spec_lengths = spec_lengths.cuda() if args.cuda else spec_lengths

            spec_padded = torch.FloatTensor(1, spec.size(0), spec.size(1))
            spec_padded.zero_()
            spec_padded[0, :, :spec.size(1)] = spec
            spec_padded = spec_padded.cuda() if args.cuda else spec_padded

            sid_src = torch.LongTensor([8])
            sid_src = sid_src.cuda() if args.cuda else sid_src
            audio = net_g.voice_conversion(spec_padded,spec_lengths,sid_src=sid_src, sid_tgt=sid)[0][0,0].data.cpu().float().numpy()
        else:
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(args.output,22050,audio)