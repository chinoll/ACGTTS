import torch
import vits.commons as commons
import vits.utils as utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from scipy.io.wavfile import write
import argparse
import vits.text as text
import os
import requests
import math
import tqdm

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def download_model(model_url='https://huggingface.co/chinoll/ACGTTS/resolve/main/acg_base/38000.pth',model_name='38000.pth'):
    if not os.path.exists('models'):
        os.makedirs('models')
    if os.path.exists(f'models/{model_name}'):
        return f'models/{model_name}'

    response=requests.get(model_url,stream=True)
    data_size=math.ceil(int(response.headers['Content-Length'])/1024)
    with open(f'models/{model_name}','wb') as f:
        for data in tqdm.tqdm(iterable=response.iter_content(1024),total=data_size,desc='正在下载',unit='KB'):
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
        model_path = download_model()
    else:
        model_path = args.model

    net_g.load_state_dict(torch.load(model_path,map_location='cpu'))
    cleanned_text = text._clean_text(' '.join([i for i in list(args.text) if i != ' ']), ['transliteration_cleaners'])
    stn_tst = get_text(cleanned_text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0) if args.cuda else stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        x_tst_lengths = x_tst_lengths.cuda() if args.cuda else x_tst_lengths
        sid = torch.LongTensor([args.sid])
        sid = sid.cuda() if args.cuda else sid
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(args.output,22050,audio)