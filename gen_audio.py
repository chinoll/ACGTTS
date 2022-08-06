import torch
import vits.commons as commons
import vits.utils as utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from scipy.io.wavfile import write
import argparse
import vits.text as text
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--config", default="./configs/acg_base.json", type=str, help="Config file")
    parse.add_argument("-t", "--text", type=str, help="Text")
    parse.add_argument("-m", "--model", default='models/27000.pth', type=str, help="model path")
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
    net_g.load_state_dict(torch.load(args.model,map_location='cpu'))
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