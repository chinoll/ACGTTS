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


jpurl = "https://huggingface.co/chinoll/ACGTTS/resolve/main/new/acg_base/jp_1196000.pth"
zhurl = "https://huggingface.co/chinoll/ACGTTS/resolve/main/new/acg_base/zh_323000.pth"


def download_model(model_url: str, model_name: str):
    if not os.path.exists("models"):
        os.makedirs("models")
    if os.path.exists(f"models/{model_name}"):
        return f"models/{model_name}"

    response = requests.get(model_url, stream=True)
    data_size = math.ceil(int(response.headers["Content-Length"]) / 1024)
    with open(f"models/{model_name}", "wb") as f:
        for data in tqdm.tqdm(
            iterable=response.iter_content(1024),
            total=data_size,
            desc=f"正在下载{model_name}",
            unit="KB",
        ):
            f.write(data)
    return f"models/{model_name}"


def generate_audio(text, output_path, net_g, hps, sid=0, use_cuda=False):
    kks = kakasi()
    result = kks.convert(text)
    cleanned_text = " ".join([i["kana"] for i in result])
    
    stn_tst = get_text(cleanned_text, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0) if use_cuda else stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        x_tst_lengths = x_tst_lengths.cuda() if use_cuda else x_tst_lengths
        sid_tensor = torch.LongTensor([sid])
        sid_tensor = sid_tensor.cuda() if use_cuda else sid_tensor

        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid_tensor,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    write(output_path, 22050, audio)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-c",
        "--config",
        default="./configs/acg_base.json",
        type=str,
        help="Config file",
    )
    parse.add_argument("-t", "--text", type=str, help="Single text to generate")
    parse.add_argument("-b", "--batch", type=str, help="Text file for batch processing")
    parse.add_argument("-m", "--model", default=None, type=str, help="model path")
    parse.add_argument(
        "-o", "--output", default="output.wav", type=str, help="output wav file path"
    )
    parse.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="output directory for batch processing",
    )
    parse.add_argument("--sid", default=0, type=int, help="SID of the character")
    parse.add_argument("--cuda", default=False, action="store_true", help="use cuda")
    parse.add_argument(
        "--zh", default=False, action="store_true", help="use chinese language"
    )
    parse.add_argument(
        "--old", default=False, action="store_true", help="use old model"
    )

    args = parse.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    net_g = net_g.cuda() if args.cuda else net_g
    _ = net_g.eval()

    # Download or load model
    if args.model is None:
        model_path = download_model(jpurl, jpurl.split("/")[-1])
    else:
        model_path = args.model
    net_g.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Process single text or batch
    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

        for i, text in enumerate(texts):
            output_path = os.path.join(args.output_dir, f"output_{i+1}.wav")
            generate_audio(text, output_path, net_g, hps, args.sid, args.cuda)
            print(f"Generated: {output_path}")
    else:
        if not args.text:
            print("Error: Either --text or --batch must be specified")
            exit(1)
        generate_audio(args.text, args.output, net_g, hps, args.sid, args.cuda)
        print(f"Generated: {args.output}")
