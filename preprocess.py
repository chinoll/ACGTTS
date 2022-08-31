from pykakasi import kakasi
from vits.utils import load_filepaths_and_text
import vits.text as text
import argparse

clearlist = ['%','... ... ...','(',')','[',']','\ n','-','~','"','*','&','... ...','●','・','♪','“','… … …','{','}','+','/','>']
replacedict = {'1':'iti','2':'ni','3':'san','4':'si','5':'go','6':'roku','7':'siti','8':'hati','9':'kyujuu','0':'jyuu','…':' ','^':' '}
def filter_text(text):
    for x in replacedict:
        text = text.replace(x,replacedict[x])
    for j in clearlist:
        text = text.replace(j,'')
    return text

def filter_NSFW_audio(text):
    filter_word = ['精液','乳首','子宮','媚薬','イく','はっ']
    filter_count = [1,1,1,1,1,3]
    for j in range(len(filter_word)):
        if text.count(filter_word[j]) >= filter_count[j]:
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--filelists", nargs="+", default=['audio.txt'])
    parser.add_argument("--zh", default=False,action="store_true",help="change language to zh")
    parser.add_argument("--old",default=False, action="store_true")
    args = parser.parse_args()
    kakasi = kakasi()

    for filelist in args.filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][2]
            if not args.zh:
                if filter_NSFW_audio(original_text):
                    original_text = ''
                if args.old:
                    result = kakasi.convert(' '.join(list(original_text.replace("「","").replace("」",""))))
                else:
                    result = kakasi.convert(''.join(list(original_text.replace("「","").replace("」",""))))
            
                original_text = ' '.join([i['kana'] for i in result])
            cleaned_text = text._clean_text(original_text, ["transliteration_cleaners"])
            cleaned_text = filter_text(cleaned_text)

            if not args.zh:
                if len(cleaned_text.replace(" ","")) == 0:
                    cleaned_text = ''
                while cleaned_text.find('  ') != -1:
                    cleaned_text = cleaned_text.replace("  "," ")
                try:
                    cleaned_text = cleaned_text if cleaned_text[0] != ' ' else cleaned_text[1:]
                except:
                    cleaned_text = ''

            if cleaned_text != '':
                z = cleaned_text
                for j in ['.','?','!',',']:
                    z = z.replace(j,"")
                if len(z) == 0:
                    cleaned_text = ''

            filepaths_and_text[i][2] = cleaned_text

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text if x[2] != ""])