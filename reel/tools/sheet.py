"""Contact sheet: python3 tools/sheet.py <dir> <out.png> [cols] [thumb_w] [start] [count]"""
import sys, glob, os
from PIL import Image, ImageDraw
d, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
tw = int(sys.argv[4]) if len(sys.argv) > 4 else 240
start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
count = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
fs = sorted(glob.glob(os.path.join(d, '*.png')))[start:start + count]
th = int(tw * 1920 / 1080)
rows = (len(fs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * (th + 22)), (40, 40, 40))
dr = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    im = Image.open(f).convert('RGB').resize((tw, th), Image.LANCZOS)
    x, y = (i % cols) * tw, (i // cols) * (th + 22)
    sheet.paste(im, (x, y + 22))
    t = float(os.path.basename(f)[1:-4])
    dr.text((x + 4, y + 4), f"t={t:.2f}  b{i + start}", fill=(255, 255, 0))
sheet.save(out)
print(out, sheet.size)
