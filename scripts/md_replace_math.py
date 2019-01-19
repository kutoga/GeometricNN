import re
import os
import sys
import argparse
import hashlib
import pathlib
import urllib.request
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', type=str, default=None, nargs='?', help='The output image directory.')
args = parser.parse_args()

image_directory = args.image_directory
latex_math_regex = re.compile(r'\$(?P<math_expr>[^\$]*)\$')
image_type = 'svg'
base_url = 'http://latex.codecogs.com/{image_type}.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20{math_exp}'

if image_directory is not None:
    pathlib.Path(image_directory).mkdir(parents=True, exist_ok=True)

def replace_latex_math(match) -> str:
    math_expr = match.groups('math_expr')[0]
    img_url = base_url.format(
        math_exp=quote(math_expr),
        image_type=quote(image_type))
    if image_directory is None:
        markdown_img = f'![mathematical expression]({img_url})'
    else:
        img = urllib.request.urlopen(img_url).read()
        m = hashlib.md5()
        m.update(img)
        img_file = os.path.join(image_directory, f'{m.hexdigest()}.{image_type}')
        with open(img_file, 'wb') as fh:
            fh.write(img)
        markdown_img = f'![mathematical expression]({img_file})'
    return markdown_img

for line in sys.stdin:
    print(latex_math_regex.sub(replace_latex_math, line), end='')

