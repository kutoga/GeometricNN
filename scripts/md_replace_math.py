import re
import sys
from urllib.parse import quote

latex_math_regex = re.compile(r'\$(?P<math_expr>[^\$]*)\$')
image_type = 'png'
base_url = 'http://latex.codecogs.com/{image_type}.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20{math_exp}'

def replace_latex_math(match) -> str:
    math_expr = match.groups('math_expr')[0]
    img_url = base_url.format(
        math_exp=quote(math_expr),
        image_type=quote(image_type))
    markdown_img = f'![mathematicla expression]({img_url})'
    return markdown_img

for line in sys.stdin:
    print(latex_math_regex.sub(replace_latex_math, line), end='')

