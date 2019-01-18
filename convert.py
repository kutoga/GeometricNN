import re
import sys
from urllib.parse import quote

latex_math_regex = re.compile(r'\$(?P<math_expr>[^\$]*)\$')
image_type = 'png'
font_size = 24
base_url = 'http://www.sciweavers.org/tex2img.php?eq={math_exp}&bc=White&fc=Black&im={image_type}&fs={font_size}&ff=arev&edit=0'

def replace_latex_math(match) -> str:
    math_expr = match.groups('math_expr')
    img_url = base_url.format(
        math_exp=quote(math_expr),
        image_type=quote(image_type),
        font_size=quote(str(font_size)))
    markdown_img = f'![mathematicla expression]({img_url})'
    return markdown_img

for line in sys.stdin:
    print(latex_math_regex.sub(replace_latex_math, line))

