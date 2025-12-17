# aurainsight_typography.py

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_aurainsight_fonts():
    pdfmetrics.registerFont(
        TTFont("Noto", "assets/fonts/NotoSansSC-Regular.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Noto-Bold", "assets/fonts/NotoSansSC-Bold.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Roboto", "assets/fonts/Roboto-Regular.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Roboto-Bold", "assets/fonts/Roboto-Bold.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Roboto-Italic", "assets/fonts/Roboto-Italic.ttf")
    )
