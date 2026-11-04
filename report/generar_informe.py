r"""Genera informe.pdf desde informe.md y comprueba el límite de dos páginas.

Instalación: .venv\Scripts\python -m pip install -r report/requirements.txt
Uso desde la raíz: .venv\Scripts\python report/generar_informe.py
"""
from html import escape
from pathlib import Path
import re

import pymupdf
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

CARPETA = Path(__file__).resolve().parent
FUENTES = Path("C:/Windows/Fonts")
if (FUENTES / "arial.ttf").exists():
    regular, negrita = FUENTES / "arial.ttf", FUENTES / "arialbd.ttf"
else:
    FUENTES = Path(reportlab.__file__).parent / "fonts"
    regular, negrita = FUENTES / "Vera.ttf", FUENTES / "VeraBd.ttf"
pdfmetrics.registerFont(TTFont("Informe", str(regular)))
pdfmetrics.registerFont(TTFont("InformeBold", str(negrita)))
pdfmetrics.registerFontFamily("Informe", normal="Informe", bold="InformeBold", italic="Informe", boldItalic="InformeBold")
TINTA = colors.HexColor("#18334B")
estilos = {
    "texto": ParagraphStyle("texto", fontName="Informe", fontSize=9.5, leading=11.7, spaceAfter=5),
    "titulo": ParagraphStyle("titulo", fontName="InformeBold", fontSize=15, leading=18,
                             alignment=TA_CENTER, textColor=TINTA, spaceAfter=6),
    "seccion": ParagraphStyle("seccion", fontName="InformeBold", fontSize=10, leading=12,
                              textColor=TINTA, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    "tabla": ParagraphStyle("tabla", fontName="Informe", fontSize=8.6, leading=10.5),
}


def formato(texto):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(texto))


def pie(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#B9C4CE"))
    canvas.line(36, 29, A4[0] - 36, 29)
    canvas.setFont("Informe", 7)
    canvas.setFillColor(TINTA)
    canvas.drawString(36, 18, "Taller 2 · Aprendizaje por Refuerzo · 05/11/2026")
    canvas.drawRightString(A4[0] - 36, 18, f"{doc.page} / 2")


def generar():
    bloques = (CARPETA / "informe.md").read_text(encoding="utf-8").strip().split("\n\n")
    elementos = []
    ancho = A4[0] - 72
    for bloque in bloques:
        if bloque == "<!-- pagebreak -->":
            elementos.append(PageBreak())
        elif bloque.startswith("# "):
            elementos.append(Paragraph(formato(bloque[2:]), estilos["titulo"]))
        elif bloque.startswith("## "):
            elementos.append(Paragraph(formato(bloque[3:]), estilos["seccion"]))
        elif bloque.startswith("|"):
            filas = [l.strip("| ").split("|") for l in bloque.splitlines()
                     if not re.fullmatch(r"[|:\-\s]+", l)]
            n = len(filas[0])
            proporciones = {3: [.55, .20, .25], 4: [.46, .18, .18, .18], 5: [.28, .17, .19, .18, .18]}[n]
            datos = [[Paragraph(formato(c.strip()) if i else "<b>"+formato(c.strip())+"</b>", estilos["tabla"])
                      for c in fila] for i, fila in enumerate(filas)]
            tabla = Table(datos, colWidths=[ancho * f for f in proporciones], repeatRows=1, hAlign="LEFT")
            tabla.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EDF2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("LINEBELOW", (0, 0), (-1, 0), .6, TINTA),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elementos.extend([tabla, Spacer(1, 5)])
        else:
            elementos.append(Paragraph(formato(bloque.replace("\n", " ")), estilos["texto"]))
    temporal = CARPETA / "informe_revision.pdf"
    doc = SimpleDocTemplate(str(temporal), pagesize=A4, leftMargin=36, rightMargin=36,
                            topMargin=28, bottomMargin=37,
                            title="DQN y A3C para invertir en un activo",
                            author="Josep Peréz Segura; Emilio Sanchez Martínez; Natalia San José Ortega")
    doc.build(elementos, onFirstPage=pie, onLaterPages=pie)
    with pymupdf.open(temporal) as pdf:
        if len(pdf) != 2:
            raise ValueError(f"El borrador tiene {len(pdf)} páginas; revisar el contenido antes de reemplazar el PDF.")
        vista = CARPETA / "preview"
        vista.mkdir(exist_ok=True)
        for i, pagina in enumerate(pdf, 1):
            pagina.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(vista / f"pagina_{i}.png")
    temporal.replace(CARPETA / "informe.pdf")
    print("Generado report/informe.pdf: dos páginas.")


if __name__ == "__main__":
    generar()
