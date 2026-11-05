"""Genera el informe de dos páginas con la figura de operaciones, sin pie de página."""
import re
import shutil
from copy import deepcopy

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

from generar_informe import CARPETA, estilos, formato, TINTA


def generar():
    ancho = A4[0] - 64
    estilos['texto'].fontSize = 9
    estilos['texto'].leading = 10.6
    estilos['texto'].spaceAfter = 4
    estilos['tabla'].fontSize = 8
    estilos['tabla'].leading = 9.4
    elementos = []
    for bloque in (CARPETA / 'informe_v2.md').read_text(encoding='utf-8').strip().split('\n\n'):
        if bloque == '<!-- pagebreak -->':
            elementos.append(PageBreak())
            # La segunda página permite un cuerpo mayor sin comprimir la figura.
            estilos['texto'] = deepcopy(estilos['texto'])
            estilos['texto'].fontSize = 10
            estilos['texto'].leading = 12.4
            estilos['texto'].spaceAfter = 6
        elif bloque.startswith('!['):
            ruta = CARPETA / re.search(r'\]\((.*?)\)', bloque).group(1)
            figura = Image(str(ruta))
            figura.drawHeight *= ancho / figura.drawWidth
            figura.drawWidth = ancho
            elementos.append(figura)
        elif bloque.startswith('# '):
            elementos.append(Paragraph(formato(bloque[2:]), estilos['titulo']))
        elif bloque.startswith('## '):
            elementos.append(Paragraph(formato(bloque[3:]), estilos['seccion']))
        elif bloque.startswith('|'):
            filas = [l.strip('| ').split('|') for l in bloque.splitlines()
                     if not re.fullmatch(r'[|:\-\s]+', l)]
            proporciones = {6: [0.19, 0.18, 0.11, 0.13, 0.12, 0.27],
                            3: [0.44, 0.28, 0.28]}[len(filas[0])]
            # Las seis columnas conservan todas las filas y métricas del original.
            estilo_tabla = deepcopy(estilos['tabla'])
            if len(filas[0]) == 6:
                estilo_tabla.fontSize = 7.5
                estilo_tabla.leading = 9
            tabla = Table([[Paragraph(formato(c.strip()), estilo_tabla) for c in fila] for fila in filas],
                          colWidths=[ancho * f for f in proporciones])
            tabla.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7EDF2')),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, TINTA),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            elementos.extend([tabla, Spacer(1, 4)])
        else:
            elementos.append(Paragraph(formato(bloque), estilos['texto']))
    temporal = CARPETA / 'informe_v2_revision.pdf'
    SimpleDocTemplate(str(temporal), pagesize=A4, leftMargin=32, rightMargin=32,
                      topMargin=24, bottomMargin=24, title='DQN y A3C: resultados y operaciones',
                      author='Josep Peréz Segura; Emilio Sanchez Martínez; Natalia San José Ortega').build(elementos)
    with pymupdf.open(temporal) as pdf:
        if len(pdf) != 2:
            raise ValueError(f'El informe tiene {len(pdf)} páginas; no se reemplaza el original.')
        assert pdf[0].get_images(), 'Falta la figura'
        vista = CARPETA / 'preview'
        vista.mkdir(exist_ok=True)
        for i, pagina in enumerate(pdf, 1):
            pagina.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(vista / f'informe_v2_pagina_{i}.png')
    destino = CARPETA / 'informe_v2.pdf'
    respaldo = CARPETA / 'informe_v2_sin_grafico.pdf'
    if destino.exists() and not respaldo.exists():
        shutil.copy2(destino, respaldo)
    temporal.replace(destino)
    print('Generado informe_v2.pdf: dos páginas, gráfico incluido y sin pie de página.')


if __name__ == '__main__':
    generar()
