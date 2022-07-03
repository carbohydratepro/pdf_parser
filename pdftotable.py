# pdftotable.py

import sys
import re
import os
import time
import itertools
import json
import subprocess
import tempfile
import codecs
from pathlib import Path, PurePath
from itertools import *
from statistics import mean
from functools import reduce
from argparse import ArgumentParser

import numpy as np
import cv2

from image_parse import ImageParse
from text_matcher import TextMatcher

class PDFtoHTMLError(Exception):
    pass



argparser = ArgumentParser()
argparser.add_argument('file_name')
argparser.add_argument('--silent', action='store_true')
argparser.add_argument('--location', action='store_true')
argparser.add_argument('--image', action='store_true')
argparser.add_argument('--html', action='store_true')
argparser.add_argument('--page')
args = argparser.parse_args()

if args.page:
    skip_pages = [int(p) for p in args.page.split(',')]
pdf_file_path = args.file_name
base_name = os.path.basename(pdf_file_path).split('.')[0]
if (os.path.dirname(pdf_file_path)) == '':
    base_dir = os.path.dirname(pdf_file_path)
    base_dir = '.'

with tempfile.TemporaryDirectory() as tmp_dir_path:
    tmp_pdf_path = tmp_dir_path + '/table.pdf'
    cp_result = subprocess.run(
        ['cp', pdf_file_path, tmp_pdf_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pdftohtml_result = subprocess.run(
        ['pdftohtml', '-c', tmp_pdf_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if pdftohtml_result.returncode != 0:
        raise PDFtoHTMLError(pdftohtml_result.stderr.decode())

    html_paths = sorted([str(p) for p in Path(tmp_dir_path).glob('table-*.html') if re.match(
        r".*table-\d+\.html$", str(p))], key=lambda p: int(re.findall(r"\d+(?=.html)", p)[0]))
    image_paths = sorted([str(p) for p in Path(tmp_dir_path).glob('table*.png') if re.match(
        r".*table\d{3}\.png$", str(p))], key=lambda p: int(re.findall(r"\d+(?=.png)", p)[0]))

    for html_path, image_path in zip(html_paths, image_paths):
        page_number = int(re.findall(r"\d+(?=.html)", html_path)[0])
        if args.page and not (page_number in skip_pages):
            continue
        if not args.silent:
            print(f"page: {str(page_number)}/{str(len(html_paths))}")

        image = cv2.imread(image_path)
        colored_line_image = Parser.color_line(image)
        colored_corner_image = Parser.color_corner(colored_line_image)
        grouped_corner_pixels = Parser.group_corner_pixels(
            colored_corner_image)
        colored_corner_spot_image = Parser.color_only_small_corner(
            colored_corner_image, grouped_corner_pixels)
        colored_emph_line_image = Parser.color_emphasized_line(
            colored_line_image)
        corner_areas = Parser.format_corner_pixels(grouped_corner_pixels)
        cells = Parser.extruct_cells(
            colored_corner_spot_image, colored_emph_line_image, corner_areas)
        colored_diagonal_image = Parser.color_cells(
            colored_corner_spot_image, cells)
        tables = Parser.merge_cells_to_tables(cells)
        tables_with_size = [Parser.guess_table_size(table) for table in tables]
        colored_table_size_image = Parser.color_table_size(
            colored_diagonal_image, tables_with_size)
        formatted_tables = [Parser.parse_table_to_array2d(
            table) for table in tables_with_size if (Parser.parse_table_to_array2d(table))]
        output_image = Parser.color_2d_array(
            colored_table_size_image, formatted_tables)
        with codecs.open(html_path, 'r', 'utf-8', 'ignore') as html_file:
            html = html_file.read()
        nodes = TextMatcher.parse_nodes(html)
        text_combined_tables = [TextMatcher.combine_text(
            table, nodes) for table in formatted_tables]
        text_casted_tables = [TextMatcher.cast_text(
            table, text_table) for table, text_table in zip(formatted_tables, text_combined_tables)]
        text_table_html = '<head><meta charset="UTF-8" /></head>' + \
            "<hr>".join([TextMatcher.create_table_document(text_casted_table)
                         for text_casted_table in text_casted_tables])

        with open(f"{base_dir}/{base_name}_{page_number}.json", "w") as json_file:
            json_file.write(json.dumps(text_casted_tables))
        if args.location:
            with open(f"{base_dir}/{base_name}_location_{page_number}.json", "w") as json_location_file:
                json_location_file.write(json.dumps(formatted_tables))
        if args.image:
            cv2.imwrite(
                f"{base_dir}/{base_name}_image_{page_number}.png", output_image)
        if args.html:
            with open(f"{base_dir}/{base_name}_html_{page_number}.html", "w") as html_file:
                html_file.write(text_table_html)
