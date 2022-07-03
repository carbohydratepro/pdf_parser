# text_matcher.py

import sys
import numpy as np
import cv2
import re  # 正規表現
import os
import time
import itertools
import json
from itertools import *
from statistics import mean
from functools import reduce
from bs4 import BeautifulSoup as bs


class TextMatcher:

    @classmethod
    def parse_nodes(cls, html):
        """
        htmlを受け取り、pタグのx,y座標とテキストをリストにして返す
        """

        doc = bs(html, "html.parser")

        def format_node(node):
            y = int(re.findall(r'(?<=top:)-?\d+(?=px)', node["style"])[0])
            x = int(re.findall(r'(?<=left:)-?\d+(?=px)', node["style"])[0])
            return {'text': node.text, 'y': y, 'x': x}
        return [format_node(n) for n in doc.select("p[style]")]

    @classmethod
    def combine_text(cls, table, nodes):
        """
        引数のテーブルと同じ大きさの、html内テキストとセルを照らし合わせたテーブルを返す
        要素が文字列の二次元配列を返す
        """

        text_table = [['' for _ in range(len(table[0]))]
                      for _ in range(len(table))]
        for rowi, coli in itertools.product(range(len(table)), range(len(table[0]))):
            cell = table[rowi][coli]
            if cell is None:
                continue

            def is_inner_cell(node):
                return cell['unit_position']['x'] <= node['x'] < cell['unit_position']['x'] + cell['unit_position']['w'] and cell['unit_position']['y'] <= node['y'] < cell['unit_position']['y'] + cell['unit_position']['h']
            for inner_cell in filter(is_inner_cell, nodes):
                text_table[rowi][coli] += inner_cell['text'] + "\n"
            text_table[rowi][coli] = text_table[rowi][coli].rstrip()
        return text_table

    @classmethod
    def cast_text(cls, table, text_table):
        """
        結合されたセルのテキストを、一旦マージし、それを結合されたセルに再配分したテーブルを返す
        要素が文字列の二次元配列を返す
        """

        merged_table = [['' for _ in range(len(table[0]))]
                        for _ in range(len(table))]
        for rowi, coli in itertools.product(range(len(table)), range(len(table[0]))):
            cell = table[rowi][coli]
            text = text_table[rowi][coli]
            if cell is None:
                continue
            merged_texts = [text]
            for ri, ci in itertools.product(range(len(table)), range(len(table[0]))):
                target_cell = table[ri][ci]
                if (target_cell is None) or (ri == rowi and ci == coli):
                    continue
                if target_cell['merged_position']['x'] == cell['merged_position']['x'] and target_cell['merged_position']['y'] == cell['merged_position']['y'] and target_cell['merged_position']['w'] == cell['merged_position']['w'] and target_cell['merged_position']['h'] == cell['merged_position']['h']:
                    merged_texts.append(text_table[ri][ci])
            for merged_text in merged_texts:
                merged_table[rowi][coli] += merged_text
        return merged_table

    @classmethod
    def create_table_document(cls, text_casted_table):
        """
        テーブルの確認のためのhtmlのtableタグを作る
        """

        html = '<table border="1">'
        for rowi in range(len(text_casted_table)):
            html += '<tr>'
            for coli in range(len(text_casted_table[0])):
                html += '<td>'
                html += text_casted_table[rowi][coli]
                html += '</td>'
            html += '</tr>'
        html += '</table>'
        return html

if __name__ == '__main__':
    pass