# parser.py

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


class ImageParse:

    @classmethod
    def color_line(cls, image):
        """
        画像(何でも良い)を受け取り、白黒に二値化したのち、垂直・水平な線のみを抽出してそれを返す
        """
        BLACK_WHITE_BINARY_THRESHOLD = 150  # 白黒への二値化処理の閾値(これよりも黒いと真っ黒になる)
        VERTICAL_KERNEL_SIZE = 20  # 水平方向の線のみを抽出するときの、抽出する横線の長さの最小値
        HORIZONAL_KERNEL_SIZE = 20  # 垂直方向の線のみを抽出するときの、抽出する縦線の長さの最小値

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # (元画像, 白黒閾値, ...) -> 白か黒のみに二値化された画像
        _ret, bw_image = cv2.threshold(
            gray_image, BLACK_WHITE_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, VERTICAL_KERNEL_SIZE))
        vertical_lines_pre = cv2.erode(bw_image, vertical_kernel, iterations=1)
        vertical_lines = cv2.dilate(
            vertical_lines_pre, vertical_kernel, iterations=1)
        horizonal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (HORIZONAL_KERNEL_SIZE, 1))
        horizontal_lines_pre = cv2.erode(
            bw_image, horizonal_kernel, iterations=1)
        horizontal_lines = cv2.dilate(
            horizontal_lines_pre, horizonal_kernel, iterations=1)
        merged_lines = cv2.addWeighted(
            vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # 二値化した白黒をマージした画像を黒くしたいので、第2引数は1以上の小さい数字なら何でも良い
        _ret, lined_image = cv2.threshold(
            merged_lines, 10, 255, cv2.THRESH_BINARY_INV)
        return lined_image

    @classmethod
    def color_corner(cls, image):
        """
        画像(何でも良い)を受け取り、グレースケール化したのちコーナー検出をする
        検出したコーナーに色を付けたグレースケール画像を返す
        """
        HARRIS_BLOCK_SIZE = 5  # ハリスのコーナー検出で塗られるコーナーの大きさ

        gray_image = image
        gray_image_f32 = np.float32(gray_image)  # uint8からfloat32に変換
        converted_image_raw = cv2.cornerHarris(
            gray_image_f32, HARRIS_BLOCK_SIZE, 3, 0.04)  # グレースケール画像が作られる
        converted_image = cv2.dilate(converted_image_raw, None)  # 白色領域を膨張
        gray_image_bgr = cv2.cvtColor(
            gray_image, cv2.COLOR_GRAY2BGR)  # カラーの灰色画像に変換
        # 配列内の値(白黒度合い)の最大値 * 0.01 よりも大きい範囲
        gray_image_bgr[converted_image > 0.01 *
                       converted_image.max()] = [0, 0, 255]
        return gray_image_bgr

    @classmethod
    def color_emphasized_line(cls, image):
        """
        白黒画像を受け取り、黒い領域を膨張させたものを返す
        """
        EMPHASIZE_LEVEL = 5  # 膨張処理の回数

        lined_image_emph = cv2.erode(image, None, iterations=EMPHASIZE_LEVEL)
        return lined_image_emph

    @classmethod
    def group_corner_pixels(cls, corner_colored_image):
        """
        コーナーに赤く(0,0,255)色付けされた画像を受け取り、隣接したコーナーの画素をグループ化したものを返す
        具体的には3次元配列を返す(グループ複数 > グループ > x,y座標)
        """
        height = len(corner_colored_image)
        width = len(corner_colored_image[0])
        image = [[{'corner': False, 'used': False}
                  for w in range(width)] for h in range(height)]
        for y, x in itertools.product(range(height), range(width)):
            blue, green, red = corner_colored_image[y][x]
            is_corner = (blue == 0 and green == 0 and red == 255)
            image[y][x]['corner'] = is_corner
        corner_areas = []
        for y, x in itertools.product(range(height), range(width)):
            if (not image[y][x]['corner']) or image[y][x]['used']:
                continue
            stack = [[y, x]]
            area = []
            image[y][x]['used'] = True
            while len(stack) != 0:
                point = stack.pop()
                h, w = point
                area.append(point)
                for hh, ww in [[h+1, w], [h-1, w], [h, w+1], [h, w-1]]:
                    if hh < 0 or height <= hh or ww < 0 or width <= ww:
                        continue
                    if image[hh][ww]['corner'] and not image[hh][ww]['used']:
                        image[hh][ww]['used'] = True
                        stack.append([hh, ww])
            corner_areas.append(area)
        return corner_areas

    @classmethod
    def color_only_small_corner(cls, image, corner_areas):
        """
        画像(中身は何でも良い)とグループ化されたコーナーを受け取り、
        その画像について、サイズが大きいグループに入っている座標の画素を青く塗ったものを返す
        """
        CORNER_SIZE_LIMIT = 300  # 画素がこのサイズ以上のコーナーを除外する

        output_image = image.copy()
        corner_spot_areas = filter(lambda a: len(
            a) > CORNER_SIZE_LIMIT, corner_areas)
        for area in corner_spot_areas:
            for pixel in area:
                y = pixel[0]
                x = pixel[1]
                # 広すぎるコーナーは文字の可能性が高いので除外(青く色づけ)
                output_image[y][x] = [255, 0, 0]
        return output_image

    @classmethod
    def format_corner_pixels(cls, grouped_corner_pixels):
        """
        グループ化されたコーナーを受け取り、そのグループのx,y座標平均と画素数を計算し、辞書の配列に整形して返す
        """

        corner_areas = []
        for corner_pixels in grouped_corner_pixels:
            x = int(mean(map(lambda a: a[0], corner_pixels)))
            y = int(mean(map(lambda a: a[1], corner_pixels)))
            length = len(corner_pixels)
            corner_areas.append({'x': x, 'y': y, 'length ': length})
        return corner_areas

    @classmethod
    def extruct_cells(cls, corner_image, line_image, corner_areas):
        """
        (コーナーが赤く色付けされた画像, 罫線が黒く色付けされた画像, 整形済みのグループ化されたコーナー)を受け取る
        罫線で繋がっているコーナー4個を、セル1個としてまとめ、それを返す
        具体的には x,y,w,h がキーの辞書の配列を返す
        """
        POSITION_CORRECTION = 5  # 下(右)に移動してコーナーに触れたときに、もう少し下(右)に移動する補正ピクセル数
        # 下(右)に移動してコーナーに触れたときに、その右(下)に罫線があるか判定するときにどれくらい右(下)を見るか
        LINE_CHECK_SIZE = 10

        cells = []
        for area in corner_areas:
            mean_y = area['x']
            mean_x = area['y']
            nearest_bottom_corner_y = None
            nearest_right_corner_x = None
            not_found = False  # 画像外に一度でも行ったらセルは無く、IndexErrorが起きてしまうので、そのフラグ
            # 下直近のコーナーを探す
            pos_y = mean_y
            pos_x = mean_x
            while (corner_image[pos_y][pos_x] == [0, 0, 255]).all():
                pos_y += 1  # 自身の赤い点から下へ抜ける
                if pos_y >= len(line_image):
                    not_found = True
                    break
            while (not not_found) and (line_image[pos_y][pos_x] == [0, 0, 0]).all():
                pos_y += 1  # テーブルの黒い線上の間、下へ動く
                if pos_y + POSITION_CORRECTION >= len(line_image) or pos_x + LINE_CHECK_SIZE >= len(line_image[0]):
                    not_found = True
                    break
                # 別のコーナーに当たったら、少し右を見て、右が罫線上ならば位置を取得(余裕を持たせるため少し位置を足す)
                if (corner_image[pos_y][pos_x] == [0, 0, 255]).all() and (line_image[pos_y + POSITION_CORRECTION][pos_x + LINE_CHECK_SIZE] == [0, 0, 0]).all():
                    nearest_bottom_corner_y = pos_y + POSITION_CORRECTION
                    break
            # 右直近のコーナーを探す
            pos_y = mean_y
            pos_x = mean_x
            while (not not_found) and (corner_image[pos_y][pos_x] == [0, 0, 255]).all():
                pos_x += 1  # 自身の赤い点から右へ抜ける
                if pos_x >= len(line_image[0]):
                    not_found = True
                    break
            while (not not_found) and (line_image[pos_y][pos_x] == [0, 0, 0]).all():
                pos_x += 1  # テーブルの黒い線上の間、右へ動く
                if pos_x + POSITION_CORRECTION >= len(line_image[0]) or pos_y + LINE_CHECK_SIZE >= len(line_image):
                    not_found = True
                    break
                # 別のコーナーに当たったら、少し下を見て、下が罫線上ならば位置を取得(余裕を持たせるため少し位置を足す)
                if (corner_image[pos_y][pos_x] == [0, 0, 255]).all() and (line_image[pos_y + LINE_CHECK_SIZE][pos_x + POSITION_CORRECTION] == [0, 0, 0]).all():
                    nearest_right_corner_x = pos_x + POSITION_CORRECTION
                    break
            # 右or下に直近のコーナーが無いならそのコーナーはセルにはならない
            if not_found or nearest_bottom_corner_y is None or nearest_right_corner_x is None:
                continue
            if (corner_image[nearest_bottom_corner_y][nearest_right_corner_x] == [0, 0, 255]).all():
                # 右下もコーナーならこの4コーナーで1セル
                cell = {'x': mean_x, 'y': mean_y, 'w': nearest_right_corner_x -
                        mean_x, 'h': nearest_bottom_corner_y - mean_y}
                cells.append(cell)
        return cells

    @classmethod
    def color_cells(cls, image, cells):
        """
        画像(何でも良い)とセルの配列を受け取り、セルの対角線を緑色に塗った画像を返す
        """
        colored_diagonal_image = image.copy()
        for cell in cells:
            cv2.line(colored_diagonal_image, (cell['x'], cell['y']), (
                cell['x'] + cell['w'], cell['y'] + cell['h']), (0, 255, 0), 3)
        return colored_diagonal_image

    @classmethod
    def merge_cells_to_tables(cls, raw_cells):
        """
        セルの配列を受け取り、隣接したセルでグループ化したものをテーブル1個としてまとめ、まとめられたテーブルの配列を返す
        """

        tables = []
        cells = [{**cell, 'used': False} for cell in raw_cells]
        for start_cell in cells:
            if start_cell['used']:
                continue
            table = []
            stack = [start_cell]
            start_cell['used'] = True
            while len(stack) != 0:
                cell = stack.pop()
                table.append(
                    {'x': cell['x'], 'y': cell['y'], 'w': cell['w'], 'h': cell['h']})

                def is_unused_and_neighbor(target_cell):  # filterのための一時関数
                    distance_x = abs(
                        (cell['x'] + cell['w']/2) - (target_cell['x'] + target_cell['w']/2))
                    half_w_sum = cell['w']/2 + target_cell['w']/2
                    is_overlap_x = distance_x < half_w_sum
                    distance_y = abs(
                        (cell['y'] + cell['h']/2) - (target_cell['y'] + target_cell['h']/2))
                    half_h_sum = cell['h']/2 + target_cell['h']/2
                    is_overlap_y = distance_y < half_h_sum
                    return is_overlap_x and is_overlap_y and not target_cell['used']

                unused_and_neighbor_cells = filter(
                    is_unused_and_neighbor, cells)

                for unused_and_neighbor_cell in unused_and_neighbor_cells:
                    stack.append(unused_and_neighbor_cell)
                    unused_and_neighbor_cell['used'] = True
            tables.append(table)
        return tables

    @classmethod
    def guess_table_size(cls, cells):
        """
        セルの配列(=テーブル)を受け取る
        テーブルの行と列の位置と数を計算してそれを返す
        """
        ACCEPTABLE_NEIGHBOUR_RANGE = 5  # 隣あう行(列)の座標の許容画素数. 例えば2を指定すると2画素分の行の位置のギャップは無視して同一行とみなしてくれる

        row_lines = []
        col_lines = []
        height = max(map(lambda r: r['y'] + r['h'], cells)
                     ) + ACCEPTABLE_NEIGHBOUR_RANGE * 10
        width = max(map(lambda r: r['x'] + r['w'], cells)
                    ) + ACCEPTABLE_NEIGHBOUR_RANGE * 10
        blank = np.zeros((height, width)) + 255
        for cell in cells:
            xl = cell['x']
            blank[10][(xl - ACCEPTABLE_NEIGHBOUR_RANGE)                      :(xl + ACCEPTABLE_NEIGHBOUR_RANGE)] = 0
            xr = cell['x'] + cell['w']
            blank[10][(xr - ACCEPTABLE_NEIGHBOUR_RANGE)                      :(xr + ACCEPTABLE_NEIGHBOUR_RANGE)] = 0
            yt = cell['y']
            yb = cell['y'] + cell['h']
            for y in range(yt - ACCEPTABLE_NEIGHBOUR_RANGE, yt + ACCEPTABLE_NEIGHBOUR_RANGE):
                blank[y][10] = 0
            for y in range(yb - ACCEPTABLE_NEIGHBOUR_RANGE, yb + ACCEPTABLE_NEIGHBOUR_RANGE):
                blank[y][10] = 0
        # 列情報の検出
        col_cnt = 0
        pos_x = 0
        vertical_line_begin = 0
        vertical_line_end = 0
        while pos_x < width - 1:
            pos_x += 1
            if blank[10][pos_x] == 0:
                col_cnt += 1  # 黒に当たったら列1つ分
                vertical_line_begin = pos_x
                while blank[10][pos_x] == 0:
                    pos_x += 1  # 当たった列を抜ける
                    vertical_line_end = pos_x
                col_lines.append(
                    (vertical_line_begin + vertical_line_end) // 2)
        if col_cnt > 0:
            col_cnt -= 1
        # 行情報の検出
        row_cnt = 0
        pos_y = 0
        horizonal_line_begin = 0
        horizonal_line_end = 0
        while pos_y < height - 1:
            pos_y += 1
            if blank[pos_y][10] == 0:
                row_cnt += 1  # 黒に当たったら行1つ分
                horizonal_line_begin = pos_y
                while blank[pos_y][10] == 0:
                    pos_y += 1  # 当たった行を抜ける
                    horizonal_line_end = pos_y
                row_lines.append(
                    (horizonal_line_begin + horizonal_line_end) // 2)
        if row_cnt > 0:
            row_cnt -= 1  # 罫線N本でN-1行なので
        return {'cells': cells, 'col_cnt': col_cnt, 'row_cnt': row_cnt, 'row_lines': row_lines, 'col_lines': col_lines}

    @classmethod
    def color_table_size(cls, image, tables_with_size):
        """
        画像(何でも良い)と行列情報付与済みのテーブルを受け取り、
        テーブルの行数、列数、それらの位置を書き込んだ画像を返す
        """

        output_image = image.copy()
        for table_with_size in tables_with_size:
            cells = table_with_size['cells']
            first_cell = cells[0]
            cv2.putText(output_image, "row:{}, col:{}".format(table_with_size['row_cnt'], table_with_size['col_cnt']), (
                first_cell['x'], first_cell['y']), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 0))
            for row_line in table_with_size['row_lines']:
                cv2.line(output_image, (10, row_line),
                         (20, row_line), (255, 0, 255), 3)
            for col_line in table_with_size['col_lines']:
                cv2.line(output_image, (col_line, 10),
                         (col_line, 20), (255, 0, 255), 3)
        return output_image

    @classmethod
    def parse_table_to_array2d(cls, table):
        """
        テーブルを受け取り、2次元配列に変換して返す
        行と列の座標の情報から、セルの2次元配列におけるインデックスを推定している
        """
        ACCEPTABLE_NEIGHBOUR_RANGE = 10  # 線の座標とセルの座標の、重なりを判定するときの許容ピクセル数

        cells = table['cells']
        row_cnt = table['row_cnt']
        col_cnt = table['col_cnt']
        if row_cnt == 0 or col_cnt == 0:
            return None

        row_lines = table['row_lines']
        col_lines = table['col_lines']
        array_2d = [[None for _ in range(col_cnt)] for _ in range(row_cnt)]
        for rowi12, coli12 in itertools.product(itertools.product(range(row_cnt + 1), range(row_cnt + 1)), itertools.product(range(col_cnt + 1), range(col_cnt + 1))):
            rowi1 = rowi12[0]
            rowi2 = rowi12[1]
            coli1 = coli12[0]
            coli2 = coli12[1]
            if rowi1 >= rowi2 or coli1 >= coli2:
                continue
            line_top = row_lines[rowi1]
            line_bottom = row_lines[rowi2]
            line_left = col_lines[coli1]
            line_right = col_lines[coli2]

            def is_covered(cell):  # cellの四隅が[up,down,right,left]に入っていたらTrue
                cell_top = cell['y']
                cell_bottom = cell['y'] + cell['h']
                cell_left = cell['x']
                cell_right = cell['x'] + cell['w']
                return abs(line_top - cell_top) < ACCEPTABLE_NEIGHBOUR_RANGE and abs(line_bottom - cell_bottom) < ACCEPTABLE_NEIGHBOUR_RANGE and abs(line_left - cell_left) < ACCEPTABLE_NEIGHBOUR_RANGE and abs(line_right - cell_right) < ACCEPTABLE_NEIGHBOUR_RANGE
            covered_cells = [*filter(is_covered, cells)]
            if len(covered_cells) > 0:
                for rowi, coli in itertools.product(range(rowi1, rowi2), range(coli1, coli2)):
                    unit_top = row_lines[rowi]
                    unit_bottom = row_lines[rowi + 1]
                    unit_left = col_lines[coli]
                    unit_right = col_lines[coli + 1]
                    unit_covered_cell = {'x': unit_left, 'y': unit_top,
                                         'w': unit_right - unit_left, 'h': unit_bottom - unit_top}
                    cell = {
                        'merged_position': covered_cells[0], 'unit_position': unit_covered_cell}
                    array_2d[rowi][coli] = cell
        return array_2d

    @classmethod
    def color_2d_array(cls, image, array_2ds):
        """
        画像(何でも良い)と2次元配列化したテーブルを受け取る
        全セルについて自身の2次元配列におけるインデックスを画像に書き込み、その画像を返す
        """

        output_image = image.copy()
        for array_2d in array_2ds:
            for rowi, coli in itertools.product(range(len(array_2d)), range(len(array_2d[0]))):
                cell = array_2d[rowi][coli]
                if cell is None:
                    continue
                merged_position = cell['merged_position']
                unit_position = cell['unit_position']
                cv2.putText(output_image, "[{},{}]".format(rowi, coli), (merged_position['x'] + merged_position['w'] //
                                                                         2, merged_position['y'] + merged_position['h']//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200))
                cv2.putText(output_image, "[{},{}]".format(rowi, coli), (unit_position['x'] + unit_position['w'] //
                                                                         2, unit_position['y'] + unit_position['h']//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100))
        return output_image

if __name__ == '__main__':
    pass