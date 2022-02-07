import os

import cv2
import numpy as np

from .configs import *


def mark(image, mask):
    shape = image.shape
    flag = np.zeros(image.shape, dtype=np.uint8)

    def dfs(x, y, depth, flow):
        # noinspection PyChainedComparisons
        if 0 <= x < shape[0] and 0 <= y < shape[1] and mask[x, y] == 0 and not flag[x, y]:
            flow.append((x, y))
            flag[x, y] = 1
            count = 1
            for way in ((-1, 0), (0, 1), (0, -1), (1, 0)):
                count += dfs(x + way[0], y + way[1], depth + 1, flow)
            return count
        else:
            return 0

    results = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if flag[i, j]:
                continue
            if mask[i, j] == 0:
                pixels = []
                if MIN_BLOCK_SIZE <= dfs(i, j, 0, pixels) <= MAX_BLOCK_SIZE:
                    results.append((
                        image[
                            min(x[0] for x in pixels):max(x[0] for x in pixels),
                            min(x[1] for x in pixels):max(x[1] for x in pixels)
                        ],
                        sum(x[1] for x in pixels) / len(pixels)
                    ))
            flag[i, j] = 1

    if len(results) != 4:
        return None

    results.sort(key=lambda x: x[1])
    return [x[0] for x in results]


def split(image):
    image = cv2.cvtColor(image[:HEIGHT, :WIDTH], cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.threshold(
        cv2.GaussianBlur(binary, BLUR_CORE_SIZE, 0),
        BLUR_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY
    )[1]

    digits = mark(image, mask)

    if digits is None:
        return None

    for i, digit in enumerate(digits):
        scale = max(digit.shape[0] / DIGIT_HEIGHT, digit.shape[1] / DIGIT_WIDTH)
        digit = cv2.resize(
            digit, (int(digit.shape[1] / scale), int(digit.shape[0] / scale)),
            interpolation=cv2.INTER_LANCZOS4
        )
        height, width = digit.shape
        background = np.full((DIGIT_HEIGHT, DIGIT_WIDTH), 255)
        left, top = (DIGIT_WIDTH - width) // 2, (DIGIT_HEIGHT - height) // 2
        background[top:top+height, left:left+width] = digit

        digits[i] = background

    return digits
