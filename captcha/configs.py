DIGITS = 4  # 验证码的字符数
WIDTH, HEIGHT = 67, 27  # 从左上角裁剪的长宽
BINARY_THRESHOLD = 180  # 二值化的阈值
BLUR_CORE_SIZE = (3, 5)  # 高斯模糊核心大小
BLUR_BINARY_THRESHOLD = 210  # 高斯模糊后再次二值化的阈值
MIN_BLOCK_SIZE, MAX_BLOCK_SIZE = 20, 150  # 像素块大小范围
DIGIT_WIDTH, DIGIT_HEIGHT = 10, 15  # 验证码数字尺寸
