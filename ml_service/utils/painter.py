import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = './cargan/utils/FiraMono-Medium.otf'


def draw_boxes(img, bboxes, classes, scores):
    """Draw bounding boxes + class + probabilities for
    an image

    Args:
      img:
      bboxes:
      classes:
      scores:

    Returns:

    """
    if len(bboxes) == 0:
        return img

    height, width, _ = img.shape
    image = Image.fromarray(img)
    font = ImageFont.truetype(
        font=FONT_PATH,
        size=np.floor(3e-2 * image.size[1] + 0.4).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)

    for box, category, score in zip(bboxes, classes, scores):
        y1, x1, y2, x2 = [int(i) for i in box]
        p1 = (x1, y1)
        p2 = (x2, y2)
        label = '{} {:.1f}%   '.format(category, score * 100)
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - label_size[1]])

        color = np.array([0, 255, 0])
        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

        draw.rectangle(
            [tuple(text_origin),
             tuple(text_origin + label_size)],
            fill=tuple(color))

        draw.text(
            tuple(text_origin),
            label, fill=(0, 0, 0),
            font=font)

    del draw
    return np.array(image)


def draw_fps(img, fps, detection_fps):
    height, width, _ = img.shape
    image = Image.fromarray(img)
    font = ImageFont.truetype(
        font=FONT_PATH,
        size=np.floor(3e-2 * image.size[1] + 0.4).astype('int32'))

    draw = ImageDraw.Draw(image)
    draw.text((10, height - 20),
              "Camera FPS: {:.2f} fps".format(fps),
              fill=(0, 255, 0),
              font=font)

    draw.text((width - 200, height - 20),
              "Detection FPS: {:.2f} fps".format(detection_fps),
              fill=(0, 255, 0),
              font=font)
    return np.array(image)
