from PIL import Image, ImageDraw
import numpy as np

def generate_gsr(n):
    COLORS = ['red', 'blue', 'green']
    SHAPES = ['t', 'r', 'c']
    POSITIONS = ['right', 'left', 'above' 'under', 'in']

    for i in range(n):
        color1 = np.random.choice(COLORS)
        color2 = np.random.choice(COLORS)
        shape1 = np.random.choice(SHAPES)
        shape2 = np.random.choice(SHAPES)
        position = np.random.choice(POSITIONS)

        image = Image.new('RGB', (100, 100), 'white')
        draw = ImageDraw.Draw(image)

        center1 = np.random.random(2) * 100
        radius1 = np.random.random(1) * 50

        if shape1 == 't':
            draw.regular_polygon((center1, radius1), 3, fill=color1)
        elif shape1 == 'r':
            draw.rectangle((center1 + radius1, center1 - radius1, center1 - radius1, center1 + radius1), fill='color1')
        elif shape1 == 'c':
            draw.ellipse((center1 + radius1, center1 - radius1, center1 - radius1, center1 + radius1), fill='color1')

        if position == 'right':
            x1, y1 = center1
            x2 = x1 + np.random.random(1) * (100-x1)



        image.save('.png')