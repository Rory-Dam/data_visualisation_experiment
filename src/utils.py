import streamlit as st

def center_st_col(size=1):
    _, middle_col, _ = st.columns([1, size, 1])
    return middle_col


def center_title(title):
    return st.markdown(f'''<h1 style='text-align: center;'>{title}</h1>''', unsafe_allow_html=True)


def remove_padding(max_width=95):
    return st.markdown(f'''<style> \
                            .css-1y4p8pa {{
                                max-width: {max_width}%;
                            }}
                        </style>''',
                       unsafe_allow_html=True)


def make_grid(rows, cols):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)

    return grid

def hex_to_rgb(hex):
  rgb = []
  hex = hex[1:]
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)

  return tuple(rgb)

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def normalised_hue_range(n: int, hue_bounds: tuple[float, float]=(0, 1), saturation: float=0.85, value: float=0.85):
    '''
    Creates a colourmap with n different hues.
    It uses the normalised rainbow colourmap from https://dominoweb.draco.res.ibm.com/reports/rc24542.pdf
    '''
    import colorsys as cs
    import numpy as np

    NORMALIZE_BREAK_POINTS = [0, 8/27, 8/18, 1]
    NORMALIZE_FUNCTIONS = [lambda hue: (3/4) * hue,
                           lambda hue: (3/2) * hue - 4/18,
                           lambda hue: hue]
    def normalize_hue(hue):
        if hue < NORMALIZE_BREAK_POINTS[1]:
            return NORMALIZE_FUNCTIONS[0](hue)
        if hue < NORMALIZE_BREAK_POINTS[2]:
            return NORMALIZE_FUNCTIONS[1](hue)
        else:
            return NORMALIZE_FUNCTIONS[2](hue)
    # NORMALIZE_BREAK_POINTS = [0, 3/12, 5/12, 9/12, 1]
    # NORMALIZE_FUNCTIONS = [lambda hue: (1/2) * hue,
    #                         lambda hue: 3 * hue - 15/24,
    #                         lambda hue: (3/4) * hue + 15/48,
    #                         lambda hue: (1/2) * hue + 1/2]
    def normalize_hue(hue):
        if hue < NORMALIZE_BREAK_POINTS[1]:
            return NORMALIZE_FUNCTIONS[0](hue)
        if hue < NORMALIZE_BREAK_POINTS[2]:
            return NORMALIZE_FUNCTIONS[1](hue)
        if hue < NORMALIZE_BREAK_POINTS[3]:
            return NORMALIZE_FUNCTIONS[2](hue)
        else:
            return NORMALIZE_FUNCTIONS[3](hue)

    hue_space = [normalize_hue(hue % 1) for hue in np.linspace(hue_bounds[0], hue_bounds[1], n, endpoint=False)]
    hues = []

    for hue in hue_space:
        decimal_r, decimal_g, decimal_b = cs.hsv_to_rgb(hue, saturation, value)
        r, g, b = int(decimal_r*255), int(decimal_g*255), int(decimal_b*255)
        hues.append(rgb_to_hex(r, g, b))

    return hues


def brightness_range(colour_hex: tuple[int, int, int],
                     n: int,
                     saturation_bounds: tuple[float, float]=(0.5, 0.9),
                     value_bounds: tuple[float, float]=(1, 0.7),
                     hue_shift_factors: list[float]=None,
                     hue_shift_max: float=1/12,
                     use_linear_luminance_function: bool=False):
    import colorsys as cs
    import numpy as np
    import math

    LUMINANCE_BREAK_POINTS = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]
    # differentiate: 26.64 x^2+ 0 x+ 0.195,
    #                17/2 x^2 - 17/3 x + 493/300
    #                300/97 x^2 - 200/97 x + 10163/9700
    #                26.1 x^2 - 34.8 x + 11.665
    #                15.66 x^2 - 20.88 x + 7.025
    #                10.98 x^2 - 21.96 x + 11.175
    LUMINANCE_SLOPES = [lambda hue: 53.28 * hue,
                           lambda hue: 17 * hue - 17/3,
                           lambda hue: 600/97 * hue - 200/97,
                           lambda hue: 52.2 * hue - 34.8,
                           lambda hue: 31.32 * hue - 20.88,
                           lambda hue: 21.96 * hue - 21.96
                           ]
    MAX_LUMINANCE_SLOPE = abs(max([LUMINANCE_SLOPES[0](LUMINANCE_BREAK_POINTS[1]),
                              LUMINANCE_SLOPES[1](LUMINANCE_BREAK_POINTS[1]),
                              LUMINANCE_SLOPES[2](LUMINANCE_BREAK_POINTS[3]),
                              LUMINANCE_SLOPES[3](LUMINANCE_BREAK_POINTS[3]),
                              LUMINANCE_SLOPES[4](LUMINANCE_BREAK_POINTS[5]),
                              LUMINANCE_SLOPES[5](LUMINANCE_BREAK_POINTS[5])], key=abs))

    LIN_LUMINANCE_SLOPES = [4.41,
                        -1.83,
                        0.54,
                        -4.35,
                        2.61,
                        -1.83
                        ]

    LIN_MAX_LUMINANCE_SLOPE = abs(max(LIN_LUMINANCE_SLOPES, key=abs))
    LIN_MIN_SHIFT = 1.1

    def shift_hue(hue, luminance_direction, max_shift):
        # percenption of hues, for hues of different acts:
        # https://dominoweb.draco.res.ibm.com/reports/rc24542.pdf
        def slope_percieved_luminance(hue):
            if hue < LUMINANCE_BREAK_POINTS[1]:
                return LUMINANCE_SLOPES[0](hue)

            if hue < LUMINANCE_BREAK_POINTS[2]:
                return LUMINANCE_SLOPES[1](hue)

            if hue < LUMINANCE_BREAK_POINTS[3]:
                return LUMINANCE_SLOPES[2](hue)

            if hue < LUMINANCE_BREAK_POINTS[4]:
                return LUMINANCE_SLOPES[3](hue)

            if hue < LUMINANCE_BREAK_POINTS[5]:
                return LUMINANCE_SLOPES[4](hue)

            return LUMINANCE_SLOPES[5](hue)

        def lin_slope_percieved_luminance(hue):
            if hue < LUMINANCE_BREAK_POINTS[1]:
                return LIN_LUMINANCE_SLOPES[0]

            if hue < LUMINANCE_BREAK_POINTS[2]:
                return LIN_LUMINANCE_SLOPES[1]

            if hue < LUMINANCE_BREAK_POINTS[3]:
                return LIN_LUMINANCE_SLOPES[2]

            if hue < LUMINANCE_BREAK_POINTS[4]:
                return LIN_LUMINANCE_SLOPES[3]

            if hue < LUMINANCE_BREAK_POINTS[5]:
                return LIN_LUMINANCE_SLOPES[4]

            return LIN_LUMINANCE_SLOPES[5]

        if luminance_direction == 0:
            return hue

        slope = slope_percieved_luminance(hue)
        if use_linear_luminance_function:
            slope = lin_slope_percieved_luminance(hue)

        if slope == 0:
            return hue

        hue_direction = luminance_direction * slope

        if hue_direction > 0:
            closest_break_point = [point for point in LUMINANCE_BREAK_POINTS if point > hue][0]
        else:
            closest_break_point = [point for point in LUMINANCE_BREAK_POINTS if point < hue][-1]

        break_point_distance = closest_break_point - hue

        max_slope = MAX_LUMINANCE_SLOPE
        if use_linear_luminance_function:
            max_slope = LIN_MAX_LUMINANCE_SLOPE * LIN_MIN_SHIFT

        hue_diff = min(break_point_distance * abs(luminance_direction),
                       max_shift / 2 * abs(luminance_direction) * hue_direction/abs(hue_direction), key=abs) \
                   * (1 - (abs(slope) / max_slope))

        return hue + hue_diff

    def luminance_directions(n):
        divisor = math.floor(n / 2)
        directions = [(divisor - i) / divisor for i in range(divisor)]
        if n % 2:
            directions.append(0)

        inverse = [-(i + 1) / divisor for i in range(divisor)]
        directions.extend(inverse)
        return directions

    r, g, b = hex_to_rgb(colour_hex)
    hue, _, _ = cs.rgb_to_hsv(r, g, b)
    gradient = []

    saturation_space = np.linspace(saturation_bounds[0], saturation_bounds[1], n, endpoint=True)
    value_space = np.linspace(value_bounds[0], value_bounds[1], n, endpoint=True)

    if not hue_shift_factors:
        hue_shift_factors = luminance_directions(n)

    elif len(hue_shift_factors) != n:
        hue_shift_factors.extend([max(hue_shift_factors[-1], 1)] * (n - len(hue_shift_factors)))

    for i in range(n):
        shifted_hue = shift_hue(hue, hue_shift_factors[i], hue_shift_max)
        saturation = saturation_space[i]
        value = value_space[i]
        decimal_r, decimal_g, decimal_b = cs.hsv_to_rgb(shifted_hue, saturation, value)
        r, g, b = int(decimal_r*255), int(decimal_g*255), int(decimal_b*255)
        gradient.append(rgb_to_hex(r, g, b))

    return gradient

def manual_colour_scheme(palettes: list[int], range_sizes: list[int]) -> list[list[str]]:
    '''Provide a colour scheme based on manually fine-tuned colours.

    Arguments:
    palettes -- List of integers between 0 and 4 indicating which colour ranges should be used.
             or Integer indicating how many colour ranges should be used,
                the colours will be picked in a pre-determined order.
    range_sizes --  List of integers between 1 and 8 indication how many colour should be used in each corresponding range.
                or  Integer between 1 and 8 indication how many colour should be used in every range.
    '''

    COLOURS = [['#E8E5FF', '#CFC9FF', '#A89FEE', '#746AD6', '#2489FF', '#4558FF', '#0015CE', '#000C7C'],
               ['#FFF345', '#FFCE18', '#FEAA61', '#E37F43', '#FF624D', '#F03434', '#C00606', '#740035'],
               ['#EBFFE5', '#D0FFC1', '#AFF09A', '#9BD987', '#00FFD2', '#04E4BC', '#009E82', '#00473B'],
               ['#FFD9F5', '#F2B0CD', '#FF8BC1', '#E745BB', '#FB2FFF', '#CE01D2', '#930096', '#5B006B'],
               ['#C7F5FF', '#C7F5FF', '#12CDD4', '#02B3BA', '#00C2ED', '#00A1C5', '#006981', '#003444']]

    # COLOUR_ORDERS = [[[4], [4,5], [2,4,5], [2,3,4,5], [1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]],
    #                  [[4], [4,5], [2,4,5], [2,3,4,5], [1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]],
    #                  [[5], [3,5], [2,3,5], [2,3,4,5], [1,2,3,4,5], [1,2,3,4,5,6], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]],
    #                  [[4], [4,5], [2,4,5], [2,3,4,5], [1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]],
    #                  [[4], [2,4], [1,2,4], [1,2,4,5], [1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]]]
    COLOUR_ORDERS = [[3], [3,4], [3,4,5], [2,3,4,5], [1,2,3,4,5], [1,2,3,4,5,6], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7]]

    NUM_RANGES = 5
    MAX_RANGE_SIZE = 8

    if type(palettes) == int:
        palettes = [i % 5 for i in range(palettes)]
    elif max(palettes) > NUM_RANGES-1:
            return -1

    if type(range_sizes) == int:
        if range_sizes > MAX_RANGE_SIZE:
            return -1

        range_sizes = [range_sizes] * len(palettes)
    elif max(range_sizes) > MAX_RANGE_SIZE:
            return -1

    if len(palettes) != len(range_sizes):
        return -1

    colour_scheme = []
    for palette, size in zip(palettes, range_sizes):
        colour_indices = COLOUR_ORDERS[size-1]

        colour_scheme.append([COLOURS[palette][i] for i in colour_indices])

    return colour_scheme
