import streamlit as st


def center_title(title):
    return st.markdown(f'''<h1 style='text-align: center;'>{title}</h1>''', unsafe_allow_html=True)


def hex_to_rgb(hex):
    rgb = []
    hex = hex[1:]
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)

    return tuple(rgb)


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def normalised_hue_range(n: int, hue_bounds: tuple[float, float] = (0, 1),
                         saturation: float = 0.85, value: float = 0.85):
    """Creates a color map with n different hues using a normalized rainbow color map.

    Parameters:
        - n (int): The number of different hues to generate in the color map.
        - hue_bounds (tuple[float, float]): The range of hue values (default: (0, 1)).
        - saturation (float): The saturation value for all hues (default: 0.85).
        - value (float): The value (brightness) value for all hues (default: 0.85).

    Returns:
        - hues (list[str]): A list of RGB values as hexadecimal strings representing the hues in the color map.

    Note:
        - The function requires the 'colorsys' and 'numpy' libraries.
        - The normalized rainbow color map is based on the paper: https://dominoweb.draco.res.ibm.com/reports/rc24542.pdf

    """
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

    def normalize_hue(hue):
        if hue < NORMALIZE_BREAK_POINTS[1]:
            return NORMALIZE_FUNCTIONS[0](hue)
        if hue < NORMALIZE_BREAK_POINTS[2]:
            return NORMALIZE_FUNCTIONS[1](hue)
        if hue < NORMALIZE_BREAK_POINTS[3]:
            return NORMALIZE_FUNCTIONS[2](hue)
        else:
            return NORMALIZE_FUNCTIONS[3](hue)

    hue_space = [normalize_hue(hue % 1) for hue in np.linspace(
        hue_bounds[0], hue_bounds[1], n, endpoint=False)]
    hues = []

    for hue in hue_space:
        decimal_r, decimal_g, decimal_b = cs.hsv_to_rgb(hue, saturation, value)
        r, g, b = int(decimal_r*255), int(decimal_g*255), int(decimal_b*255)
        hues.append(rgb_to_hex(r, g, b))

    return hues


def brightness_range(colour_hex: str,
                     n: int,
                     saturation_bounds: tuple[float, float] = (0.3, 0.95),
                     value_bounds: tuple[float, float] = (1, 0.4),
                     hue_shift_factors: list[float] = None,
                     hue_shift_max: float = 1/12,
                     use_linear_luminance_function: bool = False):
    """Generates a gradient of colors within a specified brightness range.

    Parameters:
        - colour_hex (str): The base color in hexadecimal format (e.g., "#RRGGBB").
        - n (int): The number of colors to generate in the gradient.
        - saturation_bounds (tuple[float, float]): The range of saturation values (default: (0.3, 0.95)).
        - value_bounds (tuple[float, float]): The range of value (brightness) values (default: (1, 0.4)).
        - hue_shift_factors (list[float]): List of factors for shifting the hue (default: generated like: [-1,-0.5,1,0.5,1]).
        - hue_shift_max (float): The maximum amount to shift the hue (default: 1/12).
        - use_linear_luminance_function (bool): Flag indicating to use a linear luminance function,
            instead of a parabola based one (default: False). Advices when n is small.

    Returns:
        - gradient (list[str]): A list of RGB values as a gradient, represented as hexadecimal strings.

    Note:
        - The function requires the 'colorsys' and 'numpy' libraries.

    """
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
            closest_break_point = [
                point for point in LUMINANCE_BREAK_POINTS if point > hue][0]
        else:
            closest_break_point = [
                point for point in LUMINANCE_BREAK_POINTS if point < hue][-1]

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

    saturation_space = np.linspace(
        saturation_bounds[0], saturation_bounds[1], n, endpoint=True)
    value_space = np.linspace(
        value_bounds[0], value_bounds[1], n, endpoint=True)

    if not hue_shift_factors:
        hue_shift_factors = luminance_directions(n)

    elif len(hue_shift_factors) != n:
        hue_shift_factors.extend(
            [max(hue_shift_factors[-1], 1)] * (n - len(hue_shift_factors)))

    for i in range(n):
        shifted_hue = shift_hue(hue, hue_shift_factors[i], hue_shift_max)
        saturation = saturation_space[i]
        value = value_space[i]
        decimal_r, decimal_g, decimal_b = cs.hsv_to_rgb(
            shifted_hue, saturation, value)
        r, g, b = int(decimal_r*255), int(decimal_g*255), int(decimal_b*255)
        gradient.append(rgb_to_hex(r, g, b))

    return gradient


MANUAL_COLOUR_SCHEME_NUM_COLOURS = 6
MANUAL_COLOUR_SCHEME_MAX_GRADIENT_SIZE = 8


def manual_colour_scheme(palettes: list[int], range_sizes: list[int]) -> list[list[str]]:
    """ Provides a color scheme based on manually fine-tuned colors.

    Arguments:
        - palettes (list[int]): List of integers between 0 and 5 indicating which color ranges should be used.
                               Alternatively, an integer indicating how many color ranges should be used, and the colors will be picked in a pre-determined order.
        - range_sizes (list[int]): List of integers between 1 and 8 indicating how many colors should be used in each corresponding range.
                                   Alternatively, an integer between 1 and 8 indicating how many colors should be used in every range.

    Returns:
        - colour_scheme (list[list[str]]): A list of lists containing RGB values as hexadecimal strings representing the color scheme.

    Note:
        - The function supports up to 6 color ranges (palettes) and a maximum of 8 colors in each range.
        - If the inputs are invalid (e.g., out of range values), the function returns -1.

    """
    COLOURS = [['#FFEFE1', '#FFDCBE', '#F0B582', '#FF983E', '#E37D24', '#B65501', '#763700', '#421F00'],
               ['#FFD9F5', '#F8B1D1', '#FF8BC1', '#FF6CB1',
                   '#F3368E', '#C61065', '#940447', '#530027'],
               ['#C7F5FF', '#93DFEF', '#88D2E1', '#63BDD0',
                   '#3E96A9', '#197083', '#004B5B', '#002C35'],
               ['#E2FFDA', '#BFFFAB', '#9EEE85', '#6EC353',
                   '#429926', '#227C05', '#114800', '#030E00'],
               ['#FFFAB0', '#EEF087', '#DDE057', '#D3D63F',
                   '#ACAF16', '#767805', '#4B4D02', '#2E2F00'],
               ['#F4E3FF', '#DCBCF2', '#BE82E5', '#A45CD3', '#8E2FCD', '#55068A', '#3A0061', '#1E0032']]

    COLOUR_ORDERS = [[3], [1, 3], [1, 3, 5], [1, 3, 5, 6], [0, 1, 3, 5, 6], [
        0, 1, 2, 3, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7]]

    NUM_RANGES = 6
    MAX_RANGE_SIZE = 8

    if type(palettes) == int:
        palettes = [i % 6 for i in range(palettes)]
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
