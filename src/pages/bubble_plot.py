import os
import inspect
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re
import utils

def format_data(data: pd.DataFrame, metric) -> pd.DataFrame:
    if data is None:
        return None

    if metric not in data.keys():
        st.error(f'''Selected file has no {metric}. Columns: {list(data.keys())}''', icon='🚨')
        return

    for i, col_type in enumerate(data.dtypes):
        if np.issubdtype(col_type, np.number):
            data[data.keys()[i]] = data[data.keys()[i]].fillna(0)


    columns = ['episode', 'act', 'chapter', 'segment', 'start_time', 'end_time', 'viewership']
    formatted_data = pd.DataFrame(columns=columns)

    formatted_data['episode'] = data['Episode name'].astype(int)
    formatted_data['act'] = data['Act'].astype(str)
    formatted_data['chapter'] = data['Chapter'].astype(str)
    formatted_data['segment'] = data['Segment'].astype(str)

    if metric == 'Kdh000':
        formatted_data['viewership'] = np.where(data['SKO minutes'] > 0, data['Kdh000'] / data['SKO minutes'], 0)
    else:
        formatted_data['viewership'] = data[metric]

    start_times = []
    end_times = []
    for _, group in data.groupby('Episode name'):
        # Since end times are not in the fucking file I do it like this.
        episode_start_times = list(group['Start Time (seconds)'])
        episode_end_times = list(episode_start_times[1:].copy())
        episode_end_times.append(episode_start_times[-1] + 60)

        start_times.extend(episode_start_times)
        end_times.extend(episode_end_times)

    formatted_data['start_time'] = [int(time) for time in start_times]
    formatted_data['end_time'] = [int(time) for time in end_times]

    ordered_data = pd.DataFrame(columns=columns)

    for chapter in formatted_data['chapter'].unique():
        ordered_data = pd.concat([ordered_data, formatted_data[formatted_data['chapter'] == chapter]], ignore_index=True)

    return ordered_data


st.set_page_config(layout="wide")

# DEVELOPMENT:
# bubble_plot_component_fun = components.declare_component('bubble_plot_component', url='http://localhost:3001')
# PRODUCTION BUILD:
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
bubble_plot_build_dir = os.path.join(parent_dir, 'components/bubble_plot_component_build')
bubble_plot_component_fun = components.declare_component('bubble_plot_component', path=bubble_plot_build_dir)

title_con = st.container()
component_con = st.container()

data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/experiment.csv'))

with title_con:
    if data is not None:
        utils.center_title(f'Viewership of {data["Instance name"][1]}')
    else:
        utils.center_title(f'Chapter Visualisations')

with st.sidebar:
    if data is not None:
        seleted_metric = st.selectbox('Data metric', ['Kdh000', 'Kdh%', 'Kta%'])
        data = format_data(data, seleted_metric)

with component_con:
    if data is not None:
        key_string_regex = re.compile('[^a-zA-Z0-9]')
        key_string = key_string_regex.sub('', 'barchart_experiment')

        metadata = {}
        metadata['metric'] = seleted_metric

        chapters = data['chapter'].unique()
        metadata['chapter_order'] = list(chapters)

        colour_map = {}
        colours = utils.normalised_hue_range(len(chapters))

        for i, chapter in enumerate(chapters):
            colour_map[chapter] = colours[i]

        metadata['colour'] = colour_map

        comp = bubble_plot_component_fun(data=data, metadata=metadata, default=0, key=key_string)
