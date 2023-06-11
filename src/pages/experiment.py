import os
import inspect
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re
import utils

STATE_BAR_FIRST = 0
STATE_BUBBLE_FIRST = 1
STATE_BUBBLE_SECOND = 2
STATE_BAR_SECOND = 3
STATE_END = 4

st.set_page_config(layout="wide")

if 'experiment_state' not in st.session_state:
    st.session_state['experiment_state'] = np.random.randint(0,2)
    st.experimental_rerun()

with st.sidebar:
    if st.button('Next Visualisation'):
        st.session_state['experiment_state'] += 2

if st.session_state['experiment_state'] >= STATE_END:
    st.title('Thank you for participating in the experiment')

elif st.session_state['experiment_state'] == STATE_BAR_FIRST or \
     st.session_state['experiment_state'] == STATE_BAR_SECOND:
    def format_data(data: pd.DataFrame, metric) -> pd.DataFrame:
        def times_abs2rel(times, start_time, end_time):
            times = [int(time) for time in times]
            start_time = int(start_time)
            end_time = int(end_time)

            rel_times = list(map(lambda time: (time - start_time) / (end_time - start_time), times))

            return rel_times

        if data is None:
            return None

        if metric not in data.keys():
            st.error(f'''Selected file has no {metric}. Columns: {list(data.keys())}''', icon='🚨')
            return

        for i, col_type in enumerate(data.dtypes):
            if np.issubdtype(col_type, np.number):
                data[data.keys()[i]].fillna(0)

        columns = ['episode', 'act', 'chapter', 'segment', 'start_time', 'end_time', 'viewership']
        formatted_data = pd.DataFrame(columns=columns)

        formatted_data['episode'] = data['Episode name'].astype(int)
        formatted_data['act'] = data['Act'].astype(str)
        formatted_data['chapter'] = data['Chapter'].astype(str)
        formatted_data['segment'] = data['Segment'].astype(str)

        if metric == 'Kdh000':
            formatted_data['viewership'] = np.where(data['SKO minutes'] > 0, data['Kdh000'] / data['SKO minutes'],
                                                    data['Kdh000'])
        else:
            formatted_data['viewership'] = data[metric]

        start_times = []
        end_times = []
        for _, group in data.groupby('Episode name'):
            # Since end times are not in the fucking file I do it like this.
            episode_start_times = times_abs2rel(group['Start Time (seconds)'],
                                                group['Start Time (seconds)'].head(1).item(),
                                                group['Start Time (seconds)'].tail(1).item() + 60)
            episode_end_times = episode_start_times[1:].copy()
            episode_end_times.append(1)
            start_times.extend(episode_start_times)
            end_times.extend(episode_end_times)

        formatted_data['start_time'] = start_times
        formatted_data['end_time'] = end_times

        return formatted_data


    def granulate_data(data: pd.DataFrame, granularity):
        if data is None:
            return None

        if granularity == 'segment':
            return data

        granular_data = pd.DataFrame(columns=data.columns)
        col_episode = []
        col_act = []
        col_chapter = []
        col_segment = []
        col_start_time = []
        col_end_time = []
        col_viewership = []

        if granularity == 'chapter':
            for episode_name, episode_group in data.groupby('episode'):
                curr_chapter = episode_group['chapter'].head(1).item()
                curr_start_time = 0
                curr_viewer_seconds_sum = 0
                for _, row in episode_group.iterrows():
                    if row['chapter'] != curr_chapter:
                        col_episode.append(episode_name)
                        col_act.append(row['act'])
                        col_segment.append('NVT')
                        col_chapter.append(curr_chapter)
                        col_start_time.append(curr_start_time)
                        col_end_time.append(row['start_time'])
                        col_viewership.append(curr_viewer_seconds_sum/(col_end_time[-1] - col_start_time[-1]))

                        curr_chapter = row['chapter']
                        curr_start_time = row['start_time']
                        curr_viewer_seconds_sum = 0

                    curr_viewer_seconds_sum += row['viewership'] * (row['end_time'] - row['start_time'])

                col_episode.append(episode_name)
                col_act.append(row['act'])
                col_chapter.append(curr_chapter)
                col_segment.append('NVT')
                col_start_time.append(curr_start_time)
                col_end_time.append(row['end_time'])
                col_viewership.append(curr_viewer_seconds_sum/(col_end_time[-1] - col_start_time[-1]))

        elif granularity == 'episode':
            for name, group in data.groupby('episode'):
                col_episode.append(name)
                col_act.append(group['act'].head(1).item())

                col_chapter.append('NVT')
                col_segment.append('NVT')

                col_start_time.append(0)
                col_end_time.append(1)

                col_viewership.append(group['viewership'].mean())

        granular_data['episode'] = col_episode
        granular_data['act'] = col_act
        granular_data['chapter'] = col_chapter
        granular_data['segment'] = col_segment
        granular_data['start_time'] = col_start_time
        granular_data['end_time'] = col_end_time
        granular_data['viewership'] = col_viewership

        return granular_data


    def find_format_structure(data: pd.DataFrame):
        chapter_occurs = {}
        act_chapters = {}
        for name, group in data.groupby('act'):
            act_chapters[name] = set()

            for chapter, start_time, end_time in zip(group['chapter'], group['start_time'], group['end_time']):
                occur_time = (end_time - start_time) * (start_time + end_time) / 2
                if chapter not in chapter_occurs:
                    chapter_occurs[chapter] = [occur_time]
                else:
                    chapter_occurs[chapter].append(occur_time)
                act_chapters[name].add(chapter)

        chapter_avg_occurs = [(chapter, sum(occur)) for (chapter, occur) in chapter_occurs.items()]
        chapter_avg_occurs.sort(key=lambda chapter_occur: chapter_occur[1])
        chapters_in_order = [chapter for chapter, _ in chapter_avg_occurs]

        structure = {}
        for act in data['act'].unique():
            structure[act] = dict.fromkeys([chapter for chapter in chapters_in_order
                                            if chapter in act_chapters[act]])

        return structure

    def colour_meta(data: pd.DataFrame):
        acts = data['act'].unique()
        chapters = data['chapter'].unique()
        format_structure = dict.fromkeys(acts, None)
        for key in format_structure:
            format_structure[key] = dict.fromkeys(chapters)

        colours = utils.normalised_hue_range(len(format_structure), (0.02, 1))

        for i, act in enumerate(format_structure):
            chapter_gradient = utils.brightness_range(colours[i], len(format_structure[act]),
                                                    saturation_bounds=(0.6, 0.95),
                                                    value_bounds=(1, 0.6),
                                                    hue_shift_max=min(1/9, 1/len(format_structure)),
                                                    use_linear_luminance_function=True)
            for j, chapter in enumerate(format_structure[act]):
                format_structure[act][chapter] = chapter_gradient[j]

        return format_structure


    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    viewership_build_dir = os.path.join(parent_dir, 'components/bar_component_build')
    viewership_component_fun = components.declare_component('viewership_component', path=viewership_build_dir)

    title_con = st.container()
    component_con = st.container()

    data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/experiment.csv'))

    with title_con:
        if data is not None:
            utils.center_title(f'Viewership of {data["Instance name"][1]}; Bar Plot')
        else:
            utils.center_title(f'Viewership Visualisations; Bar Plot')

    with st.sidebar:
        if data is not None:
            seleted_metric = st.selectbox('Data metric', ['Kdh000', 'Kdh%', 'Kta%'])
            data = format_data(data, seleted_metric)

            seleted_granularity = st.selectbox('Granularity level', ['segment', 'chapter', 'episode'])
            data = granulate_data(data, seleted_granularity)

    with component_con:
        if data is not None:
            key_string_regex = re.compile('[^a-zA-Z0-9]')
            key_string = key_string_regex.sub('', f'barchart')

            metadata = {}
            metadata['metric'] = seleted_metric
            metadata['granularity'] = seleted_granularity

            chapters = data['chapter'].unique()
            metadata['chapter_order'] = list(chapters)

            metadata['colour'] = colour_meta(data)

            comp = viewership_component_fun(data=data, metadata=metadata, default=0, key=key_string)


elif st.session_state['experiment_state'] == STATE_BUBBLE_FIRST or \
     st.session_state['experiment_state'] == STATE_BUBBLE_SECOND:
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


    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    chapter_build_dir = os.path.join(parent_dir, 'components/bubble_component_build')
    chapter_component_fun = components.declare_component('chapter_component', path=chapter_build_dir)

    title_con = st.container()
    component_con = st.container()

    data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/experiment.csv'))

    with title_con:
        if data is not None:
            utils.center_title(f'Viewership of {data["Instance name"][1]}; Bubble Plot')
        else:
            utils.center_title(f'Chapter Visualisations; Bubble Plot')

    with st.sidebar:
        if data is not None:
            seleted_metric = st.selectbox('Data metric', ['Kdh000', 'Kdh%', 'Kta%'])
            data = format_data(data, seleted_metric)

    with component_con:
        if data is not None:
            key_string_regex = re.compile('[^a-zA-Z0-9]')
            key_string = key_string_regex.sub('', f'barchart')

            metadata = {}
            metadata['metric'] = seleted_metric

            chapters = data['chapter'].unique()
            metadata['chapter_order'] = list(chapters)

            colour_map = {}
            colours = utils.normalised_hue_range(len(chapters))

            for i, chapter in enumerate(chapters):
                colour_map[chapter] = colours[i]

            metadata['colour'] = colour_map

            comp = chapter_component_fun(data=data, metadata=metadata, default=0, key=key_string)

