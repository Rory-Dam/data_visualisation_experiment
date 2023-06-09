import os
import inspect
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re
import utils

STATE_BAR_FIRST_DEMO = 0
STATE_BAR_FIRST = 2
STATE_BUBBLE_SECOND_DEMO = 4
STATE_BUBBLE_SECOND = 6

STATE_BUBBLE_FIRST_DEMO = 1
STATE_BUBBLE_FIRST = 3
STATE_BAR_SECOND_DEMO = 5
STATE_BAR_SECOND = 7

STATE_END = 8

st.set_page_config(layout="wide")

if 'experiment_state' not in st.session_state:
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    try:
        state_file = open(os.path.join(os.path.dirname(parent_dir), './public/state.txt'),"r")
        state = int(state_file.read())

        state_file = open(os.path.join(os.path.dirname(parent_dir), './public/state.txt'),"w")
        st.session_state['experiment_state'] = state
        state_file.write(str((state+1) % 2))
    except:
        state = np.random.randint(0,2)
        st.session_state['experiment_state'] = state
        try:
            state_file = open(os.path.join(os.path.dirname(parent_dir), './public/state.txt'),"w")
            state_file.write(str((state+1) % 2))
        except:
            print('ERROR!\nState.txt must be fixed.')
    finally:
        st.experimental_rerun()

with st.sidebar:
    if st.button('Next Visualisation'):
        st.session_state['experiment_state'] += 2

if st.session_state['experiment_state'] >= STATE_END:
    st.title('Thank you for participating in the experiment')
    st.text('Feel free to play around with the two visualisations,\n' + \
            'they can be accesed through the side bar on the left')

elif st.session_state['experiment_state'] == STATE_BAR_FIRST or \
     st.session_state['experiment_state'] == STATE_BAR_FIRST_DEMO or \
     st.session_state['experiment_state'] == STATE_BAR_SECOND or \
     st.session_state['experiment_state'] == STATE_BAR_SECOND_DEMO:
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
                                                        saturation_bounds=(0.35, 0.95),
                                                        value_bounds=(1, 0.55),
                                                        hue_shift_max=min(1/9, 1/len(format_structure)),
                                                        use_linear_luminance_function=True)
            for j, chapter in enumerate(format_structure[act]):
                format_structure[act][chapter] = chapter_gradient[j]

        return format_structure

    def manual_colour_meta(data: pd.DataFrame):
        acts = data['act'].unique()
        all_chapters = data['chapter'].unique()
        format_structure = dict.fromkeys(acts, None)
        for act in format_structure:
            act_chapters = data[data['act'] == act]['chapter'].unique()
            chapters_in_order = []
            for chapter in all_chapters:
                if chapter in act_chapters:
                    chapters_in_order.append(chapter)

            format_structure[act] = dict.fromkeys(chapters_in_order)

        colours = utils.manual_colour_scheme(len(acts), [len(format_structure[act]) for act in format_structure])

        for i, act in enumerate(format_structure):
            for j, chapter in enumerate(format_structure[act]):
                format_structure[act][chapter] = colours[i][j]

        return format_structure

    get_metric_postfix = {'Kdh000': '',
                            'Kdh%': '%',
                            'Kta%': '%'}


    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    bar_plot_build_dir = os.path.join(parent_dir, 'components/bar_plot_component_build')
    bar_plot_component_fun = components.declare_component('bar_plot_component', path=bar_plot_build_dir)

    title_con = st.container()
    component_con = st.container()

    if st.session_state['experiment_state'] == STATE_BAR_FIRST_DEMO or \
        st.session_state['experiment_state'] == STATE_BAR_SECOND_DEMO:
        data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/demo.csv'))
        file_name = 'demo.csv'
    else:
        data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/experiment.csv'))
        file_name = 'experiment.csv'

    with title_con:
        if st.session_state['experiment_state'] == STATE_BAR_FIRST_DEMO or \
            st.session_state['experiment_state'] == STATE_BAR_SECOND_DEMO:
            utils.center_title('Demo; Bar Plot')
        else:
            if data is not None:
                utils.center_title(f'Viewership of {data["Instance name"][1]}; Bar Plot')
            else:
                utils.center_title(f'Viewership Visualisations; Bar Plot')

        if st.session_state['experiment_state'] == STATE_BAR_FIRST_DEMO or \
            st.session_state['experiment_state'] == STATE_BAR_SECOND_DEMO:
            st.text('Every bar is a segment, its height is determined by viewership and its width by duration.\n' + \
                    'The episodes are displayed in chronological order.\n' + \
                    'Colours indicate act and chapter, act is determined by hue, chapter by brightness.\n' + \
                    'By hovering over a bar details about that segment get displayed.\n' + \
                    'Zoom in on any number of episodes using the slider below the visualisation.\n' + \
                    'The select-boxes on the left can be used to select different metrics and granularities.\n' + \
                    'Higher granularity levels are useful for looking at averages or trends over episodes or acts.')

    with st.sidebar:
        if data is not None:
            seleted_metric = st.selectbox('Data metric', ['Kdh000', 'Kdh%', 'Kta%'])
            data = format_data(data, seleted_metric)

            seleted_granularity = st.selectbox('Granularity level', ['segment', 'chapter', 'episode'])
            data = granulate_data(data, seleted_granularity)

    with component_con:
        if data is not None:
            key_string_regex = re.compile('[^a-zA-Z0-9]')
            # Use the filename in the key to remount the component when using a different file,
            # so swapping from demo to real visualisation.
            key_string = key_string_regex.sub('', f'barchart_{file_name}')

            metadata = {}
            metadata['metric'] = seleted_metric
            metadata['metric_postfix'] = get_metric_postfix[seleted_metric]
            metadata['granularity'] = seleted_granularity

            chapters = data['chapter'].unique()
            metadata['chapter_order'] = list(chapters)

            metadata['colour'] = manual_colour_meta(data)
            # metadata['colour'] = colour_meta(data)

            comp = bar_plot_component_fun(data=data, metadata=metadata, default=0, key=key_string)


elif st.session_state['experiment_state'] == STATE_BUBBLE_FIRST or \
     st.session_state['experiment_state'] == STATE_BUBBLE_FIRST_DEMO or \
     st.session_state['experiment_state'] == STATE_BUBBLE_SECOND or \
     st.session_state['experiment_state'] == STATE_BUBBLE_SECOND_DEMO:
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

    get_metric_postfix = {'Kdh000': '',
                            'Kdh%': '%',
                            'Kta%': '%'}


    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    bubble_plot_build_dir = os.path.join(parent_dir, 'components/bubble_plot_component_build')
    bubble_plot_component_fun = components.declare_component('bubble_plot_component', path=bubble_plot_build_dir)

    title_con = st.container()
    component_con = st.container()

    if st.session_state['experiment_state'] == STATE_BUBBLE_FIRST_DEMO or \
        st.session_state['experiment_state'] == STATE_BUBBLE_SECOND_DEMO:
        data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/demo.csv'))
        file_name = 'demo.csv'
    else:
        data = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'public/experiment.csv'))
        file_name = 'experiment.csv'

    with title_con:
        if st.session_state['experiment_state'] == STATE_BUBBLE_FIRST_DEMO or \
            st.session_state['experiment_state'] == STATE_BUBBLE_SECOND_DEMO:
            utils.center_title('Demo; Bubble Plot')
        else:
            if data is not None:
                utils.center_title(f'Viewership of {data["Instance name"][1]}; Bubble Plot')
            else:
                utils.center_title(f'Viewership Visualisations; Bubble Plot')

        if st.session_state['experiment_state'] == STATE_BUBBLE_FIRST_DEMO or \
            st.session_state['experiment_state'] == STATE_BUBBLE_SECOND_DEMO:
            st.text('Every bubble is a segment, its height is determined by viewership and its are by duration.\n' + \
                    'By hovering over a bubble details about that segment get displayed.\n' + \
                    'The segments are grouped by chapter.\n' + \
                    'Press the \"To Episode View\" button below the visualisation to switch to a chronological ordering.\n' + \
                    'In the episode view acts are differentiated by background colour.\n' + \
                    'Every chapter can be toggled of and on by clicking the corresponding coloured buttons.\n' + \
                    'Zoom in on any part of the y-axis by scrolling and dragging in the plot area.\n' + \
                    'The select-box on the left can be used to select different metrics.')

    with st.sidebar:
        if data is not None:
            seleted_metric = st.selectbox('Data metric', ['Kdh000', 'Kdh%', 'Kta%'])
            data = format_data(data, seleted_metric)

    with component_con:
        if data is not None:
            key_string_regex = re.compile('[^a-zA-Z0-9]')
            key_string = key_string_regex.sub('', f'barchart{file_name}')

            metadata = {}
            metadata['metric'] = seleted_metric
            metadata['metric_postfix'] = get_metric_postfix[seleted_metric]

            chapters = data['chapter'].unique()
            metadata['chapter_order'] = list(chapters)

            colour_map = {}
            colours = utils.manual_colour_scheme(len(chapters), 1)

            for i, chapter in enumerate(chapters):
                colour_map[chapter] = colours[i][0]

            metadata['colour'] = colour_map

            acts = data['act'].unique()

            act_colour_map = {}
            act_colours = utils.manual_colour_scheme(len(acts)*2, 8)[::2]

            for i, act in enumerate(acts):
                act_colour_map[act] = act_colours[i][2]

            metadata['act_colour'] = act_colour_map

            comp = bubble_plot_component_fun(data=data, metadata=metadata, default=0, key=key_string)

