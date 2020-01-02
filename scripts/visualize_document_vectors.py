# %%
import math
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from numpy import array_split
from pandas import DataFrame
from sklearn.manifold import TSNE


# %%
def partition_df(df: DataFrame, n: int) -> List[DataFrame]:
    sections = math.ceil(len(df.index) / n)
    split = array_split(df, sections)
    return split


def scatter_plot_with_tooltip(xs, ys, labels: List[str]) -> Figure:
    fig, axes = plt.subplots()
    scatter_plot = axes.scatter(xs, ys)
    annotation = axes.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
    annotation.set_visible(False)

    def update_annotation(indices: List[int]) -> None:
        pos = scatter_plot.get_offsets()[indices[0]]
        annotation.xy = pos
        text = "\n".join([labels[n] for n in indices])
        annotation.set_text(text)

    def hover(event: MouseEvent) -> None:
        is_visible = annotation.get_visible()
        if event.inaxes == axes:
            is_contained, ind = scatter_plot.contains(event)
            if is_contained:
                update_annotation(ind['ind'])
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            elif is_visible:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    return fig


# %%
test_group_size = 500

dataset_path = './resources/document_vectors.pkl'
with open(dataset_path, mode='rb') as f:
    df_all: DataFrame = pickle.load(f)
    df_all.sort_values(by=['video.url', 'published_at'], inplace=True)
    dfs = partition_df(df_all, test_group_size)

# %%
df = dfs[100]
X = numpy.stack(df['document_vector'])

embedded = TSNE(n_components=2, perplexity=50).fit_transform(X)

xs, ys = zip(*embedded)
messages = df['message'].to_list()

# %%
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
figure = scatter_plot_with_tooltip(xs, ys, messages)
