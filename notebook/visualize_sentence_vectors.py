# %%
import math
import pickle

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy
from numpy import array_split
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# %%
def partition_df(df: DataFrame, n: int):
    sections = math.ceil(len(df.index) / n)
    split = array_split(df, sections)
    return split


# %%
test_group_size = 500

dataset_path = './resources/document_vectors.pkl'
with open(dataset_path, mode='rb') as f:
    df_all: DataFrame = pickle.load(f)

    df_all.sort_values(by=['video.url', 'published_at'], inplace=True)

    dfs = partition_df(df_all, test_group_size)

# %%
df = dfs[0]
X = numpy.stack(df['document_vector'])

embedded = TSNE(n_components=2, perplexity=50).fit_transform(X)

xs, ys = zip(*embedded)
messages = df['message'].to_list()

# %%
X[0].size

# %%
fig, ax = plt.subplots()
sc = plt.scatter(xs, ys)

annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    indices = " ".join(list(map(str, ind["ind"])))
    labels = "\n".join([messages[n] for n in ind["ind"]])
    text = f"{labels}"
    annot.set_text(text)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
