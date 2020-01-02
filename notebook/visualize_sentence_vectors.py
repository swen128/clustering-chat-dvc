# %%
import pickle

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# %%
dataset_path = './resources/clustering_results.pkl'
with open(dataset_path, mode='rb') as f:
    df: DataFrame = pickle.load(f)[250]

# %%
df.drop_duplicates(subset=['message'], inplace=True)
X = numpy.stack(df['document_vector'])

embedded = TSNE(n_components=2, perplexity=10, random_state=0).fit_transform(X)
# embedded = PCA(n_components=2).fit_transform(X)

xs, ys = zip(*embedded)
messages = df['message'].to_list()

# %%

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
color_map = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
    '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]
colors = [color_map[i] for i in df['clustering_label']]

fig, ax = plt.subplots()
sc = plt.scatter(xs, ys, c=colors)

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
