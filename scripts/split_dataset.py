from pathlib import Path

import pandas
from sklearn.model_selection import train_test_split


def main(input_path: str, output_dir: str,
         r_train: float = 0.8, r_dev: float = 0.1, r_test: float = 0.1, random_seed: int = 0):
    df = pandas.read_csv(input_path)
    videos = df.groupby('video.url')
    dfs = [df for _, df in videos]

    train, others = train_test_split(dfs, train_size=r_train, random_state=random_seed)
    dev, test = train_test_split(others, train_size=r_dev / (r_dev + r_test), random_state=random_seed)

    train_df = pandas.concat(train)[['message']]
    dev_df = pandas.concat(dev)
    test_df = pandas.concat(test)

    dfs = [train_df, dev_df, test_df]
    output_paths = [Path(output_dir) / (x + '.csv') for x in ['train', 'dev', 'test']]

    for df, output_path in zip(dfs, output_paths):
        df.to_csv(output_path)


if __name__ == '__main__':
    main('resources/small_sample.csv', 'resources')
