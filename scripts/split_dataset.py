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

    train_df = pandas.concat(train)
    dev_df = pandas.concat(dev)
    test_df = pandas.concat(test)

    out_dir = Path(output_dir)

    train_df['message'].to_csv(out_dir / 'train.txt', index=False)
    dev_df.to_csv(out_dir / 'dev.csv')
    test_df.to_csv(out_dir / 'test.csv')


if __name__ == '__main__':
    main('resources/raw.csv', 'resources')
