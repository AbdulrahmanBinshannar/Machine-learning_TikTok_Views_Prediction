"""Second feature pass: peer-group context built from day-0..5 data only.

Every aggregate here uses the full 209k video catalogue but never touches any
day > 5, so it stays valid for the hidden test videos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\Admin\Documents\WCD_DataScience_Bootcamp"
           r"\generial metiral and project stuff\data\Tiktok_Dataset")
HERE = Path(__file__).resolve().parent


def main():
    df = pd.read_pickle(HERE / "data" / "features.pkl")

    # day-5 view level for every video in the catalogue, labelled or not
    e = pd.read_csv(RAW / "engagement_daily.csv",
                    usecols=["video_id", "days_since_post", "play_count",
                             "like_count", "share_count"])
    e = e[e.days_since_post <= 5].sort_values(["video_id", "days_since_post"])
    g = e.groupby("video_id").agg(p5=("play_count", "max"),
                                  l5=("like_count", "max"),
                                  s5=("share_count", "max"))
    g["lp5"] = np.log1p(g.p5)

    v = pd.read_csv(RAW / "videos.csv",
                    usecols=["video_id", "author_id", "music_id", "create_date", "topic"],
                    low_memory=False).set_index("video_id")
    cat = v.join(g, how="inner")
    cat["create_date"] = pd.to_datetime(cat.create_date, errors="coerce")

    def agg(key, tag):
        a = cat.groupby(key).lp5.agg(["mean", "std", "max", "count"])
        a.columns = [f"{tag}_lp5_{c}" for c in a.columns]
        return a

    auth = agg("author_id", "auth")
    auth["auth_share_rate"] = (cat.groupby("author_id").s5.sum()
                               / (cat.groupby("author_id").p5.sum() + 1))
    mus = agg("music_id", "mus")
    top = agg("topic", "top")

    m = cat.reindex(df.index)

    for tbl, key in ((auth, m.author_id), (mus, m.music_id), (top, m.topic)):
        for c in tbl.columns:
            df[c] = key.map(tbl[c]).to_numpy()

    lp5 = np.log1p(df.play_d5)
    # how this video is doing against the creator's own norm
    df["vs_auth"] = lp5 - df.auth_lp5_mean
    df["vs_auth_max"] = lp5 - df.auth_lp5_max
    df["vs_auth_z"] = df.vs_auth / (df.auth_lp5_std + 0.1)
    df["vs_music"] = lp5 - df.mus_lp5_mean
    df["vs_topic"] = lp5 - df.top_lp5_mean

    # rank inside the creator's catalogue and inside the posting day
    df["rank_in_auth"] = df.groupby(m.author_id.values).play_d5.rank(pct=True)
    df["rank_in_day"] = df.groupby(m.create_date.values).play_d5.rank(pct=True)
    df["rank_global"] = df.play_d5.rank(pct=True)

    # gap since the creator's previous upload
    cd = m.create_date
    prev = cd.groupby(m.author_id.values).shift(1)
    df["days_since_prev"] = (cd - prev).dt.days

    # cross-metric velocity: shares outrunning views is the classic viral tell
    df["shr_vs_play_slope"] = df.shr_slope - df.play_slope
    df["like_vs_play_slope"] = df.like_slope - df.play_slope
    df["cmt_vs_play_slope"] = df.cmt_slope - df.play_slope
    df["col_vs_play_slope"] = df.col_slope - df.play_slope

    # the raw extrapolation overflows; keep it but bounded
    df["extrap_d30"] = df.extrap_d30.clip(0, 25)

    df = df.drop(columns=["_a"], errors="ignore").replace([np.inf, -np.inf], np.nan)
    print(f"table now {df.shape[0]:,} x {df.shape[1]}")
    df.to_pickle(HERE / "data" / "features2.pkl")
    print("saved -> data/features2.pkl")


if __name__ == "__main__":
    main()
