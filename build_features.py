"""Build the modelling table: one row per video, features from days 0-5 only."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\Admin\Documents\WCD_DataScience_Bootcamp"
           r"\generial metiral and project stuff\data\Tiktok_Dataset")
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

METRICS = ["play_count", "like_count", "comment_count", "share_count",
           "collect_count", "download_count", "whatsapp_share_count"]
SHORT = {"play_count": "play", "like_count": "like", "comment_count": "cmt",
         "share_count": "shr", "collect_count": "col", "download_count": "dl",
         "whatsapp_share_count": "wa"}
CUTOFF = 5


def load_engagement():
    e = pd.read_csv(RAW / "engagement_daily.csv")
    # cumulative counts drift down occasionally; force monotone
    e = e.sort_values(["video_id", "days_since_post"])
    e[METRICS] = e.groupby("video_id")[METRICS].cummax()
    return e


def make_target(e):
    t = e.loc[e.days_since_post == 30, ["video_id", "play_count"]]
    return t.rename(columns={"play_count": "y"}).set_index("video_id")["y"]


def early_panel(e):
    """Wide table of cumulative metrics at each of days 0..5."""
    w = e[e.days_since_post <= CUTOFF]
    p = w.pivot_table(index="video_id", columns="days_since_post",
                      values=METRICS, aggfunc="last")
    p.columns = [f"{SHORT[m]}_d{d}" for m, d in p.columns]
    # a video missing day d simply had no snapshot; carry the last known value
    for m in METRICS:
        cols = [f"{SHORT[m]}_d{d}" for d in range(CUTOFF + 1)]
        p[cols] = p[cols].ffill(axis=1)
    return p


def trajectory_features(p):
    f = pd.DataFrame(index=p.index)
    f["observed_days"] = p[[f"play_d{d}" for d in range(CUTOFF + 1)]].notna().sum(axis=1)
    f["has_d0"] = p["play_d0"].notna().astype(np.int8)

    for m in METRICS:
        s = SHORT[m]
        cum = p[[f"{s}_d{d}" for d in range(CUTOFF + 1)]].bfill(axis=1).fillna(0.0)
        cum.columns = list(range(CUTOFF + 1))

        for d in range(CUTOFF + 1):
            f[f"{s}_d{d}"] = cum[d]
            f[f"log_{s}_d{d}"] = np.log1p(cum[d])

        # daily increments
        for d in range(1, CUTOFF + 1):
            f[f"{s}_inc{d}"] = (cum[d] - cum[d - 1]).clip(lower=0)

        inc = f[[f"{s}_inc{d}" for d in range(1, CUTOFF + 1)]]
        f[f"{s}_inc_mean"] = inc.mean(axis=1)
        f[f"{s}_inc_std"] = inc.std(axis=1)
        f[f"{s}_inc_max"] = inc.max(axis=1)
        f[f"{s}_inc_last"] = inc[f"{s}_inc{CUTOFF}"]
        # is the video still accelerating at the cutoff?
        f[f"{s}_accel"] = inc[f"{s}_inc{CUTOFF}"] - inc[f"{s}_inc{CUTOFF - 1}"]
        f[f"{s}_inc_share_last"] = inc[f"{s}_inc{CUTOFF}"] / (cum[CUTOFF] + 1)

        # growth multipliers between days (log so trees can split linearly)
        for a, b in [(1, 5), (2, 5), (3, 5), (4, 5), (1, 3), (3, 5), (1, 2)]:
            f[f"{s}_g{a}{b}"] = np.log1p(cum[b]) - np.log1p(cum[a])
        # how much of the day-5 total was already there on day 1
        f[f"{s}_front_load"] = cum[1] / (cum[CUTOFF] + 1)

        # slope / curvature of log cumulative over days 1..5
        ld = np.log1p(cum[[1, 2, 3, 4, 5]].to_numpy(dtype=np.float64))
        x = np.arange(1, CUTOFF + 1, dtype=np.float64)
        xc = x - x.mean()
        f[f"{s}_slope"] = (ld * xc).sum(axis=1) / (xc ** 2).sum()
        f[f"{s}_curve"] = (ld[:, -1] - ld[:, -2]) - (ld[:, 1] - ld[:, 0])

    # engagement ratios against views, at the cutoff and how they moved
    play5 = f["play_d5"]
    for m in METRICS[1:]:
        s = SHORT[m]
        f[f"{s}_per_play5"] = f[f"{s}_d5"] / (play5 + 1)
        f[f"{s}_per_play1"] = f[f"{s}_d1"] / (f["play_d1"] + 1)
        f[f"{s}_per_play_delta"] = f[f"{s}_per_play5"] - f[f"{s}_per_play1"]

    f["engage_sum5"] = f[[f"{SHORT[m]}_d5" for m in METRICS[1:]]].sum(axis=1)
    f["engage_rate5"] = f["engage_sum5"] / (play5 + 1)
    f["shr_to_like"] = f["shr_d5"] / (f["like_d5"] + 1)
    f["cmt_to_like"] = f["cmt_d5"] / (f["like_d5"] + 1)

    # naive log-linear extrapolation of the view curve out to day 30
    f["extrap_d30"] = np.log1p(play5) + f["play_slope"] * (30 - CUTOFF)
    return f


def video_features(ids):
    cols = ["video_id", "author_id", "create_time", "create_date", "duration",
            "ratio", "desc_language", "is_english", "created_by_ai",
            "music_selected_from", "music_id", "music_owner_id", "word_count",
            "emoji_count", "question_count", "hashtag_count", "speaking_rate",
            "topic", "anger", "joy", "surprise", "sadness", "disgust", "fear"]
    v = pd.read_csv(RAW / "videos.csv", usecols=cols, low_memory=False)
    v = v[v.video_id.isin(ids)].copy()

    ct = pd.to_datetime(v.create_time, errors="coerce")
    v["hour"] = ct.dt.hour
    v["dow"] = ct.dt.dayofweek
    v["day_index"] = (ct - ct.min()).dt.days
    v["create_date"] = pd.to_datetime(v.create_date, errors="coerce")

    v["has_music_owner"] = v.music_owner_id.notna().astype(np.int8)
    v["text_missing"] = v.word_count.isna().astype(np.int8)
    v["emotion_max"] = v[["anger", "joy", "surprise", "sadness", "disgust", "fear"]].max(axis=1)
    v["emotion_std"] = v[["anger", "joy", "surprise", "sadness", "disgust", "fear"]].std(axis=1)

    # popularity of the sound the video used
    v["music_uses"] = v.groupby("music_id").video_id.transform("count")
    v["author_video_total"] = v.groupby("author_id").video_id.transform("count")

    for c in ["ratio", "desc_language", "music_selected_from", "topic"]:
        v[c] = v[c].astype("category")
    return v.drop(columns=["create_time", "music_owner_id"]).set_index("video_id")


def creator_features(v):
    c = pd.read_csv(RAW / "creator_daily.csv",
                    usecols=["author_id", "date", "follower_count",
                             "following_count", "total_favorited", "video_count"])
    c["date"] = pd.to_datetime(c.date, errors="coerce")
    c = c.dropna(subset=["date"]).sort_values(["author_id", "date"])

    # momentum: how fast the creator was growing in the week before posting
    c["fol_7d_ago"] = c.groupby("author_id").follower_count.shift(7)
    c["fav_7d_ago"] = c.groupby("author_id").total_favorited.shift(7)
    c = c.sort_values("date")  # merge_asof needs the right side sorted on the key

    key = (v.reset_index()[["video_id", "author_id", "create_date"]]
             .dropna(subset=["create_date"]).sort_values("create_date"))
    # creator stats as of the video's post date, never looking forward
    m = pd.merge_asof(key, c.drop(columns=["fol_7d_ago", "fav_7d_ago"]),
                      left_on="create_date", right_on="date",
                      by="author_id", direction="backward")

    mom = pd.merge_asof(key, c[["author_id", "date", "fol_7d_ago", "fav_7d_ago"]],
                        left_on="create_date", right_on="date",
                        by="author_id", direction="backward")
    m["fol_growth_7d"] = m.follower_count - mom.fol_7d_ago
    m["fav_growth_7d"] = m.total_favorited - mom.fav_7d_ago
    m["fol_growth_rate"] = m.fol_growth_7d / (m.follower_count + 1)

    m["fol_per_video"] = m.follower_count / (m.video_count + 1)
    m["fav_per_video"] = m.total_favorited / (m.video_count + 1)
    m["fol_to_following"] = m.follower_count / (m.following_count + 1)
    m["log_follower"] = np.log1p(m.follower_count.clip(lower=0))
    return m.drop(columns=["date", "create_date", "author_id"]).set_index("video_id")


def main():
    print("loading engagement ...")
    e = load_engagement()
    y = make_target(e)
    print(f"labelled videos: {len(y):,}")

    p = early_panel(e[e.video_id.isin(y.index)])
    del e
    print("trajectory features ...")
    f = trajectory_features(p)

    print("video features ...")
    v = video_features(set(y.index))
    print("creator features ...")
    cr = creator_features(v)

    df = f.join(v, how="inner").join(cr, how="left")
    df["y"] = y.reindex(df.index)
    df = df[df.y.notna()]

    # views relative to the creator's audience
    df["play5_per_fol"] = df.play_d5 / (df.follower_count + 1)
    df["play5_per_fav"] = df.play_d5 / (df.fav_per_video + 1)

    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"final table: {df.shape[0]:,} rows x {df.shape[1]} cols")
    df.to_pickle(OUT / "features.pkl")
    print(f"saved -> {OUT / 'features.pkl'}")


if __name__ == "__main__":
    main()
