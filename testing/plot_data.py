import json

import pandas as pd
import plotly.express as px

path = "testing/eval-augmentation-final-2.json"

with open(path, "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
len_df = len(df)

score_columns = [col for col in df.columns if col.endswith("score")]
score_counts = (df[score_columns].astype(float)).mean().reset_index()
score_counts.columns = ["Category", "Similarity_Score"]
score_counts["Category"] = score_counts["Category"].str.replace("_score", "")

answer_columns = [col for col in df.columns if col.endswith("_answer")]
answer_matches = {
    col.replace("_answer", ""): (df[col] == df["answer"]).sum()
    for col in answer_columns
}
answer_matches_df = pd.DataFrame(
    list(answer_matches.items()), columns=["Category", "Matches"]
)

fig1 = px.bar(
    score_counts,
    x="Category",
    y="Similarity_Score",
    title="Similarity Scores Between Retrieved Question and Expected Question, For Each Augmentation Category.",
    labels={
        "Category": "Augmentation Category",
        "Similarity_Score": "Similarity Score",
    },
    text="Similarity_Score",
    color="Similarity_Score",
)

fig1.update_layout(
    xaxis_tickangle=0,
    xaxis_title="Category",
    yaxis_title="Similarity Score Average",
    title_x=0.5,
)

fig1.show()


# #=============================================================

fig = px.bar(
    answer_matches_df,
    x="Category",
    y="Matches",
    title=f"Exact Answer Matches to Reference, grouped by Augmentation Method. Total: {len_df}",
    labels={"Category": "Category", "Matches": "Matches"},
    color="Matches",
    text="Matches",
)

fig.update_layout(
    xaxis_tickangle=0,
    xaxis_title="Category",
    yaxis_title="Exact Matches",
    title_x=0.5,
)

fig.show()
