import json

import pandas as pd

df = pd.read_json("data/train.json")


def format_table(table):
    try:
        table_data = eval(table)
        headers = "| " + " | ".join(table_data[0]) + " |\n"
        separators = "| " + " | ".join(["---"] * len(table_data[0])) + " |\n"
        rows = "".join("| " + " | ".join(row) + " |\n" for row in table_data[1:])
        return headers + separators + rows
    except Exception as e:
        return "Table formatting failed: " + str(e)


df_additionalcontext = df[["pre_text", "table", "post_text"]]

documents = []
for _, row in df_additionalcontext.iterrows():
    pre_text = " ".join(eval(str(row["pre_text"])))
    table_text = format_table(str(row["table"]))
    post_text = " ".join(eval(str(row["post_text"])))
    document = f"{pre_text}\n\n{table_text}\n\n{post_text}"
    documents.append(document)

qa_pairs = (
    pd.json_normalize(pd.concat([df["qa"], df["qa_0"], df["qa_1"]], ignore_index=True))[
        ["question", "answer"]
    ].dropna(ignore_index=True)
).to_dict(orient="records")


with open("data/qa_pairs.json", "w") as file:
    json.dump(qa_pairs, file, indent=2)

with open("data/additional_context.json", "w") as file:
    json.dump(documents, file, indent=2)
