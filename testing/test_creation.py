import json

from transformers import pipeline

rephrase_model = pipeline("text2text-generation", model="t5-small")


def model_based_rephrased_question(original_question):
    rephrased = rephrase_model(
        f"paraphrase: {original_question}", max_length=64, num_return_sequences=1
    )
    return rephrased[0]["generated_text"]


with open("data/qa_pairs.json", "r") as f:
    qa_data = json.load(f)

model_based_qa_pairs = []
for entry in qa_data:
    question = entry.get("question", "").strip()
    answer = entry.get("answer", "").strip()
    rephrased_question = model_based_rephrased_question(question)

    model_based_qa_pairs.append(
        {
            "rephrased_question": rephrased_question,
            "question": question,
            "answer": answer,
        }
    )

# Save the model-based enhanced JSON
with open("testing/model_based_revised_qa_pairs.json", "w") as f:
    json.dump(model_based_qa_pairs, f, indent=2)

print("Rephrased QA pairs saved to 'model_based_revised_qa_pairs.json'")
