import json

with open("uwb_atcc_dialogues.json", "r") as f:
    dialogue_data = json.load(f)

converted_data = []
for dialogue in dialogue_data:
    full_text = "\n".join([f"{u['utterance_id']}: {u['text']}" for u in dialogue["utterances"]])
    converted_data.append({
        "data": {
            "dialogue_id": dialogue["dialogue_id"],
            "text": full_text
        }
    })

with open("uwb_atcc_labelstudio_dialogues.json", "w") as f:
    json.dump(converted_data, f, indent=2)

print("Converted dialogues saved to uwb_atcc_labelstudio_dialogues.json")