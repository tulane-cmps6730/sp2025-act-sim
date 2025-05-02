import torch
import torch.nn.functional as F
import json

from lstm_model import load_lstm_model
from minilm_model import load_minilm_model

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

VOCAB_PATH = "../models/utterance_lstm/vocab.json"
LABEL2ID_PATH = "../models/utterance_lstm/label2id.json"
ID2LABEL_PATH = "../models/utterance_lstm/id2label.json"

vocab = load_json(VOCAB_PATH)
label2id = load_json(LABEL2ID_PATH)
id2label = load_json(ID2LABEL_PATH)

LABEL_COLORS = {
    "INSTRUCTION": "\033[33m",
    "ALTITUDE": "\033[32m",
    "CALLSIGN": "\033[34m",
    "SPEED": "\033[36m",
    "ROUTE": "\033[35m",
    "UNKNOWN": "\033[31m",
}
RESET_COLOR = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

CLASS_ID_TO_LABEL = {
    0: "MATCH",
    1: "PARTIAL",
    2: "INCORRECT",
    3: "NO_RELATION",
}

def preprocess_utterance(utterance, vocab):
    tokens = utterance.lower().split()
    input_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    lengths = torch.tensor([len(input_ids)], dtype=torch.long)
    return input_tensor, lengths, tokens

def reconstruct_utterance_with_span(full_tokens, extracted_tokens, tag_type):
    wrapped = []
    inside_span = False

    for token in full_tokens:
        if token in extracted_tokens and not inside_span:
            if tag_type == "atc":
                wrapped.append("[ ins ]")
            else:
                wrapped.append("[ rbk ]")
            wrapped.append(token)
            inside_span = True
        elif token in extracted_tokens and inside_span:
            wrapped.append(token)
        elif token not in extracted_tokens and inside_span:
            if tag_type == "atc":
                wrapped.append("[ ins ]")
            else:
                wrapped.append("[ rbk ]")
            wrapped.append(token)
            inside_span = False
        else:
            wrapped.append(token)

    if inside_span:
        if tag_type == "atc":
            wrapped.append("[ ins ]")
        else:
            wrapped.append("[ rbk ]")

    return " ".join(wrapped)

def main():
    LSTM_PATH = "../models/utterance_lstm/slot_lstm.pt"
    MINILM_DIR = "../models/dialogue_relation"

    lstm_model = load_lstm_model(LSTM_PATH)
    tokenizer, minilm_model = load_minilm_model(MINILM_DIR)
    print(f"{GREEN}Models loaded successfully.{RESET}")


    utterance_history = []
    while True:
        print("\nEnter transmission ('atc: text' or 'pilot: text') ('exit' to quit):")
        full_input = input("> ").strip()

        if full_input.lower() == "exit":
            print("Exiting...")
            break

        if ":" not in full_input:
            print("Format must be 'atc: your text' or 'pilot: your text'")
            continue

        speaker, utterance = full_input.split(":", 1)
        speaker = speaker.strip().lower()
        utterance = utterance.strip()

        if speaker not in ["atc", "pilot"]:
            print("Speaker must be 'atc' or 'pilot'")
            continue

        input_tensor, lengths, tokens = preprocess_utterance(utterance, vocab)

        with torch.no_grad():
            slot_logits = lstm_model(input_tensor, lengths)
        probs = F.softmax(slot_logits, dim=-1)

        slot_preds = probs.argmax(dim=-1).squeeze(0)
        probs = probs.squeeze(0)

        print("\nUtterance predictions:")
        instruction_tokens = []
        altitude_tokens = []

        for idx, (token, pred_id) in enumerate(zip(tokens, slot_preds.tolist())):
            pred_label = id2label.get(str(pred_id), "UNKNOWN")
            confidence = probs[idx, pred_id].item()

            color_code = RESET_COLOR
            for key in LABEL_COLORS:
                if key in pred_label:
                    color_code = LABEL_COLORS[key]
                    break

            print(f"  {token} -> {color_code}{pred_label}{RESET_COLOR} (confidence: {confidence:.2f})")

            if "INSTRUCTION" in pred_label:
                instruction_tokens.append(token)
            if "ALTITUDE" in pred_label:
                altitude_tokens.append(token)

        reconstructed_utterance = reconstruct_utterance_with_span(
            full_tokens=tokens,
            extracted_tokens=instruction_tokens,
            tag_type=speaker
        )

        print(f"\nUtterance as seen by model: {reconstructed_utterance}")

        utterance_history.append({
            "speaker": speaker,
            "original_utterance": utterance,
            "reconstructed_utterance": reconstructed_utterance,
            "instruction_tokens": instruction_tokens,
            "altitude_tokens": altitude_tokens,
        })
        if speaker == "pilot":
            last_atc_turn = None
            for past_turn in reversed(utterance_history):
                if past_turn["speaker"] == "atc":
                    last_atc_turn = past_turn
                    break

            if last_atc_turn:
                atc_reconstructed = last_atc_turn["reconstructed_utterance"]
                pilot_reconstructed = utterance_history[-1]["reconstructed_utterance"]

                formatted_text = f"{atc_reconstructed} [SEP] {pilot_reconstructed}"

                inputs = tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                with torch.no_grad():
                    outputs = minilm_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                logits = outputs
                probs = F.softmax(logits, dim=-1)

                pred_id = probs.argmax(dim=-1).item()
                confidence = probs[0, pred_id].item()

                classification = CLASS_ID_TO_LABEL.get(pred_id, "UNKNOWN")

                if classification in ["MATCH", "PARTIAL", "INCORRECT"]:
                    print(f"\nMiniLM readback classification: {classification} (confidence: {confidence:.2f})")
                    print(f"Matched ATC: {atc_reconstructed}")
                else:
                    print(f"\nMiniLM readback classification: {classification} (confidence: {confidence:.2f})")

                # compare altitude
                pilot_altitude = utterance_history[-1]["altitude_tokens"]
                atc_altitude = last_atc_turn["altitude_tokens"]

                if pilot_altitude and atc_altitude:
                    if pilot_altitude == atc_altitude:
                        print(f"{GREEN}Altitude Match: ({' '.join(pilot_altitude)}){RESET}")
                    else:
                        print(f"{RED}READBACK INCORRECT: ATC expected {' '.join(atc_altitude)}{RESET}")
            else:
                print("No prior ATC instruction to compare against.")

if __name__ == "__main__":
    main()
