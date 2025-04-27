from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def summarize_matched_data(matched_data):
    summaries = []

    for text, ocr_texts in matched_data:
        # Merge OCR texts
        if ocr_texts:
            ocr_content = "\n".join(ocr_texts)
            prompt = (
                "<|user|>\n"
                "summarize the following youtube chunk of the video:\n\n"
                f"{text}\n"
                "Additional OCR context:\n"
                f"{ocr_content}\n"
                "<|assistant|>\n"
            )
        else:
            prompt = (
                "<|user|>\n"
                "summarize the following youtube chunk of the video:\n\n"
                f"{text}\n"
                "<|assistant|>\n"
            )

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate summary
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and clean
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = decoded.split('<|assistant|>')[-1].strip()
        print("Summary: ",summary)
        summaries.append(summary)

    return summaries

# Example usage
if __name__ == "__main__":
    matched_data = [
        ("everyone welcome back to my channel...", ["welcome text", "bootcamp intro"]),
        ("in today's video we will learn trees...", ["binary trees", "leaf nodes"]),
    ]

    results = summarize_matched_data(matched_data)

    for idx, summary in enumerate(results):
        print(f"Summary {idx + 1}: {summary}\n")
