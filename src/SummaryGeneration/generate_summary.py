from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Model loaded on: {model.device}")


def build_prompt(text,title, ocr_texts=None, context="youtube" ):
    base_instruction = (
        f"Summarize the following part of {title} youtube video:"
        if context == "youtube"
        else "Summarize this part of video:"
    )
    prompt = f"<|user|>\n{base_instruction}\n\n{text}\n"

    MAX_OCR_CHARS = 1000

    if ocr_texts:
        content = []
        total = 0

        for ocr in ocr_texts:
            if total + len(ocr) > MAX_OCR_CHARS:
                break
            content.append(ocr)
            total += len(ocr)

        ocr_content = "\n".join(content)
        prompt += f"With help of this OCR:\n{ocr_content}\n"

    prompt += "<|assistant|>\n"
    return prompt


async def summarize_matched_data(matched_data, context, websocket, batch_size=14, title=None):
    prompts = [build_prompt(text, title, ocr_texts, context) for text, ocr_texts in matched_data]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        end = time.time()
        print(f"[Batch {i // batch_size + 1}] Inference time: {end - start:.2f} sec")

        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for decoded in decoded_batch:
            summary = decoded.split("<|assistant|>")[-1].strip()
            print("Summary:", summary)
            await websocket.send_text(f"[SUMMARY]{summary}")

