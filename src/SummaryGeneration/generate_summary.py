# import asyncio
#
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import time
#
#
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
# tokenizer.pad_token = tokenizer.eos_token
#
# # Load model
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
#
# print(f"Model loaded on: {model.device}")
#
#
# def build_prompt(sub_text, title, ocr_texts):
#     ocr_context = " ".join(ocr_texts) if ocr_texts else "None"
#     prompt = "Summarize this video segment clearly and briefly."
#
#     if title:
#         prompt += f"\nVideo title: {title}"
#
#     prompt += f"\nVisual cues: {ocr_context}"
#     prompt += f"\nTranscript: {sub_text}"
#     return prompt
#
#
#
# async def summarize_matched_data(matched_data, websocket, batch_size=10, title=None):
#     prompts = [
#         {
#             "prompt": build_prompt(item["text"], title, item["ocr_texts"]),
#             "start": item["start"],
#             "end": item["end"]
#         }
#         for item in matched_data
#     ]
#
#     # Notify frontend that summarization has started
#     # await send_status(websocket, "[STATUS]🧠 Summarization in progress...")
#
#     try:
#         # Process batches concurrently
#         tasks = []
#         for i in range(0, len(prompts), batch_size):
#             batch = prompts[i:i + batch_size]
#             tasks.append(process_batch(batch, websocket))
#
#         # Wait for all batch processing to complete
#         await asyncio.gather(*tasks)
#
#         # After all summaries are generated, send success message
#         # await send_status(websocket, "[STATUS]✅ Summary generated successfully.")
#
#     except Exception as e:
#         # Catch any errors and send status
#         print(str(e))
#         # await send_status(websocket, f"[STATUS]❌ Error occurred: {str(e)}")
#
#
# async def process_batch(batch, websocket):
#     batch_prompts = [p["prompt"] for p in batch]
#
#     # Tokenization and model inference in separate threads
#     inputs = await asyncio.to_thread(
#         tokenizer,
#         batch_prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=1024
#     )
#     inputs = inputs.to(model.device)
#
#     start_time = time.time()
#
#     outputs = await asyncio.to_thread(
#         model.generate,
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=50,  # Shortened tokens for faster summaries
#         temperature=0.7,
#         top_p=0.95,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )
#
#     end_time = time.time()
#     print(f"Inference time for this batch: {end_time - start_time:.2f} sec")
#
#     decoded_batch = await asyncio.to_thread(
#         tokenizer.batch_decode,
#         outputs,
#         skip_special_tokens=True
#     )
#
#     # Send each summary over WebSocket immediately
#     for decoded, meta in zip(decoded_batch, batch):
#         summary = decoded.strip()
#         message = {
#             "start": meta["start"],
#             "end": meta["end"],
#             "summary": summary
#         }
#         print("Summary:", message)
#
#         await websocket.send_json(message)
#
# import asyncio
# import time
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# # Optional: Uncomment for 4-bit quantization
# # from transformers import BitsAndBytesConfig
#
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
# tokenizer.pad_token = tokenizer.eos_token
#
# # Load model
# # Standard loading with float16
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# # Optional: 4-bit quantization (uncomment if needed)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_id,
# #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
# #     device_map="auto"
# # )
#
# # Optional: Compile model for PyTorch 2.x (uncomment if using PyTorch 2.x)
# # model = torch.compile(model)
#
# print(f"Model loaded on: {model.device}")
#
# def build_prompt(sub_text, title, ocr_texts):
#     ocr_context = " ".join(ocr_texts) if ocr_texts else "None"
#     prompt = "Summarize this video segment clearly and briefly."
#     if title:
#         prompt += f"\nVideo title: {title}"
#     prompt += f"\nVisual cues: {ocr_context}"
#     prompt += f"\nTranscript: {sub_text}"
#     return prompt
#
# async def process_batch(batch):
#     batch_prompts = [p["prompt"] for p in batch]
#
#     # Run tokenization, inference, and decoding in one thread
#     def run_batch():
#         start_tokenize = time.time()
#         inputs = tokenizer(
#             batch_prompts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512  # Reduced for speed
#         )
#         inputs = inputs.to(model.device)
#         print(f"Tokenization time: {time.time() - start_tokenize:.2f} sec")
#
#         start_inference = time.time()
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=30,  # Reduced for faster summaries
#             do_sample=False,  # Greedy decoding for speed
#             pad_token_id=tokenizer.eos_token_id
#         )
#         print(f"Inference time: {time.time() - start_inference:.2f} sec")
#
#         start_decode = time.time()
#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         print(f"Decoding time: {time.time() - start_decode:.2f} sec")
#
#         return decoded
#
#     start_time = time.time()
#     decoded_batch = await asyncio.to_thread(run_batch)
#     print(f"Total batch processing time: {time.time() - start_time:.2f} sec")
#
#     # Prepare summaries
#     summaries = [
#         {"start": meta["start"], "end": meta["end"], "summary": decoded.strip()}
#         for decoded, meta in zip(decoded_batch, batch)
#     ]
#     return summaries
#
# async def summarize_matched_data(matched_data, websocket, batch_size=8, title=None):
#     prompts = [
#         {
#             "prompt": build_prompt(item["text"], title, item["ocr_texts"]),
#             "start": item["start"],
#             "end": item["end"]
#         }
#         for item in matched_data
#     ]
#
#     # Notify frontend that summarization has started
#     await websocket.send_json({"status": "Summarization in progress..."})
#     start_total = time.time()
#
#     try:
#         all_summaries = []
#         # Process batches sequentially
#         for i in range(0, len(prompts), batch_size):
#             batch = prompts[i:i + batch_size]
#             print(f"Processing batch {i // batch_size + 1} of {len(prompts) // batch_size + 1}")
#             batch_summaries = await process_batch(batch)
#             all_summaries.extend(batch_summaries)
#
#             # Send batch summaries immediately
#             start_send = time.time()
#             await websocket.send_json(batch_summaries)
#             print(f"WebSocket send time: {time.time() - start_send:.2f} sec")
#
#         # Send completion status
#         await websocket.send_json({"status": "Summary generated successfully."})
#         print(f"Total summarization time: {time.time() - start_total:.2f} sec")
#
#     except Exception as e:
#         error_msg = {"status": f"Error occurred: {str(e)}"}
#         print(error_msg)
#         await websocket.send_json(error_msg)
#         raise

import asyncio
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Optional: Uncomment for 4-bit quantization
# from transformers import BitsAndBytesConfig

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

# Optional: compile for PyTorch 2.x
# model = torch.compile(model)

print(f"Model loaded on: {model.device}")

def build_prompt(sub_text, title, ocr_texts):
    ocr_context = " ".join(ocr_texts) if ocr_texts else "none"
    return (
        f"Below is a transcript of a video segment.\n"
        f"Your task is to write a short and clear summary.\n\n"
        f"Title: {title or 'Untitled'}\n"
        f"Visuals: {ocr_context}\n"
        f"Transcript: {sub_text}\n\n"
        f"Summary:"
    )

def clean_summary(text):
    # Remove echoed prompt
    if "Summary:" in text:
        return text.split("Summary:", 1)[-1].strip()
    return text.strip()

async def process_batch(batch):
    batch_prompts = [p["prompt"] for p in batch]

    def run_batch():
        start_tokenize = time.time()
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        print(f"Tokenization time: {time.time() - start_tokenize:.2f} sec")

        start_inference = time.time()
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,  # Increased from 30
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"Inference time: {time.time() - start_inference:.2f} sec")

        start_decode = time.time()
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [clean_summary(text) for text in decoded]
        print(f"Decoding + Cleaning time: {time.time() - start_decode:.2f} sec")

        return cleaned

    start_total = time.time()
    decoded_batch = await asyncio.to_thread(run_batch)
    print(f"Total batch processing time: {time.time() - start_total:.2f} sec")

    summaries = [
        {"start": meta["start"], "end": meta["end"], "summary": decoded}
        for decoded, meta in zip(decoded_batch, batch)
    ]
    return summaries

async def summarize_matched_data(matched_data, websocket, batch_size=8, title=None):
    prompts = [
        {
            "prompt": build_prompt(item["text"], title, item["ocr_texts"]),
            "start": item["start"],
            "end": item["end"]
        }
        for item in matched_data
    ]

    await websocket.send_json({"status": "Summarization in progress..."})
    start_total = time.time()

    try:
        all_summaries = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {(len(prompts) - 1) // batch_size + 1}")
            batch_summaries = await process_batch(batch)
            all_summaries.extend(batch_summaries)

            await websocket.send_json(batch_summaries)

        await websocket.send_json({"status": "Summary generated successfully."})
        print(f"Total summarization time: {time.time() - start_total:.2f} sec")

    except Exception as e:
        error_msg = {"status": f"Error occurred: {str(e)}"}
        print(error_msg)
        await websocket.send_json(error_msg)
        raise
