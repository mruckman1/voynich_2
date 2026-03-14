import asyncio
import base64
import os
import io
from pypdf import PdfReader, PdfWriter
from openai import AsyncOpenAI

# 1. Setup your configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
# The script is already using the correct, up-to-date 3.1 Pro model
MODEL_ID = "google/gemini-3.1-pro-preview"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_retries=4,
)

def encode_data_to_base64(data):
    """Converts raw bytes to a base64 string."""
    return base64.b64encode(data).decode('utf-8')

async def send_to_openrouter(content_block):
    """Helper function to send the API request asynchronously."""
    try:
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # UPDATE: The file/image block is now passed first 
                        # to follow the new context management best practices.
                        content_block,
                        {
                            "type": "text",
                            "text": "Based on the information above, please provide a highly accurate, word-for-word transcription of the text in this file. Do not include any conversational filler."
                        }
                    ]
                }
            ],
            # UPDATE: Temperature is removed to allow the required 1.0 default.
            # UPDATE: Maps to Gemini's thinking_level="low" for faster, cheaper OCR.
            reasoning_effort="low",
            extra_body={
                "route": {
                    "fallbacks": ["google/gemini-3-pro", "anthropic/claude-3.5-sonnet"] 
                }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error processing chunk: {str(e)}]"

def chunk_pdf(file_path, pages_per_chunk):
    """Generator that splits a PDF into smaller byte streams."""
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    
    for start_idx in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        end_idx = min(start_idx + pages_per_chunk, total_pages)
        
        for i in range(start_idx, end_idx):
            writer.add_page(reader.pages[i])
            
        pdf_bytes_io = io.BytesIO()
        writer.write(pdf_bytes_io)
        yield pdf_bytes_io.getvalue(), start_idx + 1, end_idx

async def transcribe_file(file_path, file_type="image", pdf_chunk_size=5, max_concurrency=3):
    """Transcribes a file, handling PDF chunking concurrently for speed."""
    
    if file_type == "image":
        with open(file_path, "rb") as file:
            raw_data = file.read()
        
        base64_file = encode_data_to_base64(raw_data)
        file_extension = os.path.splitext(file_path)[1][1:].lower()
        if file_extension == 'jpg': file_extension = 'jpeg'
        
        content_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{file_extension};base64,{base64_file}"
            }
        }
        return await send_to_openrouter(content_block)
        
    elif file_type == "pdf":
        print(f"Starting async chunked processing for PDF: {file_path}")
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_chunk(chunk_bytes, start_page, end_page):
            async with semaphore:
                print(f"Transcribing pages {start_page} to {end_page}...")
                base64_file = encode_data_to_base64(chunk_bytes)
                content_block = {
                    "type": "file",
                    "file": {
                        "url": f"data:application/pdf;base64,{base64_file}"
                    }
                }
                result = await send_to_openrouter(content_block)
                return (start_page, result) 

        tasks = []
        for chunk_bytes, start_page, end_page in chunk_pdf(file_path, pdf_chunk_size):
            tasks.append(process_chunk(chunk_bytes, start_page, end_page))
            
        results = await asyncio.gather(*tasks)
        
        print("Reconstituting chunks...")
        results.sort(key=lambda x: x[0])
        
        return "\n\n--- Next Section ---\n\n".join([text for _, text in results])
        
    else:
        raise ValueError("Unsupported file type. Use 'image' or 'pdf'.")

# --- Async Execution Wrapper ---
async def main():
    # Example 1: Single image
    # result = await transcribe_file("sample_receipt.jpg", file_type="image")
    # print(result)
    
    # Example 2: Large PDF
    # result = await transcribe_file("scanned_document.pdf", file_type="pdf", pdf_chunk_size=5, max_concurrency=3)
    # print(result)
    pass

if __name__ == "__main__":
    asyncio.run(main())