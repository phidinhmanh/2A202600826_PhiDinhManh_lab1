import os
import gradio as gr
from solution import (
    call_openai,
    call_openai_mini,
    compare_models,
    retry_with_backoff,
    batch_compare,
    format_comparison_table,
    OPENAI_MODEL,
    OPENAI_MINI_MODEL,
)

# ---------------------------------------------------------------------------
# Tab 1 — Chatbot Logic (reuses call_openai, call_openai_mini, retry_with_backoff)
# ---------------------------------------------------------------------------
def respond_chat(message, history, model, temperature, max_tokens, use_backoff, base_delay, max_retries):
    # Construct a formatted prompt that includes recent conversation history context
    full_prompt = ""
    for user_msg, bot_msg in history[-2:]:
        full_prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    full_prompt += f"User: {message}"
    
    # We choose call_openai or call_openai_mini based on user selection
    def execute_call():
        if model == OPENAI_MINI_MODEL:
            # Reuses call_openai_mini (Task 2)
            return call_openai_mini(
                prompt=full_prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        else:
            # Reuses call_openai (Task 1)
            return call_openai(
                prompt=full_prompt,
                model=model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            
    try:
        if use_backoff:
            # Reuses retry_with_backoff (Bonus Task A)
            response_text, latency = retry_with_backoff(
                fn=execute_call,
                max_retries=int(max_retries),
                base_delay=float(base_delay),
            )
        else:
            response_text, latency = execute_call()
            
        return response_text
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# Tab 2 — Compare Models Logic (reuses compare_models)
# ---------------------------------------------------------------------------
def run_comparison(prompt):
    if not prompt.strip():
        return "Vui lòng nhập prompt để so sánh.", "Vui lòng nhập prompt để so sánh."
        
    try:
        # Reuses compare_models (Task 3)
        res = compare_models(prompt)
        
        gpt4o_info = (
            f"=== {OPENAI_MODEL} Response ===\n"
            f"{res['gpt4o_response']}\n\n"
            f"⏱️ Latency: {res['gpt4o_latency']:.4f}s\n"
            f"💰 Cost Estimate: ${res['gpt4o_cost_estimate']:.6f}"
        )
        
        mini_info = (
            f"=== {OPENAI_MINI_MODEL} Response ===\n"
            f"{res['mini_response']}\n\n"
            f"⏱️ Latency: {res['mini_latency']:.4f}s"
        )
        
        return gpt4o_info, mini_info
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"

# ---------------------------------------------------------------------------
# Tab 3 — Batch Compare Logic (reuses batch_compare, format_comparison_table)
# ---------------------------------------------------------------------------
def run_batch(prompts_raw):
    prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
    if not prompts:
        return "Vui lòng nhập ít nhất một prompt (mỗi dòng một prompt)."
        
    try:
        # Reuses batch_compare (Bonus Task B)
        results = batch_compare(prompts)
        
        # Reuses format_comparison_table (Bonus Task C)
        table_str = format_comparison_table(results)
        return table_str
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# UI Theme Configuration
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

# ---------------------------------------------------------------------------
# Build Gradio Multi-Tab Interface
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        # 🧪 Giao Diện Thử Nghiệm LLM Hub
        Ứng dụng này sử dụng **tất cả** các hàm được cài đặt trong `solution.py` để tương tác, so sánh và đánh giá các mô hình.
        - **Mô hình chính (Main):** `{OPENAI_MODEL}`
        - **Mô hình phụ (Mini):** `{OPENAI_MINI_MODEL}`
        """
    )
    
    with gr.Tab("💬 Tương Tác Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Tham Số Chat")
                model_dropdown = gr.Dropdown(
                    choices=[OPENAI_MODEL, OPENAI_MINI_MODEL],
                    value=OPENAI_MODEL,
                    label="Chọn Mô hình",
                )
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                tokens_slider = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=256,
                    step=64,
                    label="Max Tokens",
                )
                
                gr.Markdown("### 🔁 Cấu Hình Backoff")
                use_backoff_cb = gr.Checkbox(label="Bật Exponential Backoff", value=False)
                base_delay_num = gr.Number(value=0.1, label="Base Delay (giây)", precision=2)
                max_retries_num = gr.Number(value=3, label="Số Lần Thử Lại (Max Retries)", precision=0)
                
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=respond_chat,
                    additional_inputs=[model_dropdown, temp_slider, tokens_slider, use_backoff_cb, base_delay_num, max_retries_num],
                    textbox=gr.Textbox(placeholder="Nhập câu hỏi...", container=False, scale=7, submit_btn="Gửi"),
                )
                
    with gr.Tab("⚖️ So Sánh Mô Hình"):
        gr.Markdown("### So sánh song song phản hồi của hai mô hình dựa trên một Prompt đơn lẻ.")
        prompt_input = gr.Textbox(placeholder="Nhập prompt cần so sánh...", label="Prompt đầu vào", lines=2)
        compare_btn = gr.Button("So sánh ngay", variant="primary")
        
        with gr.Row():
            gpt4o_output = gr.Textbox(label=f"Mô hình chính: {OPENAI_MODEL}", lines=8, buttons=["copy"])
            mini_output = gr.Textbox(label=f"Mô hình phụ: {OPENAI_MINI_MODEL}", lines=8, buttons=["copy"])
            
        compare_btn.click(fn=run_comparison, inputs=[prompt_input], outputs=[gpt4o_output, mini_output])
        
    with gr.Tab("📊 So Sánh Hàng Loạt (Batch)"):
        gr.Markdown("### Gửi nhiều prompt cùng lúc và xem kết quả dưới dạng bảng thống kê.")
        batch_input = gr.Textbox(
            placeholder="Mỗi dòng là một prompt. Ví dụ:\nKể tên thủ đô của Pháp\nGiải thích AI là gì ngắn gọn",
            label="Danh sách Prompts đầu vào",
            lines=5
        )
        batch_btn = gr.Button("Chạy so sánh hàng loạt", variant="primary")
        batch_output = gr.Code(label="Bảng kết quả so sánh (ASCII Table)", language="markdown")
        
        batch_btn.click(fn=run_batch, inputs=[batch_input], outputs=[batch_output])
        
    with gr.Tab("💻 Trợ Lý CLI Chatbot"):
        gr.Markdown(
            """
            ### 🤖 Hướng dẫn chạy Chatbot dòng lệnh (CLI Chatbot)
            Hàm `streaming_chatbot()` cung cấp giao diện tương tác trực tiếp dạng dòng lệnh hỗ trợ truyền phát token thời gian thực (Streaming) và lưu giữ lịch sử.
            
            Để khởi chạy, bạn có thể thực thi lệnh sau từ terminal:
            ```bash
            python template.py
            ```
            hoặc từ thư mục `solution`:
            ```bash
            python solution.py
            ```
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme)
