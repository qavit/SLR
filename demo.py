import gradio as gr
from fetch_kp_yt import read_csv_and_download, load_and_detect

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Video Downloader and Pose Extractor")

    with gr.Row():
        url_path = gr.File(label="CSV File with Video URLs")
        video_dir = gr.Textbox(label="Video Directory", value="Videos")
        keypoint_dir = gr.Textbox(label="Keypoint Directory", value="Keypoints")
    
    with gr.Row():
        download_button = gr.Button("Download Videos")
        extract_button = gr.Button("Detect and Extract")
    
    with gr.Row():
        downloaded = gr.Textbox(label="Downloaded Videos")
        extracted = gr.Textbox(label="Extracted Videos")

    download_button.click(fn = read_csv_and_download,
                    inputs = [url_path, video_dir],
                    outputs = downloaded)
    
    extract_button.click(fn = load_and_detect,
                    inputs = [video_dir, keypoint_dir],
                    outputs = extracted)

# Run the Gradio app
demo.launch(share=False)