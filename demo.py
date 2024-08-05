import gradio as gr
from fetch import create_folder, read_csv_and_download, load_and_detect
from settings import *

create_folder(VIDEO_DIR, KP_DIR)

def video_identity(video):
    return video

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Video Downloader and Pose Extractor")

    with gr.Row():
        url_path = gr.File(label="CSV File with Video URLs")
        video_dir = gr.Textbox(label="Video Directory", value=VIDEO_DIR)
        keypoint_dir = gr.Textbox(label="Keypoint Directory", value=KP_DIR)
    
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

if __name__ == "__main__":
    # Run the Gradio app
    demo.launch(share=False)