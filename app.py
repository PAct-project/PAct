import gradio as gr
import spaces
import os
import shutil

from glob import glob
import numpy as np
import imageio.v3 as iio
from PIL import Image

os.environ["SPCONV_ALGO"] = "native"
from huggingface_hub import hf_hub_download

from app_utils import (
    generate_parts,
    prepare_models,
    process_image,
    apply_merge,
    DEFAULT_SIZE_TH,
    TMP_ROOT,
)


def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
    label_map = label_map.astype(np.int32)
    vis = np.ones((*label_map.shape, 3), dtype=np.uint8) * 255
    unique_ids = [i for i in np.unique(label_map) if i >= 0]
    # print(f"==================Unique segment IDs: {unique_ids}")
    for i, uid in enumerate(unique_ids):
        if uid == 0:
            vis[label_map == 0] = np.array(
                [0, 0, 0], dtype=np.uint8
            )  # background as black
            continue
        color = np.array(
            [
                (i * 50 + 80) % 256,
                (i * 120 + 40) % 256,
                (i * 180 + 20) % 256,
            ],
            dtype=np.uint8,
        )
        vis[label_map == uid] = color

    return vis


def _ensure_mask_display(mask_path: str) -> str:
    if not mask_path.lower().endswith(".exr"):
        return mask_path
    preview_path = mask_path.replace(".exr", "_preview_mask.png")
    # if os.path.exists(preview_path):
    #     return preview_path
    try:
        mask = iio.imread(mask_path)
        labels = mask[..., 0]
        if labels.min() < 0:
            labels = labels - labels.min()  # Shift to make all labels non-negative
        vis = _colorize_labels(labels)
        Image.fromarray(vis).save(preview_path)
        return preview_path
    except Exception as e:
        print(f"Failed to convert EXR mask for preview: {mask_path}, error: {e}")
        return mask_path


def _build_examples():
    examples = []
    for p in sorted(
        glob("assets/real_world_examples/**/*_processed.png", recursive=True)
    ):
        base = p.replace("_processed.png", "")
        candidates = [
            f"{base}_mask.exr",
            f"{base}_mask_segments_3.png",
            f"{base}_mask.png",
        ]
        mask_path = next((c for c in candidates if os.path.exists(c)), None)
        if mask_path:
            print(f"Found mask for example {p}: {mask_path}")
            disp_mask = _ensure_mask_display(mask_path)
            examples.append([p, disp_mask, 42])
    return examples


EXAMPLES = _build_examples()
EXAMPLES_PER_PAGE = 20

CUSTOM_CSS = """
.gradio-container .gr-examples .pagination button {
    padding: 10px 18px;
    font-size: 16px;
    min-width: 44px;
}
.gradio-container .gr-examples .pagination {
    gap: 8px;
}
"""

HEADER = """

# PAct: Single-View Part Articulation Generation

🔮 Turn one photo + mask into a **part-aware articulated 3D object**.

## How to Use

**🚀 Quick Start**: Pick an example from the examples and click **"▶️ Run Example"**.

**📋 Custom Workflow**:
1. **Upload Image** and **2D Mask** (EXR/PNG with segment IDs starting at 0, background = 0)
2. **Run Example** → you will get an articulation animation, an decomposed parts video, and a URDF zip that includes part meshes
"""


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    shutil.rmtree(user_dir)


with gr.Blocks(title="PAct", css=CUSTOM_CSS) as demo:
    gr.Markdown(HEADER)

    state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div style='text-align: center'>\n\n## Input\n\n</div>")
            with gr.Row():
                input_image = gr.Image(
                    label="Upload Image", type="filepath", height=250, width=250
                )
                mask_vis = gr.Image(
                    label="Mask", type="filepath", height=250, width=250
                )

            with gr.Row():
                segment_btn = None  # Placeholder for the button defined later
                # segment_btn = gr.Button("Segment Image", variant="primary", size="lg")
                run_example_btn = gr.Button(
                    "▶️ Run Example", variant="secondary", size="lg"
                )

            gr.Markdown("### 3D Generation Controls")

            seed_slider = gr.Slider(
                minimum=0,
                maximum=10000,
                value=42,
                step=1,
                label="Generation Seed",
                info="Random seed for 3D model generation",
            )

            cfg_slider = gr.Slider(
                minimum=0.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
                label="CFG Strength",
                info="Classifier-Free Guidance strength",
            )
            generate_mesh_btn = None  # Placeholder for the button defined later
            # generate_mesh_btn = gr.Button(
            #     "Generate 3D Model", variant="secondary", size="lg"
            # )

        with gr.Column(scale=2):
            gr.Markdown(
                "<div style='text-align: center'>\n\n## Results Display\n\n</div>"
            )

            with gr.Row():
                arti_video_path = gr.Video(
                    label="Articulation Animation", height=350, autoplay=True
                )
                exploded_video_path = gr.Video(
                    label="Decomposed Parts", height=350, autoplay=True
                )

            urdf_zip_file = gr.File(
                label="Download URDF Package (.zip format, texture mesh included)",
                interactive=False,
            )

            # with gr.Column(scale=2):
            # gr.Markdown(
            #     "<div style='text-align: center'>\n\n## Results Display\n\n</div>"
            # )

            # with gr.Row():
            #     combined_gs = gr.Model3D(
            #         label="Combined 3D Gaussians",
            #         clear_color=(1.0, 1.0, 1.0, 1.0),
            #         height=350,
            #     )
            #     exploded_gs = gr.Model3D(
            #         label="Decomposed 3D Gaussians",
            #         clear_color=(1.0, 1.0, 1.0, 1.0),
            #         height=350,
            #     )
            # combined_gs = gr.Model3D(label="Combined 3D Gaussians", clear_color=(0.0, 0.0, 0.0, 0.0), height=350)
            # exploded_gs = gr.Model3D(label="Exploded 3D Gaussians", clear_color=(0.0, 0.0, 0.0, 0.0), height=350)

    with gr.Row():
        examples = gr.Examples(
            examples=EXAMPLES,
            inputs=[input_image, mask_vis, seed_slider],
            examples_per_page=EXAMPLES_PER_PAGE,
            cache_examples=False,
        )

    demo.load(start_session)
    demo.unload(end_session)
    if segment_btn is not None:
        segment_btn.click(
            process_image,
            inputs=[
                input_image,
                mask_vis,
            ],
            outputs=[state],
        )

    if generate_mesh_btn is not None:
        generate_mesh_btn.click(
            generate_parts,
            inputs=[state, seed_slider, cfg_slider],
            outputs=(arti_video_path, exploded_video_path, urdf_zip_file),
        )
    run_example_btn.click(
        fn=process_image,
        inputs=[input_image, mask_vis],
        outputs=[state],
    ).then(
        fn=generate_parts,
        inputs=[state, seed_slider, cfg_slider],
        outputs=(arti_video_path, exploded_video_path, urdf_zip_file),
    )

if __name__ == "__main__":
    os.makedirs("ckpt", exist_ok=True)
    sam_ckpt_path = "ckpt/sam_vit_h_4b8939.pth"
    pipeline_path = os.environ.get(
        "PACT_PIPELINE_PATH",
        "PAct000/PAct",
    )

    prepare_models(sam_ckpt_path, pipeline_path=pipeline_path)

    demo.launch()
