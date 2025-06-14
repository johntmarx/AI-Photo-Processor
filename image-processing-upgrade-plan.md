State-of-the-Art Permissively Licensed Models & Libraries for NGAPP
This document outlines relevant models and libraries, focusing on those with MIT or Apache 2.0 licenses. Always double-check the specific license terms for both the code and the model weights before use, as they can sometimes differ.

1. Object Detection
Model: RT-DETR (Real-Time DEtection TRansformer)

Description: A strong real-time object detector often outperforming YOLO variants in speed/accuracy trade-offs, especially for complex scenes.

License: Apache 2.0 (for many implementations and pre-trained models).

Hugging Face (Models & Code):

You'll find various RT-DETR models on Hugging Face by searching "RT-DETR". Many are based on PaddlePaddle or converted to other frameworks.

Example: lyuwenyu/rt-detr (PaddlePaddle original, check conversion tools for PyTorch/ONNX)

Roboflow has pre-trained RT-DETR models: Search "RF-DETR" or "RT-DETR" on Roboflow Models (often Apache 2.0).

GitHub:

PaddlePaddle Original: PaddlePaddle/PaddleDetection (Apache 2.0) - Contains RT-DETR implementation.

PyTorch implementations: Search GitHub for "RT-DETR PyTorch". Example: lyuwenyu/RT-DETR (Apache 2.0 for the code).

PyPI: Often used via frameworks like ultralytics (which may include RT-DETR support) or directly from cloned repos.

Model: YOLOv9 (and other YOLO versions with MIT forks)

Description: While Ultralytics' official YOLOv5, YOLOv8, etc., are AGPL, there are forks and independent implementations of YOLO architectures (like YOLOv9, YOLOv7) under MIT license.

License: MIT (for specific forks).

GitHub:

MultimediaTechLab/YOLO: https://github.com/MultimediaTechLab/YOLO (Provides YOLOv9, YOLOv7, YOLO-RD under an MIT License)

PyPI: Typically installed from the cloned GitHub repository: pip install -r requirements.txt then use the provided scripts.

Hugging Face: Models from these MIT-licensed forks might be available. Search by the specific GitHub repo name or model variant.

2. Image Segmentation
Model: Segment Anything Model (SAM) / Segment Anything 2 (SAM 2)

Description: Foundation model for zero-shot image segmentation, can segment any object in an image given prompts (points, boxes, text). SAM 2 improves upon the original.

License: Apache 2.0 (for the official Meta AI SAM and SAM 2 model code and weights).

Hugging Face (Models & Code):

Official: facebook/sam-vit-huge, facebook/sam-vit-large, facebook/sam-vit-base

Transformers integration: https://huggingface.co/docs/transformers/main/en/model_doc/sam

GitHub:

Official: facebookresearch/segment-anything https://github.com/facebookresearch/segment-anything (Apache 2.0)

PyPI: segment-anything-model (often a community wrapper), or use via transformers: pip install transformers

Note: MobileSAM offers lighter variants if speed is critical, often also Apache 2.0.

Model: U-2-Net

Description: Good for salient object detection and background removal, particularly effective for well-defined foregrounds.

License: Apache 2.0 (for original model and many Hugging Face ports).

Hugging Face:

BritishWerewolf/U-2-Net: https://huggingface.co/BritishWerewolf/U-2-Net (Apache 2.0)

BritishWerewolf/U-2-Net-Human-Seg: https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg (Apache 2.0, specialized for human segmentation)

GitHub:

Original: xuebinqin/U-2-Net https://github.com/xuebinqin/U-2-Net (Apache 2.0)

PyPI: Often used via cloned repos or libraries like rembg (which uses U-2-Net and is Apache 2.0). pip install rembg

3. Aesthetic Quality Scoring
Model Framework: NIMA (Neural Image Assessment) based models

Description: Predicts technical and aesthetic quality scores for images.

License: Apache 2.0 (for the idealo implementation and its pre-trained models).

GitHub:

idealo/image-quality-assessment: https://github.com/idealo/image-quality-assessment (Apache 2.0) - Provides Keras implementations and pre-trained models on AVA and TID2013 datasets.

Hugging Face:

You might find community-converted NIMA models (e.g., to ONNX or PyTorch) by searching "NIMA aesthetic". trides/aesthetic-scorer (Apache 2.0 for code, based on NIMA ideas and CLIP).

PyPI: Usually installed from the GitHub repository or by integrating the code directly.

4. Image Embeddings / Semantic Search
Model Family: CLIP (Contrastive Language-Image Pre-Training) Variants & Alternatives

Description: Learns joint embeddings for images and text, enabling semantic search and zero-shot classification.

OpenAI CLIP:

License: Code is MIT. Model use for deployed applications is out of scope of the original license (meaning it's restrictive for commercial deployment without explicit permission from OpenAI). Caution is advised for commercial/deployed use of OpenAI's original weights.

GitHub: openai/CLIP https://github.com/openai/CLIP

PyPI: clip @ git+https://github.com/openai/CLIP.git

Permissively Licensed Alternatives:

Sentence Transformers (various models): Offers many pre-trained CLIP-like models.

License: Primarily Apache 2.0 for the library and many models.

Hugging Face: Search sentence-transformers organization or models tagged with clip. Example: sentence-transformers/clip-ViT-B-32 or sentence-transformers/clip-ViT-L-14.

GitHub: UKPLab/sentence-transformers https://github.com/UKPLab/sentence-transformers (Apache 2.0)

PyPI: pip install sentence-transformers

OpenCLIP: A community effort to reproduce CLIP with open datasets and code.

License: Apache 2.0 for code and many pre-trained weights.

GitHub: mlfoundations/open_clip https://github.com/mlfoundations/open_clip (Apache 2.0)

PyPI: pip install open_clip_torch

Hugging Face: Search OpenCLIP or models like laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k (Apache 2.0 weights).

SigLIP (Sigmoid Loss for Language Image Pre-Training): Google's alternative to CLIP's contrastive loss, often performs very well.

License: Apache 2.0 for official Google releases.

Hugging Face: Search "SigLIP". Example: google/siglip-base-patch16-224 (Apache 2.0).

GitHub: Often part of larger vision model repositories from Google Research.

5. Visual Language Models (VLMs) for Iterative Analysis & Narrative Generation
You are currently using Gemma3:4b (Ollama). Gemma models have their own "Gemma Terms of Use," which are generally permissive but not strictly MIT/Apache 2.0. Check its terms for your specific use case.

Model: Qwen2.5-VL (from Alibaba)

Description: Strong VLM with good OCR, document understanding, and object localization. Some versions support video.

License: Apache 2.0 (for many Qwen models, including Qwen2.5-VL).

Hugging Face: Search Qwen/Qwen2.5-VL. Example: Qwen/Qwen2.5-7B-VL-Chat or Qwen/Qwen2.5-7B-VL.

Ollama: Many Qwen models are available. Check ollama.com/library. Example: ollama run qwen2:7b-chat-v2.5-fp16 (or VL specific variants if listed).

GitHub: QwenLM/Qwen2 https://github.com/QwenLM/Qwen2 (Apache 2.0)

Model: Pixtral (from Mistral AI)

Description: Natively multimodal, handles interleaved image and text, good instruction following.

License: Apache 2.0.

Hugging Face: Search mistralai/Pixtral-12B-v0.1.

Ollama: Pixtral might become available; check the Ollama library.

GitHub: Mistral AI typically releases model details and usage examples on their Hugging Face pages.

Model: Falcon-2 11B VLM (from TII)

Description: Built on Falcon-2 chat model, uses CLIP ViT-L/14 vision encoder. Good for fine detail perception.

License: Apache 2.0.

Hugging Face: Search tiiuae/falcon2-11b-vlm.

Ollama: May become available; check the Ollama library.

Model: Phi-4 Multimodal (from Microsoft) (Conceptual; Phi-3 Vision is available, Phi-4 might follow)

Description: Phi models are known for being strong "small language models." Phi-3 has vision variants. Phi-4, if it follows a similar pattern, would be a powerful, potentially more compact VLM.

License: MIT (for Phi-3 and likely future Phi models).

Hugging Face: Search microsoft/Phi-3-vision-128k-instruct (MIT license for Phi-3 Vision). Keep an eye out for Phi-4 vision models.

Ollama: Phi-3 models are available; Phi-4 vision might be added. ollama run phi3.

6. RAW Image Processing
Library: RawPy

Description: A Python wrapper for the LibRaw library, providing access to RAW image decoding and pre-processing.

License: LibRaw itself has multiple licenses (LGPL/CDDL/LibRaw own). RawPy code is often MIT/BSD. Ensure compliance with LibRaw's terms depending on how it's bundled/used.

GitHub: letmaik/rawpy https://github.com/letmaik/rawpy

PyPI: pip install rawpy

7. General Image Processing & Enhancement
Library: OpenCV (Open Source Computer Vision Library)

Description: Comprehensive library for a vast range of image processing and computer vision tasks.

License: Apache 2.0.

GitHub: opencv/opencv https://github.com/opencv/opencv

PyPI: pip install opencv-python (for main modules) or pip install opencv-contrib-python (includes contrib modules).

Library: Pillow (PIL Fork)

Description: User-friendly library for opening, manipulating, and saving many different image file formats.

License: PIL Software License (permissive, similar to MIT).

GitHub: python-pillow/Pillow https://github.com/python-pillow/Pillow

PyPI: pip install Pillow

Library: Scikit-image

Description: Collection of algorithms for image processing, focusing on scientific use cases.

License: BSD-3-Clause (permissive).

GitHub: scikit-image/scikit-image https://github.com/scikit-image/scikit-image

PyPI: pip install scikit-image

Important Considerations:

Model Weights vs. Code License: Always verify the license for the specific pre-trained weights you intend to use, in addition to the codebase license. They are not always the same. Hugging Face model cards usually specify this.

Dependencies: Check the licenses of all dependencies pulled in by these libraries.

Ollama Availability: For VLMs, check ollama.com/library frequently, as new models are added regularly. You can often import GGUF-formatted models from Hugging Face into Ollama manually if they aren't in the official library yet.

Performance: "State-of-the-art" can mean different things (accuracy, speed, size). You'll need to benchmark and choose models that fit your hardware capabilities and processing time requirements. Smaller, quantized versions of models are often available if resource constraints are tight.

This list should give you a strong starting point for selecting high-quality, permissively licensed components for your advanced photo processing pipeline!
