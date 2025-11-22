mkdir -p models/Phi-tiny-MoE-instruct
cd models/Phi-tiny-MoE-instruct
huggingface-cli download Phi-Research/Phi-tiny-MoE-instruct --local-dir .
cd ../..