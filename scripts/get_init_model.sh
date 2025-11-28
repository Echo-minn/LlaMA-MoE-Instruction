# mkdir -p models/Phi-mini-MoE-instruct
# cd models/Phi-mini-MoE-instruct
# huggingface-cli download microsoft/Phi-mini-MoE-instruct --local-dir .
# cd ../..

mkdir -p models/Llama-3.2-3B
cd models/Llama-3.2-3B
huggingface-cli download meta-llama/Llama-3.2-3B --local-dir .
cd ../..

# mkdir -p models/Llama-3.2-3B-Instruct
# cd models/Llama-3.2-3B-Instruct
# huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir .
# cd ../..