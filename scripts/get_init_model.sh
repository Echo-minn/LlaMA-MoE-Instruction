# mkdir -p models/Phi-mini-MoE-instruct
# cd models/Phi-mini-MoE-instruct
# huggingface-cli download microsoft/Phi-mini-MoE-instruct --local-dir .
# cd ../..

mkdir -p models/Llama-3.1-8B
cd models/Llama-3.1-8B
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir .
cd ../..