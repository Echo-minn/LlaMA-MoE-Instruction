#!/usr/bin/env python3
"""
MoE Routing Sanity Check
========================
This script verifies that different tasks route to different experts.
Checks for routing collapse where 1-2 experts dominate all tasks.

Usage:
    python scripts/check_routing_sanity.py --model_path models/Llama-3.2-3B-Instruct-MoE-8x
"""

import torch
import argparse
from transformers import AutoTokenizer
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict
import sys
import os
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling_llama_moe import LlamaMoEForCausalLM
from configuration_llama_moe import LlamaMoEConfig


# Sample prompts for different tasks
TASK_PROMPTS = {
    "math": [
        "### Question:\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\n### Answer:",
        "### Question:\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\n### Answer:",
        "### Question:\nJames writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\n\n### Answer:",
        "### Question:\nMark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?\n\n### Answer:",
        "### Question:\nAlbert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?\n\n### Answer:",
    ],
    "code": [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nCreate a function to calculate the sum of a sequence of integers.\n ### Input: [1, 2, 3, 4, 5].\n\n### Output:",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nWrite a Python code to get the third largest element in a given row.\n ### Input: [12, 13, 13, 45, 22, 99]. \n\n### Output:",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nCreate a Python function that takes in a string and a list of words and returns true if the string contains all the words in the list.\n ### Input: 'This is a test', ['test', 'this', 'is']. \n\n### Output:",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nCreate a Python program to sort and print out the elements of an array of integers.\n ### Input: [17, 41, 5, 22, 54, 6, 29, 3, 13] \n\n### Output:",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nWrite a Python code to find the maximum element in a list.\n ### Input: [12, 13, 13, 45, 22, 99]. \n\n### Output:",
    ],
    "summarization": [
        "### Article:\nWASHINGTON (CNN) -- Vice President Dick Cheney will serve as acting president briefly Saturday while President Bush is anesthetized for a routine colonoscopy, White House spokesman Tony Snow said Friday. Bush is scheduled to have the medical procedure, expected to take about 2 1/2 hours, at the presidential retreat at Camp David, Maryland, Snow said. Bush's last colonoscopy was in June 2002, and no abnormalities were found, Snow said. The president's doctor had recommended a repeat procedure in about five years. The procedure will be supervised by Dr. Richard Tubb and conducted by a multidisciplinary team from the National Naval Medical Center in Bethesda, Maryland, Snow said. A colonoscopy is the most sensitive test for colon cancer, rectal cancer and polyps, small clumps of cells that can become cancerous, according to the Mayo Clinic. Small polyps may be removed during the procedure. Snow said that was the case when Bush had colonoscopies before becoming president. Snow himself is undergoing chemotherapy for cancer that began in his colon and spread to his liver. Snow told reporters he had a chemo session scheduled later Friday. Watch Snow talk about Bush's procedure and his own colon cancer » . \"The president wants to encourage everybody to use surveillance,\" Snow said. The American Cancer Society recommends that people without high-risk factors or symptoms begin getting screened for signs of colorectal cancer at age 50. E-mail to a friend .\n\n### TL;DR:",
        "### Article:\nSAN FRANCISCO, California (CNN) -- A magnitude 4.2 earthquake shook the San Francisco area Friday at 4:42 a.m. PT (7:42 a.m. ET), the U.S. Geological Survey reported. The quake left about 2,000 customers without power, said David Eisenhower, a spokesman for Pacific Gas and Light. Under the USGS classification, a magnitude 4.2 earthquake is considered \"light,\" which it says usually causes minimal damage. \"We had quite a spike in calls, mostly calls of inquiry, none of any injury, none of any damage that was reported,\" said Capt. Al Casciato of the San Francisco police. \"It was fairly mild.\" Watch police describe concerned calls immediately after the quake » . The quake was centered about two miles east-northeast of Oakland, at a depth of 3.6 miles, the USGS said. Oakland is just east of San Francisco, across San Francisco Bay. An Oakland police dispatcher told CNN the quake set off alarms at people's homes. The shaking lasted about 50 seconds, said CNN meteorologist Chad Myers. According to the USGS, magnitude 4.2 quakes are felt indoors and may break dishes and windows and overturn unstable objects. Pendulum clocks may stop. E-mail to a friend .\n\n### TL;DR:",
        "### Article:\n(CNN) -- At least 14 people were killed and 60 others wounded Thursday when a bomb ripped through a crowd waiting to see Algeria's president in Batna, east of the capital of Algiers, the Algerie Presse Service reported. A wounded person gets first aid shortly after Thursday's attack in Batna, Algeria. The explosion occurred at 5 p.m. about 20 meters (65 feet) from a mosque in Batna, a town about 450 kilometers (280 miles) east of Algiers, security officials in Batna told the state-run news agency. The bomb went off 15 minutes before the expected arrival of President Abdel-Aziz Bouteflika. It wasn't clear if the bomb was caused by a suicide bomber or if it was planted, the officials said. Later Thursday, Algeria's Interior Minister Noureddine Yazid Zerhouni said \"a suspect person who was among the crowd attempted to go beyond the security cordon,\" but the person escaped \"immediately after the bomb exploded,\" the press service reported. Bouteflika made his visit to Batna as planned, adding a stop at a hospital to visit the wounded before he returned to the capital. There was no immediate claim of responsibility for the bombing. Algeria faces a continuing Islamic insurgency, according to the CIA. In July, 33 people were killed in apparent suicide bombings in Algiers that were claimed by an al Qaeda-affiliated group. Bouteflika said terrorist acts have nothing in common with the noble values of Islam, the press service reported. E-mail to a friend . CNN's Mohammed Tawfeeq contributed to this report.\n\n### TL;DR:",
        "### Article:\nLONDON, England -- Chelsea are waiting on the fitness of John Terry ahead of Wednesday's Champions League match with Valencia, but Frank Lampard has been ruled out. John Terry tries out his protective mask during training for Chelsea on Tuesday. Center-back Terry suffered a broken cheekbone during Saturday's 0-0 draw with Fulham, and Chelsea manager Avram Grant will see how he fares during training on Tuesday before making a decision on his availability. Terry trained at Valencia's Mestalla stadium with a face mask on after surgery on Sunday. \"John Terry wants to play which is very good. Now we need to wait for training and then we will speak with the medical department and decide,\" said Grant. Grant has confirmed that Lampard will definitely sit the game out though as the midfielder continues to recover from his thigh injury. Midfielder Michael Essien, who scored a last-minute winner for Chelsea to knock Valencia out of last season's Champions League, has also been battling a leg injury but he took part in training on Tuesday and is expected to play. E-mail to a friend .\n\n### TL;DR:",
        "### Article:\nPARIS, France (CNN) -- Interpol on Monday took the unprecendented step of making a global appeal for help to identify a man from digitally reconstructed photos taken from the Internet that it said showed him sexually abusing underage boys. This moving image shows how police used software to unscramble the image. (Source: Interpol) The man's face was disguised by digital alteration, but the images were capable of being restored, according to a bulletin from Interpol -- the international police agency based in Lyon, France. Interpol Secretary General Ronald K. Noble said the pictures have been on the the Internet for several years, but investigators have been unable to determine the man's identity or nationality. \"We have tried all other means to identify and to bring him to justice, but we are now convinced that without the public\'s help this sexual predator could continue to rape and sexually abuse young children whose ages appear to range from six to early teens,\" Noble said. He said there is \"very good reason to believe that he travels the world in order to sexually abuse and exploit vulnerable children.\" Interpol has determined the photos were taken in Vietnam and Cambodia. \"The decision to make public this man's picture was not one which was taken lightly,\" said Kristin Kvigne, assistant director of Interpol's Trafficking in Human Beings Unit. The suspect's photo and more information can be seen online at Interpol's Web site. E-mail to a friend .\n\n### TL;DR:",
    ],
    "translation": [
        "### en:\nPARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.\n\n### zh:",
        "### en:\nAt the start of the crisis, many people likened it to 1982 or 1973, which was reassuring, because both dates refer to classical cyclical downturns.\n\n### zh:",
        "### en:\nThe tendency is either excessive restraint (Europe) or a diffusion of the effort (the United States).\n\n### zh:",
        "### en:\nEurope is being cautious in the name of avoiding debt and defending the euro, whereas the US has moved on many fronts in order not to waste an ideal opportunity to implement badly needed structural reforms.\n\n### zh:",
        "### en:\nIndeed, on the surface it seems to be its perfect antithesis: the collapse of a wall symbolizing oppression and artificial divisions versus the collapse of a seemingly indestructible and reassuring institution of financial capitalism.\n\n### zh:",
    ],
}


class RoutingMonitor:
    """Captures expert routing decisions during forward pass"""
    
    def __init__(self, model):
        self.model = model
        self.routing_data = []
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to capture routing decisions"""
        # Helper function to get layers from a model object
        def get_layers_from_model(model_obj):
            """Extract layers from model, handling different model structures"""
            # Try LlamaMoEForCausalLM structure: model.model.layers
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers'):
                return model_obj.model.layers
            # Try LlamaMoEModel structure: model.layers
            elif hasattr(model_obj, 'layers'):
                return model_obj.layers
            else:
                return None
        
        # Handle both regular models and PEFT-wrapped models
        layers = None
        
        if hasattr(self.model, 'get_base_model'):
            # PEFT model - get the base model
            base_model = self.model.get_base_model()
            layers = get_layers_from_model(base_model)
        elif hasattr(self.model, 'base_model'):
            # PEFT model - access base_model directly
            base_model = self.model.base_model
            layers = get_layers_from_model(base_model)
        else:
            # Regular model
            layers = get_layers_from_model(self.model)
        
        if layers is None:
            raise ValueError("Could not find layers in model structure. Model type: " + str(type(self.model)))
        
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                hook = layer.mlp.gate.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output is (topk_idx, topk_weight, aux_loss)
            topk_idx, topk_weight, aux_loss = output
            self.routing_data.append({
                'layer': layer_idx,
                'expert_ids': topk_idx.cpu().numpy(),
                'weights': topk_weight.float().cpu().numpy(),  # Convert bfloat16 to float32 first
            })
        return hook
    
    def clear(self):
        """Clear collected routing data"""
        self.routing_data = []
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def analyze_routing(task_name: str, routing_data: List[Dict]) -> Dict:
    """Analyze routing decisions for a task"""
    all_expert_ids = []
    layer_expert_counts = defaultdict(Counter)
    
    for data in routing_data:
        layer = data['layer']
        expert_ids = data['expert_ids'].flatten()
        all_expert_ids.extend(expert_ids.tolist())
        layer_expert_counts[layer].update(expert_ids.tolist())
    
    # Overall expert distribution
    expert_counter = Counter(all_expert_ids)
    
    # Calculate statistics
    total_routings = len(all_expert_ids)
    expert_distribution = {
        expert_id: count / total_routings 
        for expert_id, count in expert_counter.items()
    }
    
    # Check for collapse
    top_2_experts = expert_counter.most_common(2)
    if len(top_2_experts) >= 2:
        top_2_percentage = sum(count for _, count in top_2_experts) / total_routings
    else:
        top_2_percentage = 1.0
    
    return {
        'task': task_name,
        'total_routings': total_routings,
        'expert_distribution': expert_distribution,
        'expert_counter': expert_counter,
        'top_2_percentage': top_2_percentage,
        'layer_counts': dict(layer_expert_counts),
    }


def print_routing_stats(stats: Dict):
    """Pretty print routing statistics"""
    print(f"\n{'='*60}")
    print(f"Task: {stats['task'].upper()}")
    print(f"{'='*60}")
    print(f"Total routing decisions: {stats['total_routings']}")
    print(f"\nExpert Usage Distribution:")
    
    # Sort by expert ID
    expert_dist = stats['expert_distribution']
    for expert_id in sorted(expert_dist.keys()):
        percentage = expert_dist[expert_id] * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  Expert {expert_id}: {percentage:5.1f}% {bar}")
    
    # Check for collapse
    print(f"\n⚠️  Top 2 experts account for: {stats['top_2_percentage']*100:.1f}% of routings")
    if stats['top_2_percentage'] > 0.7:
        print("  🔴 WARNING: Routing may be collapsing to few experts!")
    elif stats['top_2_percentage'] > 0.5:
        print("  🟡 CAUTION: High concentration in top 2 experts")
    else:
        print("  ✅ GOOD: Routing is well distributed")
    
    # Show activated expert IDs
    top_experts = stats['expert_counter'].most_common(3)
    print(f"\n🎯 Most activated experts: {[f'E{eid}({cnt})' for eid, cnt in top_experts]}")


def compare_tasks(all_stats: List[Dict]):
    """Compare routing patterns across tasks"""
    print(f"\n{'='*80}")
    print("CROSS-TASK ROUTING COMPARISON")
    print(f"{'='*80}")
    
    # Find which experts are dominant for each task
    task_top_experts = {}
    for stats in all_stats:
        top_3 = stats['expert_counter'].most_common(3)
        task_top_experts[stats['task']] = [eid for eid, _ in top_3]
    
    print("\nTop 3 Experts per Task:")
    for task, experts in task_top_experts.items():
        print(f"  {task:15s}: {experts}")
    
    # Check if different tasks use different experts
    unique_top_experts = set()
    for experts in task_top_experts.values():
        unique_top_experts.update(experts[:2])  # Top 2 per task
    
    print(f"\n📊 Unique experts activated across all tasks: {sorted(unique_top_experts)}")
    print(f"   ({len(unique_top_experts)} out of 8 experts)")
    
    if len(unique_top_experts) <= 2:
        print("\n  🔴 CRITICAL: Only 1-2 experts dominate all tasks!")
        print("     → Routing has COLLAPSED. Router is not learning task-specific routing.")
    elif len(unique_top_experts) <= 4:
        print("\n  🟡 WARNING: Limited expert diversity across tasks")
        print("     → Consider increasing load balancing loss or training longer")
    else:
        print("\n  ✅ GOOD: Multiple experts are being used across different tasks")
        print("     → Router is learning task-specific routing patterns")
    
    # Check if tasks have distinct routing patterns
    all_same = len(set(tuple(experts[:2]) for experts in task_top_experts.values())) == 1
    if all_same:
        print("\n  🔴 All tasks route to the SAME experts → No specialization!")
    else:
        print("\n  ✅ Different tasks show different routing patterns → Specialization emerging!")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Check MoE routing sanity')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/llama-3b-moe-mixed-sft/checkpoint-4500',
                       help='Path to the MoE model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (cuda:0, cuda:4, cpu, etc.)')
    parser.add_argument('--tasks', type=str, nargs='+', 
                       default=['math', 'code', 'summarization', 'translation'],
                       help='Tasks to test')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MoE ROUTING SANITY CHECK")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Tasks: {', '.join(args.tasks)}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n❌ ERROR: Model not found at {args.model_path}")
        print("Please provide the correct path to your trained MoE model.")
        return
    
    # Detect PEFT adapter
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config_path)
    
    # Load tokenizer first (use base path if adapter)
    print("📥 Loading tokenizer...")
    try:
        tokenizer_source = args.model_path
        if is_adapter:
            tokenizer_source = PeftConfig.from_pretrained(args.model_path).base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded successfully (vocab_size: {len(tokenizer)})")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Load model
    print(f"\n📥 Loading model on {args.device}...")
    try:
        # Set the specific GPU before loading
        if args.device.startswith('cuda'):
            device_id = args.device.split(':')[1] if ':' in args.device else '0'
            torch.cuda.set_device(int(device_id))
        
        if is_adapter:
            # Try AutoPeftModel first - it handles vocab size mismatches automatically
            try:
                print("   Trying AutoPeftModelForCausalLM (handles vocab size automatically)...")
                model = AutoPeftModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=args.device,
                    trust_remote_code=True,
                )
                model.eval()
                print("   ✅ Model loaded with AutoPeftModelForCausalLM")
            except Exception as e:
                print(f"   AutoPeftModel failed: {e}")
                print("   Falling back to manual PEFT loading...")
                
                # Manual loading: load base model, resize, then load adapter
                peft_cfg = PeftConfig.from_pretrained(args.model_path)
                base_path = peft_cfg.base_model_name_or_path
                base_config = LlamaMoEConfig.from_pretrained(base_path)
                
                # Check what vocab size the adapter expects by inspecting the checkpoint
                adapter_vocab_size = len(tokenizer)  # Default to tokenizer size
                adapter_model_path = os.path.join(args.model_path, "adapter_model.safetensors")
                
                if os.path.exists(adapter_model_path):
                    try:
                        from safetensors import safe_open
                        with safe_open(adapter_model_path, framework="pt") as f:
                            # Check if adapter checkpoint has base model embeddings saved
                            if "base_model.model.model.embed_tokens.weight" in f.keys():
                                adapter_vocab_size = f.get_tensor("base_model.model.model.embed_tokens.weight").shape[0]
                                print(f"   Adapter checkpoint expects vocab_size: {adapter_vocab_size}")
                    except Exception as e:
                        print(f"   Could not inspect adapter: {e}, using tokenizer size")
                
                # Use the larger of tokenizer or adapter expected size
                target_vocab_size = max(len(tokenizer), adapter_vocab_size)
                
                # Load base model with ORIGINAL config first (don't modify config before loading)
                # The checkpoint has embeddings with the original vocab_size
                print(f"   Loading base model from: {base_path}")
                base_model = LlamaMoEForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=torch.bfloat16,
                    device_map=args.device,
                    trust_remote_code=True,
                )
                
                # NOW resize embeddings to match what adapter expects
                current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
                if current_vocab_size != target_vocab_size:
                    print(f"   Resizing base model embeddings: {current_vocab_size} → {target_vocab_size}")
                    base_model.resize_token_embeddings(target_vocab_size)
                    print(f"   ✅ Embeddings resized successfully")
                
                # Load adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    args.model_path,
                    is_trainable=False,
                )
                model.eval()
                print(f"   ✅ LoRA adapter loaded on base model: {base_path}")
        else:
            # Load config first to check vocab size
            config = LlamaMoEConfig.from_pretrained(args.model_path)
            print(f"   Config vocab_size: {config.vocab_size}, Tokenizer vocab_size: {len(tokenizer)}")
            
            # Update config vocab size to match tokenizer if needed
            if config.vocab_size != len(tokenizer):
                print(f"   ⚠️  Vocab size mismatch! Updating config: {config.vocab_size} → {len(tokenizer)}")
                config.vocab_size = len(tokenizer)
            
            model = LlamaMoEForCausalLM.from_pretrained(
                args.model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                trust_remote_code=True,
            )
            if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
            model.eval()
            print(f"✅ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Setup routing monitor
    monitor = RoutingMonitor(model)
    monitor.register_hooks()
    
    # Test each task
    all_stats = []
    for task_name in args.tasks:
        if task_name not in TASK_PROMPTS:
            print(f"\n⚠️  Unknown task: {task_name}, skipping...")
            continue
        
        print(f"\n🔍 Testing task: {task_name.upper()}")
        prompts = TASK_PROMPTS[task_name]

        monitor.clear()
        
        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            # Forward pass (routing data captured by hooks)
            # Disable cache to avoid past_key_values issues
            outputs = model(**inputs, use_cache=False)
            
            print(f"  Sample {i}/{len(prompts)}: {len(monitor.routing_data)} routing decisions captured")
        
        # Analyze routing for this task
        stats = analyze_routing(task_name, monitor.routing_data)
        all_stats.append(stats)
        print_routing_stats(stats)
    
    # Compare across tasks
    if len(all_stats) > 1:
        compare_tasks(all_stats)
    
    # Cleanup
    monitor.remove_hooks()
    
    print(f"\n{'='*80}")
    print("✅ Routing sanity check complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

