import os
import re
import json
import datetime
import time
from pathlib import Path
from dotenv import load_dotenv
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# Try to import OpenAI, handle if not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Install with: pip install openai")

# Load env variables
load_dotenv()

# Test configuration - change these to switch between models and scenario types
MODEL_TYPE = "llama32-3B"  # Options: ['gemma3-12B', 'llama31-8B', 'qwen3-8B'] + ['all']
SCENARIO_TYPE = "topic-detection-en"  # Options: "roleplay-{language}", "topic-detection-{language}", "translation-{origin-language}-{language}", "all"

# Quick test mode - set to True for quick testing with fewer scenarios
QUICK_TEST = False  # Set to False for full testing

# Config
MAX_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.7

# Model configurations
MODEL_CONFIGS = {
    # OpenAI API Models
    "gpt-4.1-mini": {
        "type": "openai",
        "model_name": "gpt-4.1-mini"
    },
    # Local GGUF Models
    # "gemma3n-4B": {
    #     "type": "local",
    #     "repo_id": "unsloth/gemma-3n-E4B-it-GGUF",
    #     "local_dir": "unsloth/gemma-3n-E4B-it-GGUF",
    #     "allow_patterns": ["*UD-Q4_K_XL*", "mmproj-BF16.gguf"],
    #     "model_file": "gemma-3n-E4B-it-UD-Q4_K_XL.gguf",
    #     "chat_format": "gemma"
    # },
    # "gemma3-27B": {
    #     "type": "local",
    #     "repo_id": "unsloth/gemma-3-27b-it-gguf",
    #     "local_dir": "unsloth/gemma-3-27b-it-gguf", 
    #     "allow_patterns": ["*Q4_K_M*"],
    #     "model_file": "gemma-3-27b-it-Q4_K_M.gguf",
    #     "chat_format": "gemma"
    # },
    "llama31-8B": {
        "type": "local",
        "repo_id": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "local_dir": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "allow_patterns": ["*Q4_K_M*"],
        "model_file": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "chat_format": "llama-3"
    },
    "llama32-3B": {
        "type": "local",
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "local_dir": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "allow_patterns": ["*Q4_K_M*"],
        "model_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "chat_format": "llama-3"
    },
    "qwen3-8B": {
        "type": "local",
        "repo_id": "unsloth/Qwen3-8B-GGUF",
        "local_dir": "unsloth/Qwen3-8B-GGUF",
        "allow_patterns": ["*Q4_K_M*"],
        "model_file": "Qwen3-8B-Q4_K_M.gguf",
        "chat_format": "qwen",
        "thinking": False
    },
    "gemma3-12B": {
        "type": "local",
        "repo_id": "unsloth/gemma-3-12b-it-qat-GGUF",
        "local_dir": "unsloth/gemma-3-12b-it-qat-GGUF", 
        "allow_patterns": ["*Q4_K_M*"],
        "model_file": "gemma-3-12b-it-qat-Q4_K_M.gguf",
        "chat_format": "gemma"
    },
}

# Scenario type configurations
SCENARIO_CONFIGS = {
    "roleplay-en": {
        "file_path": "scenarios/roleplay_en.json",
        "description": "Character roleplay scenarios (English)",
        "has_expected_output": False,
        "language": "en"
    },
    "roleplay-fr": {
        "file_path": "scenarios/roleplay_fr.json",
        "description": "Character roleplay scenarios (French)",
        "has_expected_output": False,
        "language": "fr"
    },
    "roleplay-de": {
        "file_path": "scenarios/roleplay_de.json",
        "description": "Character roleplay scenarios (German)",
        "has_expected_output": False,
        "language": "de"
    },
    "topic-detection-en": {
        "file_path": "scenarios/topic-detection_en.json", 
        "description": "Topic classification with confidence scoring (English)",
        "has_expected_output": True,
        "language": "en"
    },
    "topic-detection-fr": {
        "file_path": "scenarios/topic-detection_fr.json", 
        "description": "Topic classification with confidence scoring (French)",
        "has_expected_output": True,
        "language": "fr"
    },
    "topic-detection-de": {
        "file_path": "scenarios/topic-detection_de.json", 
        "description": "Topic classification with confidence scoring (German)",
        "has_expected_output": True,
        "language": "de"
    },
    "translation-en-de": {
        "file_path": "scenarios/translation_en_de.json",
        "description": "Language translation scenarios (English to German)",
        "has_expected_output": False,
        "language": "en",
        "source_language": "en",
        "target_language": "de"
    },
    "translation-en-fr": {
        "file_path": "scenarios/translation_en_fr.json",
        "description": "Language translation scenarios (English to French)",
        "has_expected_output": False,
        "language": "en",
        "source_language": "en",
        "target_language": "fr"
    },
    "translation-de-en": {
        "file_path": "scenarios/translation_de_en.json",
        "description": "Language translation scenarios (German to English)",
        "has_expected_output": False,
        "language": "de",
        "source_language": "de",
        "target_language": "en"
    },
    "translation-fr-en": {
        "file_path": "scenarios/translation_fr_en.json",
        "description": "Language translation scenarios (French to English)",
        "has_expected_output": False,
        "language": "fr",
        "source_language": "fr",
        "target_language": "en"
    }
}

def load_scenarios(scenario_type):
    """Load scenarios from JSON file based on scenario type"""
    config = SCENARIO_CONFIGS[scenario_type]
    file_path = config["file_path"]
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_chat_messages(scenario, model_type):
    """Prepare chat messages for the scenario"""
    messages = []
    
    # Add /think or /no_think if in MODEL_CONFIGS keys
    if "thinking" in MODEL_CONFIGS[model_type].keys():
        if MODEL_CONFIGS[model_type]["thinking"]:
            messages.append({
                "role": "system",
                "content": scenario['personality'] + " /think"
            })
        else:
            messages.append({
                "role": "system",
                "content": scenario['personality'] + " /no_think"
            })
    else:
        # Add system message with personality
        messages.append({
            "role": "system", 
            "content": scenario['personality']
        })
    
    # Add conversation history
    for msg in scenario['conversation_history']:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })
    
    # Add new user message
    messages.append({
        "role": "user",
        "content": scenario['test_prompt']
    })
    
    return messages

def test_scenario_openai(scenario, model_type, scenario_type):
    """Test a single scenario using OpenAI API"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with: pip install openai")
    
    print(f"\n--- Testing: {scenario['name']} ---")
    print(f"Scenario: {scenario['personality'][:100]}...")
    print(f"Test Prompt: {scenario['test_prompt']}")
    
    # Show expected output for topic detection
    if scenario_type.startswith("topic-detection") and 'expected_output' in scenario:
        print(f"Expected: {scenario['expected_output']}")
    
    print("\nResponse:")
    
    # Prepare chat messages (without /think or /no_think for OpenAI)
    messages = []
    messages.append({
        "role": "system", 
        "content": scenario['personality']
    })
    
    # Add conversation history
    for msg in scenario['conversation_history']:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })
    
    # Add new user message
    messages.append({
        "role": "user",
        "content": scenario['test_prompt']
    })
    
    # Performance tracking
    start_time = time.time()
    
    # Get model config
    model_config = MODEL_CONFIGS[model_type]
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        generated_text = response.choices[0].message.content.strip()
        
        print(generated_text)
        print("-" * 50)
        
        # Calculate performance metrics
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Calculate tokens per second
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        
        # Estimate time to first token (simplified estimation)
        time_to_first_token = generation_time * 0.1  # Rough estimate, first 10% of time
        
        # Calculate response length metrics
        response_chars = len(generated_text)
        response_words = len(generated_text.split())
        
        # Prepare result data
        result = {
            "scenario_id": scenario['id'],
            "scenario_name": scenario['name'],
            "prompt": scenario['test_prompt'],
            "response": generated_text,
            "personality": scenario['personality'],
            "conversation_history": scenario['conversation_history'],
            "performance": {
                "generation_time_seconds": round(generation_time, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "time_to_first_token_seconds": round(time_to_first_token, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_length_chars": response_chars,
                "response_length_words": response_words,
                "chars_per_second": round(response_chars / generation_time, 2) if generation_time > 0 else 0,
                "words_per_second": round(response_words / generation_time, 2) if generation_time > 0 else 0
            }
        }
        
        # Add expected output for topic detection
        if scenario_type.startswith("topic-detection") and 'expected_output' in scenario:
            result["expected_output"] = scenario['expected_output']
        
        return result
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def test_scenario(llm, scenario, model_type, scenario_type):
    """Test a single scenario based on its type"""
    print(f"\n--- Testing: {scenario['name']} ---")
    print(f"Scenario: {scenario['personality'][:100]}...")
    print(f"Test Prompt: {scenario['test_prompt']}")
    
    # Show expected output for topic detection
    if scenario_type.startswith("topic-detection") and 'expected_output' in scenario:
        print(f"Expected: {scenario['expected_output']}")
    
    print("\nResponse:")
    
    # Prepare chat messages
    messages = prepare_chat_messages(scenario, model_type)
    
    # Performance tracking
    start_time = time.time()
    
    # Generate response using chat completion
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=False
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        generated_text = response['choices'][0]['message']['content'].strip()
        
        # use regex to remove <think>\n\n</think>\n\n
        generated_text = re.sub(r'<think>\n\n</think>\n\n', '', generated_text)
        
        print(generated_text)
        print("-" * 50)
        
        # Calculate performance metrics
        usage = response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Calculate tokens per second
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        
        # Estimate time to first token (simplified estimation)
        time_to_first_token = generation_time * 0.1  # Rough estimate, first 10% of time
        
        # Calculate response length metrics
        response_chars = len(generated_text)
        response_words = len(generated_text.split())
        
        # Prepare result data
        result = {
            "scenario_id": scenario['id'],
            "scenario_name": scenario['name'],
            "prompt": scenario['test_prompt'],
            "response": generated_text,
            "personality": scenario['personality'],
            "conversation_history": scenario['conversation_history'],
            "performance": {
                "generation_time_seconds": round(generation_time, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "time_to_first_token_seconds": round(time_to_first_token, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_length_chars": response_chars,
                "response_length_words": response_words,
                "chars_per_second": round(response_chars / generation_time, 2) if generation_time > 0 else 0,
                "words_per_second": round(response_words / generation_time, 2) if generation_time > 0 else 0
            }
        }
        
        # Add expected output for topic detection
        if scenario_type.startswith("topic-detection") and 'expected_output' in scenario:
            result["expected_output"] = scenario['expected_output']
        
        return result
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def save_results(results, model_type, scenario_type, performance_summary=None):
    """Save test results to a JSON file"""
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{scenario_type}_test_{model_type}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Get scenario config for language information
    scenario_config = SCENARIO_CONFIGS.get(scenario_type, {})
    
    # Prepare data to save
    test_data = {
        "model_type": model_type,
        "scenario_type": scenario_type,
        "timestamp": timestamp,
        "test_type": scenario_type,
        "language": scenario_config.get("language", "unknown"),
        "results": results
    }
    
    # Add source and target languages for translation scenarios
    if scenario_type.startswith("translation"):
        test_data["source_language"] = scenario_config.get("source_language", "unknown")
        test_data["target_language"] = scenario_config.get("target_language", "unknown")
    
    # Add performance summary if provided
    if performance_summary:
        test_data["performance_summary"] = performance_summary
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def compare_results():
    """Load and compare results from different models and scenario types"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return
    
    result_files = list(results_dir.glob("*_test_*.json"))
    if not result_files:
        print("No test results found.")
        return
    
    # Group files by scenario type
    scenario_groups = {}
    for file in result_files:
        parts = file.stem.split('_')
        if len(parts) >= 3:
            scenario_type = parts[0]
            if scenario_type not in scenario_groups:
                scenario_groups[scenario_type] = []
            scenario_groups[scenario_type].append(file)
    
    print(f"\nFound results for {len(scenario_groups)} scenario types:")
    for scenario_type, files in scenario_groups.items():
        print(f"  {scenario_type}: {len(files)} files")
        for file in files:
            print(f"    - {file.name}")
    
    return result_files

def run_single_test(model_type, scenario_type):
    """Run a single test for a specific model and scenario type"""
    # Get configurations
    model_config = MODEL_CONFIGS[model_type]
    scenario_config = SCENARIO_CONFIGS[scenario_type]
    
    print(f"\n=== {scenario_config['description']} Testing ===")
    print(f"Model: {model_type}")
    print(f"Scenario Type: {scenario_type}")
    
    # Check if this is an OpenAI model or local model
    is_openai_model = model_config.get("type") == "openai"
    
    # Initialize model-specific variables
    download_time = 0
    model_load_time = 0
    llm = None
    
    if is_openai_model:
        print(f"Using OpenAI API model: {model_config['model_name']}")
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        # Set up OpenAI API key from environment
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        print("OpenAI API ready")
        
    else:
        # Handle local model setup
        print(f"Downloading {model_type} model...")
        download_start = time.time()
        snapshot_download(
            repo_id=model_config["repo_id"],
            local_dir=model_config["local_dir"],
            allow_patterns=model_config["allow_patterns"]
        )
        download_time = time.time() - download_start
        
        # Initialize the local model
        model_path = f"./{model_config['local_dir']}/{model_config['model_file']}"
        print(f"Loading model from: {model_path}")
        
        model_load_start = time.time()
        llm = Llama(
            model_path=model_path,
            chat_format=model_config["chat_format"],
            n_ctx=1024,
            n_batch=512,
            n_gpu_layers=20,
            n_threads=8,
            verbose=True,
        )
        model_load_time = time.time() - model_load_start
    
    # Load scenarios
    print(f"Loading {scenario_type} scenarios...")
    scenarios_data = load_scenarios(scenario_type)
    scenarios = scenarios_data['scenarios']
    
    print(f"Found {len(scenarios)} {scenario_type} scenarios")
    
    # For quick testing, limit to first 2 scenarios
    if QUICK_TEST:
        scenarios = scenarios[:2]
        print(f"Quick test mode: Testing only first {len(scenarios)} scenarios")
    
    # Test each scenario
    results = []
    total_test_start = time.time()
    
    for scenario in scenarios:
        if is_openai_model:
            result = test_scenario_openai(scenario, model_type, scenario_type)
        else:
            result = test_scenario(llm, scenario, model_type, scenario_type)
        
        if result:
            results.append(result)
    
    total_test_time = time.time() - total_test_start
    
    # Calculate aggregate performance metrics
    if results:
        total_tokens = sum(r['performance']['total_tokens'] for r in results)
        total_generation_time = sum(r['performance']['generation_time_seconds'] for r in results)
        avg_tokens_per_second = sum(r['performance']['tokens_per_second'] for r in results) / len(results)
        avg_generation_time = total_generation_time / len(results)
        
        performance_summary = {
            "model_download_time_seconds": round(download_time, 3),
            "model_load_time_seconds": round(model_load_time, 3),
            "total_test_time_seconds": round(total_test_time, 3),
            "total_scenarios_tested": len(results),
            "average_generation_time_seconds": round(avg_generation_time, 3),
            "average_tokens_per_second": round(avg_tokens_per_second, 2),
            "total_tokens_generated": total_tokens,
            "total_generation_time_seconds": round(total_generation_time, 3)
        }
    else:
        performance_summary = {}
    
    # Save results
    save_results(results, model_type, scenario_type, performance_summary)
    
    print(f"\nCompleted testing {len(results)} scenarios for {model_type} on {scenario_type}")
    
    # Clean up the model to free memory (only for local models)
    if not is_openai_model and llm is not None:
        del llm
        import gc
        gc.collect()
    
    return results

def get_test_combinations():
    """Get all model/scenario combinations to test based on current configuration"""
    # Determine which models to test
    if MODEL_TYPE == "all":
        models_to_test = list(MODEL_CONFIGS.keys())
    else:
        if MODEL_TYPE not in MODEL_CONFIGS:
            print(f"Error: Unknown model type '{MODEL_TYPE}'")
            print(f"Available types: {list(MODEL_CONFIGS.keys())}")
            return []
        models_to_test = [MODEL_TYPE]
    
    # Determine which scenarios to test
    if SCENARIO_TYPE == "all":
        # Get all scenario types excluding the backward compatibility aliases
        scenarios_to_test = [
            key for key in SCENARIO_CONFIGS.keys() 
            if key not in ["roleplay", "topic-detection", "translation"]
        ]
    else:
        if SCENARIO_TYPE not in SCENARIO_CONFIGS:
            print(f"Error: Unknown scenario type '{SCENARIO_TYPE}'")
            print(f"Available types: {list(SCENARIO_CONFIGS.keys())}")
            return []
        scenarios_to_test = [SCENARIO_TYPE]
    
    # Create all combinations
    combinations = []
    for model in models_to_test:
        for scenario in scenarios_to_test:
            combinations.append((model, scenario))
    
    return combinations

def run_tests():
    """Main function to run all tests based on selected model and scenario types"""
    # Get all combinations to test
    combinations = get_test_combinations()
    
    if not combinations:
        return
    
    print(f"=== LLM Testing Framework ===")
    print(f"Running {len(combinations)} test combinations:")
    
    for i, (model, scenario) in enumerate(combinations, 1):
        print(f"  {i}. {model} × {scenario}")
    
    print(f"\nQuick Test Mode: {'ON' if QUICK_TEST else 'OFF'}")
    if QUICK_TEST:
        print("  (Testing only first 2 scenarios per combination)")
    
    # Run all combinations
    all_results = []
    failed_combinations = []
    
    for i, (model_type, scenario_type) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING COMBINATION {i}/{len(combinations)}: {model_type} × {scenario_type}")
        print(f"{'='*60}")
        
        try:
            results = run_single_test(model_type, scenario_type)
            all_results.append((model_type, scenario_type, results))
            print(f"✓ Successfully completed {model_type} × {scenario_type}")
        except Exception as e:
            print(f"✗ Failed {model_type} × {scenario_type}: {e}")
            failed_combinations.append((model_type, scenario_type, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Total combinations attempted: {len(combinations)}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(failed_combinations)}")
    
    if failed_combinations:
        print(f"\nFailed combinations:")
        for model, scenario, error in failed_combinations:
            print(f"  ✗ {model} × {scenario}: {error}")
    
    if all_results:
        print(f"\nSuccessful combinations:")
        for model, scenario, results in all_results:
            print(f"  ✓ {model} × {scenario}: {len(results)} scenarios tested")
    
    print(f"\nYou can change MODEL_TYPE and SCENARIO_TYPE variables to test different combinations!")
    print(f"Set MODEL_TYPE='all' and SCENARIO_TYPE='all' to test all combinations.")
    
    return all_results

if __name__ == "__main__":
    print("=== LLM Testing Framework ===")
    print(f"Model: {MODEL_TYPE}")
    print(f"Scenario Type: {SCENARIO_TYPE}")
    
    if MODEL_TYPE != "all" and SCENARIO_TYPE != "all":
        print(f"Description: {SCENARIO_CONFIGS[SCENARIO_TYPE]['description']}")
    
    print("\nAvailable commands:")
    print("1. Run tests: python test.py")
    print("2. Change MODEL_TYPE and SCENARIO_TYPE variables to test different combinations")
    print("3. Set MODEL_TYPE='all' to test all models")
    print("4. Set SCENARIO_TYPE='all' to test all scenarios")
    print("5. Set both to 'all' to test all combinations")
    
    print(f"\nAvailable Models: {list(MODEL_CONFIGS.keys())} + ['all']")
    print(f"  OpenAI API Models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
    print(f"  Local GGUF Models: llama31-8B, qwen3-8B, gemma3-12B")
    print(f"Available Scenario Types:")
    print("  Roleplay: roleplay-en, roleplay-fr, roleplay-de, roleplay (alias for en)")
    print("  Topic Detection: topic-detection-en, topic-detection-fr, topic-detection-de, topic-detection (alias for en)")
    print("  Translation: translation-en-de, translation-en-fr, translation-de-en, translation-fr-en, translation (alias for en-de)")
    print("  All: 'all' (excludes backward compatibility aliases)")
    
    print(f"\nCurrent Configuration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Scenario: {SCENARIO_TYPE}")
    
    # Check OpenAI setup
    if any(MODEL_CONFIGS[m].get("type") == "openai" for m in MODEL_CONFIGS.keys()):
        if OPENAI_AVAILABLE:
            if os.getenv("OPENAI_API_KEY"):
                print(f"  OpenAI API: Ready ✓")
            else:
                print(f"  OpenAI API: Missing OPENAI_API_KEY environment variable ⚠️")
        else:
            print(f"  OpenAI API: Package not installed ⚠️ (pip install openai)")
    
    if MODEL_TYPE == "all" or SCENARIO_TYPE == "all":
        combinations = get_test_combinations()
        print(f"  Will run {len(combinations)} combinations:")
        for model, scenario in combinations[:5]:  # Show first 5
            model_type = MODEL_CONFIGS[model].get("type", "local")
            print(f"    - {model} ({model_type}) × {scenario}")
        if len(combinations) > 5:
            print(f"    ... and {len(combinations) - 5} more")
    elif SCENARIO_TYPE in SCENARIO_CONFIGS:
        print(f"  Description: {SCENARIO_CONFIGS[SCENARIO_TYPE]['description']}")
    
    run_tests()