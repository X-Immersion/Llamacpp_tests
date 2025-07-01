# LLM Roleplay Testing Framework

This framework allows you to systematically test and compare different language models on roleplay scenarios with specific personalities and conversation contexts.

## Files Structure

```
llamacpp_test/
├── test.py                    # Main testing script
├── compare_results.py         # Results comparison script
├── scenarios/
│   └── roleplay_en.json      # 10 roleplay scenarios
└── results/                   # Generated test results
    ├── roleplay_test_*.json  # Individual test results
    └── comparison_report_*.html # HTML comparison reports
```

## Usage

### 1. Run Tests for a Model

Edit `test.py` and change the `MODEL_TYPE` variable:

```python
MODEL_TYPE = "gemma3-small"  # Options: "gemma3n", "gemma3", "gemma3-small", "llama31"
```

Then run:
```bash
python test.py
```

This will:
- Download the selected model
- Test all 10 roleplay scenarios
- Save results to `results/roleplay_test_[model]_[timestamp].json`

### 2. Test Multiple Models

To compare models, run the test for each model separately:

```bash
# Test first model
MODEL_TYPE="gemma3-small" python test.py

# Change MODEL_TYPE in script, then test second model
MODEL_TYPE="llama31" python test.py

# etc...
```

### 3. Compare Results

After testing multiple models:

```bash
python compare_results.py
```

This will:
- Generate a console comparison report
- Create an HTML report in `results/comparison_report_[timestamp].html`
- Analyze response quality metrics

## Roleplay Scenarios

The framework includes 10 diverse roleplay scenarios:

1. **Medieval Blacksmith** - Gruff craftsman character
2. **Space Station AI** - Logical but quirky AI
3. **Elderly Librarian** - Knowledgeable academic
4. **Quirky Chef** - Passionate French chef
5. **Film Noir Detective** - Cynical 1940s investigator
6. **Young Wizard Student** - Enthusiastic magical apprentice
7. **Victorian Gentleman** - Formal 19th-century aristocrat
8. **Pirate Captain** - Strategic sea captain
9. **Robot Butler** - Polite household android
10. **Mystical Oracle** - Enigmatic fortune teller

Each scenario includes:
- Detailed personality description
- Conversation history for context
- Test prompt to evaluate response

## Results Format

Test results are saved as JSON files containing:

```json
{
  "model_type": "gemma3-small",
  "timestamp": "20231215_143022",
  "test_type": "roleplay",
  "results": [
    {
      "scenario_id": "medieval_blacksmith",
      "scenario_name": "Medieval Blacksmith",
      "prompt": "The sword needs to be lightweight but durable...",
      "response": "Aye, for swamplands ye say? Well now...",
      "personality": "You are Gareth, a gruff but skilled blacksmith...",
      "conversation_history": [...]
    }
  ]
}
```

## Evaluation Criteria

When comparing results, consider:

1. **Character Consistency**: Does the model maintain the character's personality?
2. **Context Awareness**: Does it reference the conversation history appropriately?
3. **Response Quality**: Is the response engaging and well-written?
4. **Accuracy**: Does it stay true to the character's knowledge/time period?
5. **Creativity**: How creative and interesting are the responses?

## Adding New Scenarios

To add new roleplay scenarios, edit `scenarios/roleplay_en.json`:

```json
{
  "id": "new_character",
  "name": "Character Name",
  "personality": "Character description and speaking style...",
  "conversation_history": [
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous character response"}
  ],
  "test_prompt": "New prompt to test the character"
}
```

## Model Configuration

To add support for new models, edit the `MODEL_CONFIGS` dictionary in `test.py`:

```python
"new_model": {
    "repo_id": "huggingface/model-repo",
    "local_dir": "local/directory",
    "allow_patterns": ["*pattern*"],
    "model_file": "model_filename.gguf"
}
```

## Tips for Testing

1. **Consistent Testing**: Use the same scenarios across all models
2. **Multiple Runs**: Consider running tests multiple times with different random seeds
3. **Parameter Tuning**: Adjust temperature and other parameters for fair comparison
4. **Documentation**: Keep notes about interesting observations during testing

## Troubleshooting

- **Model Download Issues**: Check internet connection and Hugging Face credentials
- **Memory Issues**: Reduce `n_ctx` or `n_gpu_layers` parameters
- **Response Quality**: Adjust `temperature`, `top_p`, and `max_tokens` parameters
- **Format Issues**: Ensure the prompt format matches your model's expected format
