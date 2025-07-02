#!/usr/bin/env python3
"""
Script to compare test results between different models and scenario types
Generates a comprehensive HTML report with tabs for all scenario types
"""

import json
import os
from pathlib import Path
from datetime import datetime
import re

def analyze_topic_detection_response(response, expected_output=None):
    """Analyze a topic detection response for JSON validity and topic correctness"""
    analysis = {
        'is_valid_json': False,
        'topic_matches': False,
        'response_json': None,
        'expected_json': None,
        'confidence': None
    }
    
    # Check if response is valid JSON
    try:
        response_json = json.loads(response.strip())
        analysis['is_valid_json'] = True
        analysis['response_json'] = response_json
        analysis['confidence'] = response_json.get('confidence')
    except json.JSONDecodeError:
        return analysis
    
    # Check if expected output is provided and valid JSON
    if expected_output:
        try:
            expected_json = json.loads(expected_output)
            analysis['expected_json'] = expected_json
            
            # Check if topics match
            if (response_json.get('topic') == expected_json.get('topic')):
                analysis['topic_matches'] = True
        except json.JSONDecodeError:
            pass
    
    return analysis

def get_latest_results_by_model_and_scenario():
    """Get result files for each model and scenario type combination"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return {}
    
    # Pattern: {scenario_type}_test_{model}.json (new format without timestamp)
    result_files = list(results_dir.glob("*_test_*.json"))
    
    # Group by model and scenario type
    results = {}
    
    for file in result_files:
        # Parse filename: scenario_test_model.json  
        parts = file.stem.split('_')
        if len(parts) < 3:
            continue
            
        # Find 'test' part to split scenario type and model
        if 'test' not in parts:
            continue
            
        test_index = parts.index('test')
        scenario_type = '_'.join(parts[:test_index])
        model = '_'.join(parts[test_index + 1:])
        
        key = (scenario_type, model)
        
        # Get timestamp from file content if available, otherwise use file modification time
        timestamp = file.stat().st_mtime
        
        results[key] = {
            'file': file,
            'timestamp': timestamp,
            'scenario_type': scenario_type,
            'model': model
        }
    
    # sort by file
    results = dict(sorted(results.items(), key=lambda item: item[1]['file']))
    
    return results

def load_result_data(file_path):
    """Load and return result data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def organize_results_by_scenario_type(results):
    """Organize results by main scenario type and language for nested tabbed interface"""
    organized = {}
    
    for (scenario_type, model), info in results.items():
        data = load_result_data(info['file'])
        if not data:
            continue
        
        # Extract main scenario type and language
        main_type = scenario_type.split('-')[0]  # roleplay, translation, topic
        if main_type == "topic":
            main_type = "topic-detection"
        
        # Get language information
        language = data.get('language', 'unknown')
        source_lang = data.get('source_language')
        target_lang = data.get('target_language')
        
        # Create language identifier
        if main_type == "translation" and source_lang and target_lang:
            lang_key = f"{source_lang}-{target_lang}"
        else:
            lang_key = language
            
        if main_type not in organized:
            organized[main_type] = {}
            
        if lang_key not in organized[main_type]:
            organized[main_type][lang_key] = {}
            
        organized[main_type][lang_key][model] = {
            'data': data,
            'timestamp': info['timestamp'],
            'filename': info['file'].name,
            'full_scenario_type': scenario_type
        }
    
    return organized

def create_comprehensive_html_report(organized_results):
    """Create a comprehensive HTML report with nested tabs for scenario types and languages"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/comprehensive_comparison_report_{timestamp}.html"
    
    # Get all main scenario types
    main_scenario_types = list(organized_results.keys())
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Comprehensive Comparison Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #007acc, #005aa3);
            color: white;
            text-align: center;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        /* Main Tab Navigation */
        .main-tab-container {{
            background-color: white;
            border-bottom: 2px solid #e0e0e0;
        }}
        .main-tab-nav {{
            display: flex;
            overflow-x: auto;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .main-tab-button {{
            background: none;
            border: none;
            padding: 20px 30px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 600;
            color: #666;
            border-bottom: 4px solid transparent;
            white-space: nowrap;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .main-tab-button:hover {{
            background-color: #f8f9fa;
            color: #007acc;
        }}
        .main-tab-button.active {{
            color: #007acc;
            border-bottom-color: #007acc;
            background-color: #f8f9fa;
        }}
        
        /* Sub Tab Navigation */
        .sub-tab-container {{
            background-color: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            padding: 0 30px;
        }}
        .sub-tab-nav {{
            display: flex;
            overflow-x: auto;
            gap: 5px;
        }}
        .sub-tab-button {{
            background: none;
            border: none;
            padding: 12px 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .sub-tab-button:hover {{
            background-color: white;
            color: #007acc;
            border-radius: 6px 6px 0 0;
        }}
        .sub-tab-button.active {{
            color: #007acc;
            border-bottom-color: #007acc;
            background-color: white;
            border-radius: 6px 6px 0 0;
        }}
        
        /* Main Tab Content */
        .main-tab-content {{
            display: none;
        }}
        .main-tab-content.active {{
            display: block;
        }}
        
        /* Sub Tab Content */
        .sub-tab-content {{
            display: none;
            padding: 30px;
        }}
        .sub-tab-content.active {{
            display: block;
        }}
        
        /* Model Summary */
        .model-summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #007acc;
        }}
        .model-summary h3 {{
            margin-top: 0;
            color: #333;
        }}
        .model-summary table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .model-summary th, .model-summary td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .model-summary th {{
            background-color: #007acc;
            color: white;
            font-weight: 600;
        }}
        .model-summary tr:last-child td {{
            border-bottom: none;
        }}
        
        /* Scenario Cards */
        .scenario {{ 
            margin-bottom: 30px; 
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background-color: white;
        }}
        .scenario-header {{
            background: linear-gradient(135deg, #007acc, #0066cc);
            color: white;
            padding: 20px 25px;
            font-size: 18px; 
            font-weight: 600;
        }}
        .scenario-content {{
            padding: 25px;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .section-title {{
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        .description {{ 
            background-color: #f8f9fa; 
            padding: 18px; 
            border-radius: 8px;
            border-left: 4px solid #007acc;
            font-style: italic;
            line-height: 1.6;
        }}
        .conversation-history {{
            background-color: #fff3cd;
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }}
        .history-message {{
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 6px;
            line-height: 1.5;
        }}
        .user-message {{
            background-color: #e3f2fd;
            border-left: 3px solid #2196f3;
        }}
        .assistant-message {{
            background-color: #f3e5f5;
            border-left: 3px solid #9c27b0;
        }}
        .test-prompt {{ 
            background-color: #e8f5e8; 
            padding: 18px; 
            border-radius: 8px;
            border-left: 4px solid #28a745;
            font-weight: 500;
            line-height: 1.6;
        }}
        .expected-output {{
            background-color: #fff3e0;
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
        }}
        .responses-container {{
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }}
        .responses-container.two-models {{
            grid-template-columns: 1fr 1fr;
        }}
        .responses-container.three-plus-models {{
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        }}
        .response {{
            background-color: #fafafa;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
        }}
        .model-name {{ 
            background: linear-gradient(135deg, #0066cc, #004499);
            color: white;
            font-weight: 600; 
            padding: 15px 18px;
            margin: 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .model-response {{ 
            padding: 18px; 
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.5;
            min-height: 120px;
            background-color: white;
        }}
        
        /* Topic Detection Indicators */
        .topic-indicators {{
            padding: 8px 18px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }}
        .indicator {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            cursor: help;
        }}
        .json-indicator {{
            background-color: #e3f2fd;
            color: #1976d2;
        }}
        .topic-indicator {{
            background-color: #e8f5e8;
            color: #388e3c;
        }}
        
        /* Model Toggle Controls */
        .model-toggles {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
        }}
        .model-toggles h4 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .toggle-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .toggle-button {{
            background-color: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .toggle-button:hover {{
            background-color: #005aa3;
            transform: translateY(-1px);
        }}
        .toggle-button.inactive {{
            background-color: #6c757d;
            opacity: 0.6;
        }}
        .toggle-button.inactive:hover {{
            background-color: #5a6268;
        }}
        .response.hidden {{
            display: none !important;
        }}
        
        /* Difficulty Toggle Controls */
        .difficulty-toggles {{
            background-color: #fff8e7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ff9800;
        }}
        .difficulty-toggles h4 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .difficulty-button {{
            background-color: #ff9800;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .difficulty-button:hover {{
            background-color: #f57c00;
            transform: translateY(-1px);
        }}
        .difficulty-button.inactive {{
            background-color: #6c757d;
            opacity: 0.6;
        }}
        .difficulty-button.inactive:hover {{
            background-color: #5a6268;
        }}
        .scenario.hidden {{
            display: none !important;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .responses-container.two-models {{
                grid-template-columns: 1fr;
            }}
            .main-tab-nav, .sub-tab-nav {{
                flex-wrap: wrap;
            }}
            .container {{
                margin: 0 10px;
            }}
            .toggle-controls {{
                justify-content: center;
            }}
        }}
    </style>
    <script>
        function showMainTab(mainTabName) {{
            // Hide all main tab contents
            const mainTabContents = document.getElementsByClassName('main-tab-content');
            for (let i = 0; i < mainTabContents.length; i++) {{
                mainTabContents[i].classList.remove('active');
            }}
            
            // Remove active class from all main tab buttons
            const mainTabButtons = document.getElementsByClassName('main-tab-button');
            for (let i = 0; i < mainTabButtons.length; i++) {{
                mainTabButtons[i].classList.remove('active');
            }}
            
            // Show selected main tab content
            document.getElementById(mainTabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
            
            // Show first sub-tab by default
            const firstSubTab = document.querySelector(`#${{mainTabName}} .sub-tab-button`);
            if (firstSubTab) {{
                firstSubTab.click();
            }}
        }}
        
        function showSubTab(mainTabName, subTabName) {{
            // Hide all sub tab contents within the main tab
            const subTabContents = document.querySelectorAll(`#${{mainTabName}} .sub-tab-content`);
            subTabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all sub tab buttons within the main tab
            const subTabButtons = document.querySelectorAll(`#${{mainTabName}} .sub-tab-button`);
            subTabButtons.forEach(button => button.classList.remove('active'));
            
            // Show selected sub tab content
            document.getElementById(subTabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }}
        
        function toggleModel(tabId, modelName) {{
            const toggleButton = document.querySelector(`#${{tabId}} .toggle-button[data-model="${{modelName}}"]`);
            const responseElements = document.querySelectorAll(`#${{tabId}} .response[data-model="${{modelName}}"]`);
            
            if (toggleButton.classList.contains('inactive')) {{
                // Show model
                toggleButton.classList.remove('inactive');
                responseElements.forEach(el => el.classList.remove('hidden'));
            }} else {{
                // Hide model
                toggleButton.classList.add('inactive');
                responseElements.forEach(el => el.classList.add('hidden'));
            }}
            
            // Update grid layout based on visible responses
            updateResponseLayout(tabId);
        }}
        
        function updateResponseLayout(tabId) {{
            const scenarios = document.querySelectorAll(`#${{tabId}} .scenario`);
            scenarios.forEach(scenario => {{
                const responsesContainer = scenario.querySelector('.responses-container');
                const visibleResponses = scenario.querySelectorAll('.response:not(.hidden)');
                const numVisible = visibleResponses.length;
                
                // Remove existing grid classes
                responsesContainer.classList.remove('two-models', 'three-plus-models');
                
                // Add appropriate grid class based on number of visible responses
                if (numVisible === 2) {{
                    responsesContainer.classList.add('two-models');
                }} else if (numVisible >= 3) {{
                    responsesContainer.classList.add('three-plus-models');
                }}
            }});
        }}
        
        function selectAllModels(tabId) {{
            const toggleButtons = document.querySelectorAll(`#${{tabId}} .toggle-button[data-model]`);
            toggleButtons.forEach(button => {{
                if (button.classList.contains('inactive')) {{
                    const modelName = button.getAttribute('data-model');
                    toggleModel(tabId, modelName);
                }}
            }});
        }}
        
        function toggleDifficulty(tabId, difficulty) {{
            const difficultyButton = document.querySelector(`#${{tabId}} .difficulty-button[data-difficulty="${{difficulty}}"]`);
            const scenarioElements = document.querySelectorAll(`#${{tabId}} .scenario[data-difficulty="${{difficulty}}"]`);
            
            if (difficultyButton.classList.contains('inactive')) {{
                // Show difficulty
                difficultyButton.classList.remove('inactive');
                scenarioElements.forEach(el => el.classList.remove('hidden'));
            }} else {{
                // Hide difficulty
                difficultyButton.classList.add('inactive');
                scenarioElements.forEach(el => el.classList.add('hidden'));
            }}
        }}
        
        function selectAllDifficulties(tabId) {{
            const difficultyButtons = document.querySelectorAll(`#${{tabId}} .difficulty-button[data-difficulty]`);
            difficultyButtons.forEach(button => {{
                if (button.classList.contains('inactive')) {{
                    const difficulty = button.getAttribute('data-difficulty');
                    toggleDifficulty(tabId, difficulty);
                }}
            }});
        }}
        
        function extractDifficulty(scenarioName) {{
            const difficulties = ['Very Hard', 'Hard', 'Medium', 'Easy'];
            for (const diff of difficulties) {{
                if (scenarioName.includes(diff)) {{
                    return diff;
                }}
            }}
            return 'Unknown';
        }}
        
        // Show first main tab by default when page loads
        window.onload = function() {{
            const firstMainTab = document.querySelector('.main-tab-button');
            if (firstMainTab) {{
                firstMainTab.click();
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LLM Comprehensive Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Scenario Types: {', '.join([s.replace('-', ' ').title() for s in main_scenario_types])}</p>
        </div>
        
        <div class="main-tab-container">
            <div class="main-tab-nav">"""
    
    # Add main tab buttons
    for main_type in main_scenario_types:
        display_name = main_type.replace('-', ' ').title()
        html_content += f"""
                <button class="main-tab-button" onclick="showMainTab('{main_type}')">{display_name}</button>"""
    
    html_content += """
            </div>
        </div>"""
    
    # Add main tab content for each scenario type
    for main_type, languages_data in organized_results.items():
        html_content += f"""
        <div id="{main_type}" class="main-tab-content">
            <div class="sub-tab-container">
                <div class="sub-tab-nav">"""
        
        # Add sub-tab buttons for each language
        for lang_key in languages_data.keys():
            lang_display = lang_key.upper() if '-' not in lang_key else lang_key.replace('-', '→').upper()
            tab_id = f"{main_type}-{lang_key}"
            html_content += f"""
                    <button class="sub-tab-button" onclick="showSubTab('{main_type}', '{tab_id}')">{lang_display}</button>"""
        
        html_content += """
                </div>
            </div>"""
        
        # Add sub-tab content for each language
        for lang_key, models_data in languages_data.items():
            tab_id = f"{main_type}-{lang_key}"
            lang_display = lang_key.upper() if '-' not in lang_key else lang_key.replace('-', '→').upper()
            
            html_content += f"""
            <div id="{tab_id}" class="sub-tab-content">
                <div class="model-summary">
                    <h3>{main_type.replace('-', ' ').title()} Results Summary ({lang_display})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Scenarios</th>
                                <th>Avg Tokens/s</th>
                                <th>Avg Gen Time</th>
                                <th>Total Tokens</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>"""
            
            for model, info in models_data.items():
                num_scenarios = len(info['data']['results'])
                timestamp = info['timestamp']
                data = info['data']
                
                # Extract performance metrics
                perf_summary = data.get('performance_summary', {})
                avg_tokens_per_sec = perf_summary.get('average_tokens_per_second', 0)
                avg_gen_time = perf_summary.get('average_generation_time_seconds', 0)
                total_tokens = perf_summary.get('total_tokens_generated', 0)
                
                html_content += f"""
                            <tr>
                                <td><strong>{model}</strong></td>
                                <td>{num_scenarios}</td>
                                <td>{avg_tokens_per_sec:.1f}</td>
                                <td>{avg_gen_time:.2f}s</td>
                                <td>{total_tokens}</td>
                                <td>{timestamp}</td>
                            </tr>"""
            
            html_content += """
                        </tbody>
                    </table>
                </div>"""
            
            # Add performance details section
            html_content += f"""
                <div class="model-summary">
                    <h3>Performance Details</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Model Load Time</th>
                                <th>Total Test Time</th>
                                <th>Avg Response Length</th>
                                <th>Words/Second</th>
                                <th>File</th>
                            </tr>
                        </thead>
                        <tbody>"""
            
            for model, info in models_data.items():
                data = info['data']
                filename = info['filename']
                
                # Extract performance metrics
                perf_summary = data.get('performance_summary', {})
                model_load_time = perf_summary.get('model_load_time_seconds', 0)
                total_test_time = perf_summary.get('total_test_time_seconds', 0)
                
                # Calculate average response metrics from individual results
                results = data.get('results', [])
                if results:
                    avg_response_chars = sum(r.get('performance', {}).get('response_length_chars', 0) for r in results) / len(results)
                    avg_words_per_sec = sum(r.get('performance', {}).get('words_per_second', 0) for r in results) / len(results)
                else:
                    avg_response_chars = 0
                    avg_words_per_sec = 0
                
                html_content += f"""
                            <tr>
                                <td><strong>{model}</strong></td>
                                <td>{model_load_time:.2f}s</td>
                                <td>{total_test_time:.2f}s</td>
                                <td>{avg_response_chars:.0f} chars</td>
                                <td>{avg_words_per_sec:.1f}</td>
                                <td>{filename}</td>
                            </tr>"""
            
            html_content += """
                        </tbody>
                    </table>
                </div>"""
            
            # Add topic detection accuracy table for topic-detection scenarios
            if main_type == "topic-detection":
                html_content += f"""
                <div class="model-summary">
                    <h3>Topic Detection Accuracy</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Valid JSON Responses</th>
                                <th>Correct Topics</th>
                                <th>Topic Accuracy</th>
                                <th>Avg Confidence</th>
                            </tr>
                        </thead>
                        <tbody>"""
                
                for model, info in models_data.items():
                    results = info['data']['results']
                    
                    # Calculate topic detection metrics
                    valid_json_count = 0
                    correct_topics = 0
                    confidence_scores = []
                    
                    for result in results:
                        response = result['response']
                        expected = result.get('expected_output')
                        
                        analysis = analyze_topic_detection_response(response, expected)
                        
                        if analysis['is_valid_json']:
                            valid_json_count += 1
                            
                        if analysis['confidence'] is not None:
                            confidence_scores.append(analysis['confidence'])
                            
                        if analysis['topic_matches']:
                            correct_topics += 1
                    
                    total_scenarios = len(results)
                    json_percentage = (valid_json_count / total_scenarios) * 100 if total_scenarios else 0
                    topic_accuracy = (correct_topics / total_scenarios) * 100 if total_scenarios else 0
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    
                    html_content += f"""
                            <tr>
                                <td><strong>{model}</strong></td>
                                <td>{valid_json_count}/{total_scenarios} ({json_percentage:.1f}%)</td>
                                <td>{correct_topics}/{total_scenarios}</td>
                                <td>{topic_accuracy:.1f}%</td>
                                <td>{avg_confidence:.1f}</td>
                            </tr>"""
                
                html_content += """
                        </tbody>
                    </table>
                </div>"""
            
            # Add model toggle controls
            models_list = list(models_data.keys())
            html_content += f"""
                <div class="model-toggles">
                    <h4>Model Response Visibility Controls</h4>
                    <div class="toggle-controls">
                        <button onclick="selectAllModels('{tab_id}')" class="toggle-button" style="background-color: #28a745;">Show All</button>"""
            
            for model in models_list:
                html_content += f"""
                        <button onclick="toggleModel('{tab_id}', '{model}')" class="toggle-button" data-model="{model}">{model.upper()}</button>"""
            
            html_content += """
                    </div>
                </div>"""
            
            # Organize scenarios for comparison and extract difficulties
            scenario_comparison = {}
            all_difficulties = set()
            for model, info in models_data.items():
                for scenario_result in info['data']['results']:
                    scenario_id = scenario_result['scenario_id']
                    scenario_name = scenario_result['scenario_name']
                    
                    # Extract difficulty from scenario name
                    difficulty = 'Unknown'
                    for diff in ['Very Hard', 'Hard', 'Medium', 'Easy']:
                        if diff in scenario_name:
                            difficulty = diff
                            break
                    all_difficulties.add(difficulty)
                    
                    if scenario_id not in scenario_comparison:
                        scenario_comparison[scenario_id] = {
                            'name': scenario_name,
                            'difficulty': difficulty,
                            'prompt': scenario_result['prompt'],
                            'personality': scenario_result.get('personality', ''),
                            'conversation_history': scenario_result.get('conversation_history', []),
                            'expected_output': scenario_result.get('expected_output'),
                            'responses': {}
                        }
                    scenario_comparison[scenario_id]['responses'][model] = scenario_result['response']
            
            # Add difficulty toggle controls (only if there are scenarios with difficulty indicators)
            if all_difficulties and 'Unknown' not in all_difficulties:
                sorted_difficulties = []
                difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
                for diff in difficulty_order:
                    if diff in all_difficulties:
                        sorted_difficulties.append(diff)
                
                html_content += f"""
                <div class="difficulty-toggles">
                    <h4>Difficulty Level Visibility Controls</h4>
                    <div class="toggle-controls">
                        <button onclick="selectAllDifficulties('{tab_id}')" class="difficulty-button" style="background-color: #28a745;">Show All</button>"""
                
                for difficulty in sorted_difficulties:
                    html_content += f"""
                        <button onclick="toggleDifficulty('{tab_id}', '{difficulty}')" class="difficulty-button" data-difficulty="{difficulty}">{difficulty.upper()}</button>"""
                
                html_content += """
                    </div>
                </div>"""
            
            # Add scenario comparisons
            for scenario_id, data in scenario_comparison.items():
                difficulty = data.get('difficulty', 'Unknown')
                html_content += f"""
                <div class="scenario" data-difficulty="{difficulty}">
                    <div class="scenario-header">{data['name']}</div>
                    <div class="scenario-content">"""
                
                # Description
                if data['personality']:
                    html_content += f"""
                        <div class="section">
                            <div class="section-title">Description</div>
                            <div class="description">{data['personality']}</div>
                        </div>"""
                
                # Conversation History
                if data['conversation_history']:
                    html_content += f"""
                        <div class="section">
                            <div class="section-title">Conversation History</div>
                            <div class="conversation-history">"""
                    for msg in data['conversation_history']:
                        role = msg['role'].upper()
                        content = msg['content']
                        css_class = "user-message" if msg['role'] == "user" else "assistant-message"
                        html_content += f"""
                                <div class="history-message {css_class}">
                                    <strong>{role}:</strong> {content}
                                </div>"""
                    html_content += """
                            </div>
                        </div>"""
                
                # Test Prompt
                html_content += f"""
                        <div class="section">
                            <div class="section-title">Test Prompt</div>
                            <div class="test-prompt">{data['prompt']}</div>
                        </div>"""
                
                # Expected Output (for certain scenario types)
                if data['expected_output']:
                    html_content += f"""
                        <div class="section">
                            <div class="section-title">Expected Output</div>
                            <div class="expected-output">{data['expected_output']}</div>
                        </div>"""
                
                # Responses
                num_models = len(data['responses'])
                grid_class = "two-models" if num_models == 2 else "three-plus-models"
                
                html_content += f"""
                        <div class="section">
                            <div class="section-title">Model Responses</div>
                            <div class="responses-container {grid_class}">"""
                
                for model, response in data['responses'].items():
                    # Add topic detection analysis for topic-detection scenarios
                    topic_analysis_html = ""
                    if main_type == "topic-detection" and data['expected_output']:
                        analysis = analyze_topic_detection_response(response, data['expected_output'])
                        
                        # JSON validity indicator
                        json_icon = "✅" if analysis['is_valid_json'] else "❌"
                        json_status = "Valid JSON" if analysis['is_valid_json'] else "Invalid JSON"
                        
                        # Topic match indicator
                        if analysis['is_valid_json']:
                            topic_icon = "✅" if analysis['topic_matches'] else "❌"
                            topic_status = "Topic Match" if analysis['topic_matches'] else "Topic Mismatch"
                        else:
                            topic_icon = "❓"
                            topic_status = "Cannot Check (Invalid JSON)"
                        
                        topic_analysis_html = f"""
                                    <div class="topic-indicators">
                                        <span class="indicator json-indicator" title="{json_status}">
                                            {json_icon} JSON
                                        </span>
                                        <span class="indicator topic-indicator" title="{topic_status}">
                                            {topic_icon} Topic
                                        </span>
                                    </div>"""
                    
                    html_content += f"""
                                <div class="response" data-model="{model}">
                                    <div class="model-name">{model}</div>{topic_analysis_html}
                                    <div class="model-response">{response}</div>
                                </div>"""
                
                html_content += """
                            </div>
                        </div>
                    </div>
                </div>"""
            
            html_content += """
            </div>"""
        
        html_content += """
        </div>"""
    
    html_content += """
    </div>
</body>
</html>"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Comprehensive HTML report saved to: {report_path}")
    return report_path

def analyze_response_quality(organized_results):
    """Analyze response quality metrics for all scenario types"""
    print("\n=== COMPREHENSIVE RESPONSE QUALITY ANALYSIS ===")
    
    for scenario_type, languages_data in organized_results.items():
        print(f"\n{scenario_type.upper()} ANALYSIS:")
        print("=" * 60)
        
        for lang_key, models_data in languages_data.items():
            print(f"\nLanguage: {lang_key.upper()}")
            print("-" * 40)
            
            for model, info in models_data.items():
                responses = info['data']['results']
                
                total_length = sum(len(r['response']) for r in responses)
                avg_length = total_length / len(responses) if responses else 0
                
                print(f"\n  {model}:")
                print(f"    Total scenarios: {len(responses)}")
                print(f"    Average response length: {avg_length:.1f} characters")
                
                if scenario_type == "topic-detection":
                    # Analyze topic detection accuracy
                    correct_topics = 0
                    valid_json_count = 0
                    confidence_scores = []
                    
                    for response_data in responses:
                        response = response_data['response'].strip()
                        expected = response_data.get('expected_output')
                        
                        # Try to parse JSON response
                        try:
                            response_json = json.loads(response)
                            valid_json_count += 1
                            
                            if 'confidence' in response_json:
                                confidence_scores.append(response_json['confidence'])
                            
                            if expected:
                                expected_json = json.loads(expected)
                                if response_json.get('topic') == expected_json.get('topic'):
                                    correct_topics += 1
                                    
                        except json.JSONDecodeError:
                            pass
                    
                    accuracy = (correct_topics / len(responses)) * 100 if responses else 0
                    json_validity = (valid_json_count / len(responses)) * 100 if responses else 0
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    
                    print(f"    Topic classification accuracy: {accuracy:.1f}%")
                    print(f"    Valid JSON responses: {json_validity:.1f}%")
                    print(f"    Average confidence: {avg_confidence:.1f}")
                    
                elif scenario_type == "roleplay":
                    # Analyze personality consistency (simple heuristic)
                    personality_keywords = 0
                    for response_data in responses:
                        response = response_data['response'].lower()
                        scenario_name = response_data['scenario_name'].lower()
                        
                        # Simple check for character-specific language
                        if any(word in response for word in ['arr', 'aye', 'matey']) and 'pirate' in scenario_name:
                            personality_keywords += 1
                        elif any(word in response for word in ['sir', 'madam', 'protocols']) and 'robot' in scenario_name:
                            personality_keywords += 1
                        elif any(word in response for word in ['mon ami', 'ze', 'très']) and 'chef' in scenario_name:
                            personality_keywords += 1
                    
                    consistency_score = (personality_keywords / len(responses)) * 100 if responses else 0
                    print(f"    Personality consistency: {consistency_score:.1f}%")
                    
                elif scenario_type == "translation":
                    # Analyze translation quality (basic checks)
                    non_empty_translations = sum(1 for r in responses if r['response'].strip())
                    completeness = (non_empty_translations / len(responses)) * 100 if responses else 0
                    print(f"    Translation completeness: {completeness:.1f}%")

def main():
    """Main function to generate comprehensive comparison report"""
    print("=== LLM COMPREHENSIVE COMPARISON TOOL ===")
    print("Generating report for all available scenario types with latest results...")
    
    # Get results for each model/scenario combination
    results = get_latest_results_by_model_and_scenario()
    
    if not results:
        print("No test result files found.")
        return
    
    # Organize results by scenario type
    organized_results = organize_results_by_scenario_type(results)
    
    if not organized_results:
        print("No valid test results found.")
        return
    
    print(f"Found results for scenario types: {list(organized_results.keys())}")
    
    # Generate comprehensive HTML report
    report_path = create_comprehensive_html_report(organized_results)
    
    # Analyze response quality
    analyze_response_quality(organized_results)
    
    print(f"\n=== REPORT COMPLETE ===")
    print(f"HTML report: {report_path}")

if __name__ == "__main__":
    main()
