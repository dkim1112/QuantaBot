#!/usr/bin/env python3
"""
Real MedQA Evaluation for QuantaBot
==================================

This script runs the actual evaluation using the real MedQA dataset
and official medical textbook content.
"""

import os
import sys
import json
from pathlib import Path

# Project root is one level up from this script; src/ holds the quantabot package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if not os.getenv('OPENAI_API_KEY'):
    print("❌ Error: OpenAI API key not found!")
    print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

from evaluations.medqa.evaluator import MedQAEvaluator

def setup_data_paths():
    """Set up paths to the real MedQA data and textbooks."""

    # MedQA questions (using test set for evaluation)
    medqa_file = "data/questions/US/test.jsonl"

    # Medical textbook paths
    textbook_dir = Path("data/textbooks/en/")
    textbook_files = []

    # Key medical textbooks for USMLE preparation
    important_textbooks = [
        "InternalMed_Harrison.txt",  # Harrison's Internal Medicine
        "Pathology_Robbins.txt",     # Robbins Pathology
        "First_Aid_Step1.txt",       # First Aid USMLE Step 1
        "First_Aid_Step2.txt",       # First Aid USMLE Step 2
        "Pediatrics_Nelson.txt",     # Nelson Pediatrics
        "Surgery_Schwartz.txt",      # Schwartz Surgery
        "Pharmacology_Katzung.txt"   # Katzung Pharmacology
    ]

    for textbook in important_textbooks:
        textbook_path = textbook_dir / textbook
        if textbook_path.exists():
            textbook_files.append(str(textbook_path))
            print(f"✅ Found textbook: {textbook}")
        else:
            print(f"⚠️ Missing textbook: {textbook}")

    if not Path(medqa_file).exists():
        print(f"❌ Error: MedQA test file not found: {medqa_file}")
        return None, []

    if not textbook_files:
        print("❌ Error: No medical textbooks found!")
        return None, []

    print(f"📚 Using {len(textbook_files)} medical textbooks")
    print(f"📄 Using MedQA test questions: {medqa_file}")

    return medqa_file, textbook_files

def run_evaluation():
    """Run the MedQA evaluation with real data."""

    print("🏥 Starting Real MedQA Evaluation for QuantaBot")
    print("=" * 60)

    # Set up data paths
    medqa_file, textbook_files = setup_data_paths()
    if not medqa_file or not textbook_files:
        return

    # Create output directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Initialize evaluator
    print("\n🤖 Initializing QuantaBot evaluator...")
    try:
        evaluator = MedQAEvaluator(
            medqa_file=medqa_file,
            textbook_paths=textbook_files,
            output_dir=str(results_dir)
        )
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return

    # Run evaluation on a subset first (100 questions for manageable time)
    max_questions = 20
    print(f"\n📊 Running evaluation on {max_questions} questions...")
    print("⏱️ This may take several minutes...")

    try:
        results = evaluator.run_evaluation(max_questions=max_questions)

        if 'error' in results:
            print(f"\n❌ Evaluation failed: {results['error']}")
            return

        # Display results
        print("\n" + "=" * 60)
        print("🎯 REAL MEDQA EVALUATION RESULTS")
        print("=" * 60)

        dataset_info = results['dataset_info']
        accuracy = results['accuracy_metrics']
        quality = results['quality_metrics']
        performance = results['performance_metrics']

        print(f"\n📊 Dataset Information:")
        print(f"  Total Questions Attempted: {dataset_info['total_questions_attempted']}")
        print(f"  Successfully Completed: {dataset_info['successfully_completed']}")
        print(f"  Failed Questions: {dataset_info['failed_questions']}")

        print(f"\n🎯 Accuracy Metrics:")
        print(f"  Letter Accuracy (strict MC): {accuracy['letter_accuracy']}% ({accuracy['letter_correct']}/{dataset_info['successfully_completed']})")
        print(f"  Letter Extraction Rate: {accuracy['letter_extraction_rate']}%")
        print(f"  Retrieval Contains Answer: {accuracy['retrieval_contains_answer_rate']}% (retrieval-vs-LLM diagnostic)")
        print(f"  Exact Match Accuracy (loose): {accuracy['exact_match_accuracy']}%")
        print(f"  Partial Match Accuracy (loose): {accuracy['partial_match_accuracy']}%")
        print(f"  Exact Matches: {accuracy['exact_matches']}")
        print(f"  Partial Matches: {accuracy['partial_matches']}")

        print(f"\n🏥 Clinical Quality Assessment:")
        print(f"  Clinical Relevance Score: {quality['average_clinical_relevance_score']}/5")
        print(f"  Retrieval Quality Score: {quality['average_retrieval_quality_score']}/5")

        print(f"\n⚡ Performance Metrics:")
        print(f"  Average Response Time: {performance['average_response_time_seconds']}s")
        print(f"  Questions per Minute: {performance['questions_per_minute']}")

        # Show sample detailed results
        print(f"\n📝 Sample Question Results:")
        print("-" * 60)

        sample_results = results['detailed_results'][:3]  # Show first 3
        for i, result in enumerate(sample_results, 1):
            if result.get('status') == 'completed':
                print(f"\n🔍 Question {i}:")
                print(f"📋 Question: {result['question'][:100]}...")
                print(f"✅ Correct Answer: {result['correct_answer']}")
                print(f"🤖 QuantaBot Response Preview: {result['quantabot_answer'][:150]}...")
                print(f"✓ Exact Match: {'Yes' if result['exact_match'] else 'No'}")
                print(f"✓ Clinical Relevance: {result['clinical_relevance_score']}/5")
                print(f"⏱️ Response Time: {result['response_time']:.1f}s")
                print(f"📚 Sources: {result['source_document_count']} citations")

        # Save detailed results
        results_file = results_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Detailed results saved to: {results_file}")
        print("\n" + "=" * 60)
        print("✅ Real MedQA Evaluation Complete!")
        print("=" * 60)

        return results

    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)