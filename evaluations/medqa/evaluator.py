#!/usr/bin/env python3
"""
MedQA Evaluation Script for QuantaBot
====================================

This script evaluates QuantaBot's clinical relevance and performance on MedQA (USMLE-style) questions
using medical textbook content as reference material.

Dataset Information:
- MedQA: Medical multiple-choice questions from USMLE exams
- Textbooks: 18 widely-used medical textbooks (125,847 chunks, avg 182 tokens)
- Format: Questions with 4-5 multiple choice options

Usage:
    python medqa_evaluation.py --medqa_file path/to/medqa.jsonl --textbook_dir path/to/textbooks/

Requirements:
    - MedQA JSONL file with questions and answers
    - Medical textbook content (can use MedRAG/textbooks from HuggingFace)
    - QuantaBot system with LangChain components
"""

import json
import os
import re
import sys
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from quantabot.core.rag import LangChainQuantaBot
from quantabot.utils.embedding_wrapper import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medqa_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedQAEvaluator:
    """
    Evaluates QuantaBot performance on MedQA dataset for clinical relevance assessment.
    """

    def __init__(self, medqa_file: str, textbook_paths: List[str],
                 output_dir: str = "medqa_results"):
        """
        Initialize the MedQA evaluator.

        Args:
            medqa_file: Path to MedQA JSONL file with questions and answers
            textbook_paths: List of paths to medical textbook files
            output_dir: Directory to save evaluation results
        """
        self.medqa_file = medqa_file
        self.textbook_paths = textbook_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize QuantaBot
        self.embedding_model = HuggingFaceEmbeddings()
        self.quantabot = LangChainQuantaBot(
            collection_name=f"medqa_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            embedding_function=self.embedding_model
        )

        # Evaluation results storage
        self.results = []
        self.evaluation_metrics = {
            'total_questions': 0,
            'correct_answers': 0,
            'partial_matches': 0,
            'letter_correct': 0,
            'letter_extracted': 0,
            'retrieval_contains_answer': 0,
            'clinical_relevance_scores': [],
            'response_times': [],
            'retrieval_quality_scores': []
        }

    def load_medqa_questions(self) -> List[Dict[str, Any]]:
        """
        Load MedQA questions from JSONL file.

        Returns:
            List of question dictionaries
        """
        questions = []
        logger.info(f"Loading MedQA questions from {self.medqa_file}")

        with open(self.medqa_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    question_data = json.loads(line.strip())
                    questions.append(question_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    def setup_quantabot_with_textbooks(self):
        """
        Set up QuantaBot with medical textbook content as reference material.
        """
        logger.info("Setting up QuantaBot with medical textbooks...")

        # Verify textbook files exist
        valid_textbooks = []
        for path in self.textbook_paths:
            if os.path.exists(path):
                valid_textbooks.append(path)
            else:
                logger.warning(f"Textbook file not found: {path}")

        if not valid_textbooks:
            raise ValueError("No valid textbook files found")

        logger.info(f"Processing {len(valid_textbooks)} textbook files...")

        # Create file mapping for citation purposes
        file_mapping = {}
        for path in valid_textbooks:
            file_mapping[path] = os.path.basename(path)

        # Process documents with QuantaBot
        try:
            documents = self.quantabot.preprocess_documents(
                valid_textbooks,
                file_mapping=file_mapping,
                batch_size=5  # Process in smaller batches for memory efficiency
            )
            logger.info(f"Successfully processed {len(documents)} document chunks")

        except Exception as e:
            logger.error(f"Error processing textbooks: {e}")
            raise

    def evaluate_single_question(self, question_data: Dict[str, Any],
                                question_id: int) -> Dict[str, Any]:
        """
        Evaluate QuantaBot's performance on a single MedQA question.

        Args:
            question_data: Dictionary containing question, options, and answer
            question_id: Unique identifier for the question

        Returns:
            Dictionary with evaluation results
        """
        question_text = question_data['question']
        correct_answer = question_data['answer']
        options = question_data.get('options', [])

        logger.info(f"Evaluating question {question_id}")

        # Format question for QuantaBot
        formatted_question = self._format_question_for_quantabot(
            question_text, options
        )

        # Query QuantaBot
        start_time = time.time()
        try:
            result = self.quantabot.query(formatted_question)
            response_time = time.time() - start_time

            quantabot_answer = result['answer']
            source_documents = result.get('source_documents', [])

        except Exception as e:
            logger.error(f"Error querying QuantaBot for question {question_id}: {e}")
            return {
                'question_id': question_id,
                'error': str(e),
                'status': 'failed'
            }

        # Evaluate the response
        evaluation_result = self._evaluate_response(
            question_data, quantabot_answer, source_documents, response_time
        )
        evaluation_result['question_id'] = question_id

        return evaluation_result

    def _format_question_for_quantabot(self, question: str, options) -> str:
        """
        Format MedQA question for QuantaBot input.

        Args:
            question: The medical question text
            options: Multiple choice options as a dict {"A": "...", ...} or list

        Returns:
            Formatted question string
        """
        formatted = f"Medical Question: {question}\n\n"

        if options:
            formatted += "Multiple Choice Options:\n"
            if isinstance(options, dict):
                for letter, text in options.items():
                    formatted += f"{letter}. {text}\n"
            else:
                for i, text in enumerate(options):
                    formatted += f"{chr(65 + i)}. {text}\n"
            formatted += (
                "\nRespond in the following format:\n"
                "Answer: <single letter A-E>\n"
                "Reasoning: <your medical reasoning with citations>"
            )

        return formatted

    @staticmethod
    def _extract_answer_letter(response: str) -> Optional[str]:
        """
        Extract the selected option letter (A-E) from QuantaBot's response.

        Tries several patterns in order of specificity. Returns None if no
        letter can be confidently extracted.
        """
        if not response:
            return None

        patterns = [
            r"\bAnswer\s*[:\-]\s*\(?([A-E])\)?\b",
            r"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s*\(?([A-E])\)?\b",
            r"\b(?:option|choice)\s*\(?([A-E])\)?\b",
            r"^\s*\(?([A-E])\)[\.\):\s]",
            r"^\s*([A-E])\s*[\.\):\-]",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        return None

    def _evaluate_response(self, question_data: Dict[str, Any],
                          quantabot_answer: str, source_documents: List,
                          response_time: float) -> Dict[str, Any]:
        """
        Evaluate QuantaBot's response against the correct answer.

        Args:
            question_data: Original question data
            quantabot_answer: QuantaBot's response
            source_documents: Retrieved source documents
            response_time: Time taken to generate response

        Returns:
            Evaluation metrics dictionary
        """
        correct_answer = question_data['answer']
        options = question_data.get('options', [])
        correct_letter = question_data.get('answer_idx')

        # Check for exact answer match
        exact_match = self._check_exact_match(quantabot_answer, correct_answer)

        # Check for partial match (answer mentioned in response)
        partial_match = self._check_partial_match(quantabot_answer, correct_answer)

        # Strict MC scoring: extract picked letter and compare to answer_idx
        predicted_letter = self._extract_answer_letter(quantabot_answer)
        letter_match = (
            predicted_letter is not None
            and correct_letter is not None
            and predicted_letter == correct_letter.upper()
        )

        # Retrieval proxy: did the correct answer text reach the LLM at all?
        retrieval_contains_answer = self._retrieval_contains_answer(
            source_documents, correct_answer
        )

        # Evaluate clinical relevance (1-5 scale)
        clinical_relevance = self._evaluate_clinical_relevance(
            question_data, quantabot_answer
        )

        # Evaluate retrieval quality
        retrieval_quality = self._evaluate_retrieval_quality(
            question_data, source_documents
        )

        # Extract citations
        citations = self._extract_citations(source_documents)

        return {
            'question': question_data['question'],
            'correct_answer': correct_answer,
            'correct_letter': correct_letter,
            'predicted_letter': predicted_letter,
            'letter_match': letter_match,
            'retrieval_contains_answer': retrieval_contains_answer,
            'quantabot_answer': quantabot_answer,
            'exact_match': exact_match,
            'partial_match': partial_match,
            'clinical_relevance_score': clinical_relevance,
            'retrieval_quality_score': retrieval_quality,
            'response_time': response_time,
            'citations': citations,
            'source_document_count': len(source_documents),
            'options': options,
            'status': 'completed'
        }

    def _check_exact_match(self, response: str, correct_answer: str) -> bool:
        """Check if the response exactly matches the correct answer."""
        # Normalize both strings for comparison
        response_lower = response.lower().strip()
        correct_lower = correct_answer.lower().strip()

        # Check direct mention
        return correct_lower in response_lower

    @staticmethod
    def _retrieval_contains_answer(source_documents, correct_answer: str) -> bool:
        """Check whether the correct answer text appears verbatim in any retrieved chunk.

        Loose proxy for retrieval recall: if False, the LLM never had the info.
        If True, the failure (when one exists) is on the LLM, not the retriever.
        """
        if not correct_answer or not source_documents:
            return False
        needle = correct_answer.lower().strip()
        if not needle:
            return False
        for doc in source_documents:
            content = getattr(doc, "page_content", "") or ""
            if needle in content.lower():
                return True
        return False

    def _check_partial_match(self, response: str, correct_answer: str) -> bool:
        """Check if key terms from the correct answer appear in the response."""
        # Split correct answer into words and check for key terms
        correct_words = set(correct_answer.lower().split())
        response_lower = response.lower()

        # Check if at least 50% of words from correct answer appear in response
        matching_words = sum(1 for word in correct_words if word in response_lower)
        return matching_words >= len(correct_words) * 0.5

    def _evaluate_clinical_relevance(self, question_data: Dict[str, Any],
                                   response: str) -> int:
        """
        Evaluate clinical relevance of the response (1-5 scale).

        This is a simplified heuristic evaluation. In practice, this would
        require medical expert review.
        """
        score = 3  # Base score

        # Check for medical terminology usage
        medical_terms = [
            'diagnosis', 'treatment', 'symptoms', 'patient', 'clinical',
            'therapy', 'pathology', 'syndrome', 'disease', 'medication',
            'differential', 'prognosis', 'etiology'
        ]

        term_count = sum(1 for term in medical_terms if term in response.lower())

        # Adjust score based on medical terminology density
        if term_count >= 5:
            score += 1
        elif term_count <= 1:
            score -= 1

        # Check for evidence-based reasoning
        evidence_terms = ['evidence', 'studies', 'research', 'guidelines']
        if any(term in response.lower() for term in evidence_terms):
            score += 1

        return max(1, min(5, score))

    def _evaluate_retrieval_quality(self, question_data: Dict[str, Any],
                                  source_documents: List) -> int:
        """
        Evaluate quality of retrieved source documents (1-5 scale).
        """
        if not source_documents:
            return 1

        # Base score
        score = 3

        # Check document count (optimal range: 3-8 documents)
        doc_count = len(source_documents)
        if 3 <= doc_count <= 8:
            score += 1
        elif doc_count > 10:
            score -= 1

        # Check for relevant medical content in retrieved documents
        question_lower = question_data['question'].lower()
        relevant_docs = 0

        for doc in source_documents:
            doc_content = doc.page_content.lower()
            # Simple relevance check based on shared medical terms
            if any(word in doc_content for word in question_lower.split()
                   if len(word) > 4):
                relevant_docs += 1

        relevance_ratio = relevant_docs / doc_count if doc_count > 0 else 0
        if relevance_ratio >= 0.7:
            score += 1
        elif relevance_ratio < 0.3:
            score -= 1

        return max(1, min(5, score))

    def _extract_citations(self, source_documents: List) -> List[Dict[str, str]]:
        """Extract citation information from source documents."""
        citations = []

        for doc in source_documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                citation = {
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'content_preview': doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100 else doc.page_content
                }
                citations.append(citation)

        return citations

    def run_evaluation(self, max_questions: int = None) -> Dict[str, Any]:
        """
        Run the complete MedQA evaluation.

        Args:
            max_questions: Maximum number of questions to evaluate (None for all)

        Returns:
            Dictionary with overall evaluation results
        """
        logger.info("Starting MedQA evaluation...")

        # Setup phase
        try:
            self.setup_quantabot_with_textbooks()
        except Exception as e:
            logger.error(f"Failed to setup QuantaBot: {e}")
            return {'error': f"Setup failed: {e}"}

        # Load questions
        questions = self.load_medqa_questions()

        if max_questions:
            questions = questions[:max_questions]
            logger.info(f"Limiting evaluation to {max_questions} questions")

        self.evaluation_metrics['total_questions'] = len(questions)

        # Evaluate each question
        for i, question_data in enumerate(questions):
            try:
                result = self.evaluate_single_question(question_data, i + 1)
                self.results.append(result)

                # Update metrics
                if result.get('status') == 'completed':
                    if result.get('exact_match', False):
                        self.evaluation_metrics['correct_answers'] += 1
                    if result.get('partial_match', False):
                        self.evaluation_metrics['partial_matches'] += 1
                    if result.get('predicted_letter') is not None:
                        self.evaluation_metrics['letter_extracted'] += 1
                    if result.get('letter_match', False):
                        self.evaluation_metrics['letter_correct'] += 1
                    if result.get('retrieval_contains_answer', False):
                        self.evaluation_metrics['retrieval_contains_answer'] += 1

                    self.evaluation_metrics['clinical_relevance_scores'].append(
                        result.get('clinical_relevance_score', 0)
                    )
                    self.evaluation_metrics['response_times'].append(
                        result.get('response_time', 0)
                    )
                    self.evaluation_metrics['retrieval_quality_scores'].append(
                        result.get('retrieval_quality_score', 0)
                    )

                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(questions)} questions")

            except Exception as e:
                logger.error(f"Error evaluating question {i + 1}: {e}")
                self.results.append({
                    'question_id': i + 1,
                    'error': str(e),
                    'status': 'failed'
                })

        # Generate summary
        summary = self._generate_evaluation_summary()

        # Save results
        self._save_results(summary)

        logger.info("MedQA evaluation completed!")
        return summary

    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary with key metrics."""
        metrics = self.evaluation_metrics
        total = metrics['total_questions']
        completed = len([r for r in self.results if r.get('status') == 'completed'])

        if completed == 0:
            return {'error': 'No questions were successfully evaluated'}

        # Calculate accuracy metrics
        exact_accuracy = (metrics['correct_answers'] / completed) * 100
        partial_accuracy = (metrics['partial_matches'] / completed) * 100
        letter_accuracy = (metrics['letter_correct'] / completed) * 100
        letter_extraction_rate = (metrics['letter_extracted'] / completed) * 100
        retrieval_contains_answer_rate = (metrics['retrieval_contains_answer'] / completed) * 100

        # Calculate average scores
        avg_clinical_relevance = sum(metrics['clinical_relevance_scores']) / len(metrics['clinical_relevance_scores']) if metrics['clinical_relevance_scores'] else 0
        avg_retrieval_quality = sum(metrics['retrieval_quality_scores']) / len(metrics['retrieval_quality_scores']) if metrics['retrieval_quality_scores'] else 0
        avg_response_time = sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0

        return {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_questions_attempted': total,
                'successfully_completed': completed,
                'failed_questions': total - completed
            },
            'accuracy_metrics': {
                'letter_accuracy': round(letter_accuracy, 2),
                'letter_extraction_rate': round(letter_extraction_rate, 2),
                'letter_correct': metrics['letter_correct'],
                'letter_extracted': metrics['letter_extracted'],
                'retrieval_contains_answer_rate': round(retrieval_contains_answer_rate, 2),
                'retrieval_contains_answer': metrics['retrieval_contains_answer'],
                'exact_match_accuracy': round(exact_accuracy, 2),
                'partial_match_accuracy': round(partial_accuracy, 2),
                'exact_matches': metrics['correct_answers'],
                'partial_matches': metrics['partial_matches']
            },
            'quality_metrics': {
                'average_clinical_relevance_score': round(avg_clinical_relevance, 2),
                'average_retrieval_quality_score': round(avg_retrieval_quality, 2),
                'clinical_relevance_distribution': self._get_score_distribution(metrics['clinical_relevance_scores']),
                'retrieval_quality_distribution': self._get_score_distribution(metrics['retrieval_quality_scores'])
            },
            'performance_metrics': {
                'average_response_time_seconds': round(avg_response_time, 2),
                'total_evaluation_time': sum(metrics['response_times']),
                'questions_per_minute': round(completed / (sum(metrics['response_times']) / 60), 2) if sum(metrics['response_times']) > 0 else 0
            },
            'detailed_results': self.results
        }

    def _get_score_distribution(self, scores: List[int]) -> Dict[str, int]:
        """Get distribution of scores (1-5 scale)."""
        distribution = {str(i): 0 for i in range(1, 6)}
        for score in scores:
            if 1 <= score <= 5:
                distribution[str(score)] += 1
        return distribution

    def _save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results as JSON
        results_file = self.output_dir / f"medqa_evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save summary as CSV for easy analysis
        if self.results:
            csv_file = self.output_dir / f"medqa_evaluation_summary_{timestamp}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)

        logger.info(f"Results saved to {results_file} and {csv_file}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Evaluate QuantaBot on MedQA dataset')
    parser.add_argument('--medqa_file', required=True,
                       help='Path to MedQA JSONL file')
    parser.add_argument('--textbook_dir', required=True,
                       help='Directory containing medical textbook files')
    parser.add_argument('--max_questions', type=int, default=None,
                       help='Maximum number of questions to evaluate')
    parser.add_argument('--output_dir', default='medqa_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Find textbook files
    textbook_dir = Path(args.textbook_dir)
    if not textbook_dir.exists():
        print(f"Error: Textbook directory not found: {textbook_dir}")
        sys.exit(1)

    # Support common medical document formats
    textbook_files = []
    for ext in ['*.txt', '*.pdf', '*.docx']:
        textbook_files.extend(textbook_dir.glob(ext))

    if not textbook_files:
        print(f"Error: No textbook files found in {textbook_dir}")
        sys.exit(1)

    print(f"Found {len(textbook_files)} textbook files")

    # Initialize evaluator
    evaluator = MedQAEvaluator(
        medqa_file=args.medqa_file,
        textbook_paths=[str(f) for f in textbook_files],
        output_dir=args.output_dir
    )

    # Run evaluation
    try:
        results = evaluator.run_evaluation(max_questions=args.max_questions)

        if 'error' in results:
            print(f"Evaluation failed: {results['error']}")
            sys.exit(1)

        # Print summary
        print("\n" + "="*60)
        print("MEDQA EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {results['dataset_info']['total_questions_attempted']}")
        print(f"Completed Successfully: {results['dataset_info']['successfully_completed']}")
        print(f"Exact Match Accuracy: {results['accuracy_metrics']['exact_match_accuracy']}%")
        print(f"Partial Match Accuracy: {results['accuracy_metrics']['partial_match_accuracy']}%")
        print(f"Average Clinical Relevance: {results['quality_metrics']['average_clinical_relevance_score']}/5")
        print(f"Average Retrieval Quality: {results['quality_metrics']['average_retrieval_quality_score']}/5")
        print(f"Average Response Time: {results['performance_metrics']['average_response_time_seconds']}s")
        print("="*60)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()