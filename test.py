"""
Comprehensive Testing Suite for AI Graduate Advisor RAG System
===============================================================
This script runs extensive experiments to evaluate the retrieval-augmented
generation system across multiple dimensions: 

1. Retrieval Quality Tests
2. Fuzzy Name Matching Tests
3. Query Expansion Impact Analysis
4. Special Query Handling Tests
5. Response Latency Measurements
6. Baseline Comparisons

Results are saved to test_results/ directory as JSON files.
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Import your response engine (refactored module version)
from engine import ResponseEngine

# Create results directory
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class ComprehensiveTestSuite:
    """Main testing class that runs all experiments"""
    
    def __init__(self):
        """Initialize the test suite with ResponseEngine"""
        print("=" * 70)
        print("AI GRADUATE ADVISOR - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("\n[INIT] Initializing Response Engine...")
        self.engine = ResponseEngine()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "retrieval_tests": [],
            "fuzzy_match_tests": [],
            "query_expansion_tests": [],
            "special_query_tests": [],
            "latency_tests": [],
            "baseline_comparison":  {}
        }
        print("[INIT] ✓ Response Engine initialized successfully\n")
    
    def test_retrieval_quality(self):
        """Test 1: Retrieval Quality Across Research Domains"""
        print("\n" + "=" * 70)
        print("TEST 1: RETRIEVAL QUALITY EVALUATION")
        print("=" * 70)
        
        # Define test queries with expected faculty (ground truth)
        test_queries = [
            {
                "query": "Who does artificial intelligence and machine learning research? ",
                "domain": "AI/ML",
                "expected_keywords": ["machine learning", "artificial intelligence", "neural", "deep learning"],
                "expected_faculty": ["Tim Andersen", "Edoardo Serra", "Francesca Spezzano", "Jun Zhuang", "Xinyi Zhou"]
            },
            {
                "query": "Which faculty work on cybersecurity and network security?",
                "domain":  "Security",
                "expected_keywords":  ["security", "cyber", "privacy", "crypto"],
                "expected_faculty": ["Jyh-haw Yeh", "Gaby Dagher", "Hoda Mehrpouyan", "Shane Panter"]
            },
            {
                "query": "Who researches human-computer interaction and user experience?",
                "domain":  "HCI",
                "expected_keywords": ["human-computer interaction", "hci", "user experience", "interface"],
                "expected_faculty":  ["Jerry Alan Fails"]
            },
            {
                "query":  "Tell me about professors working on distributed systems and parallel computing",
                "domain": "Systems",
                "expected_keywords":  ["distributed", "parallel", "systems", "computing"],
                "expected_faculty":  ["Amit Jain", "Max Taylor", "Steven Cutchin"]
            },
            {
                "query":  "Who does computer vision and image processing research?",
                "domain": "Vision",
                "expected_keywords": ["vision", "image", "visual"],
                "expected_faculty": ["Yu Zhang"]
            },
            {
                "query":  "Which professors work on natural language processing? ",
                "domain": "NLP",
                "expected_keywords": ["natural language", "nlp", "text", "language"],
                "expected_faculty": ["Casey Kennington", "Eric Henderson"]
            },
            {
                "query":  "Who researches software engineering and program analysis?",
                "domain": "Software Engineering",
                "expected_keywords": ["software", "engineering", "program", "testing"],
                "expected_faculty":  ["Bogdan Dit", "Elena Sherman", "Jim Buffenbarger", "Sarah Frost"]
            }
        ]
        
        for test_case in test_queries:
            print(f"\n[TEST] Query: {test_case['query']}")
            print(f"[TEST] Domain: {test_case['domain']}")
            
            start_time = time.time()
            results = self.engine.retrieve_faculty(test_case['query'], top_k=5)
            latency = time.time() - start_time
            
            print(f"[TEST] Retrieved {len(results)} faculty in {latency:.3f}s")
            
            # Calculate metrics
            retrieved_names = [r['name'] for r in results]
            scores = [r['score'] for r in results]
            
            # Precision at k
            precision_at_1 = 1 if results and results[0]['name'] in test_case['expected_faculty'] else 0
            precision_at_3 = sum(1 for r in results[: 3] if r['name'] in test_case['expected_faculty']) / min(3, len(results)) if results else 0
            precision_at_5 = sum(1 for r in results[:5] if r['name'] in test_case['expected_faculty']) / min(5, len(results)) if results else 0
            
            avg_score = np.mean(scores) if scores else 0
            
            test_result = {
                "query": test_case['query'],
                "domain": test_case['domain'],
                "retrieved_faculty": [{"name": r['name'], "score": r['score']} for r in results],
                "expected_faculty": test_case['expected_faculty'],
                "metrics": {
                    "precision_at_1": precision_at_1,
                    "precision_at_3": precision_at_3,
                    "precision_at_5": precision_at_5,
                    "avg_similarity_score": float(avg_score),
                    "latency_seconds": latency
                }
            }
            
            self.results["retrieval_tests"].append(test_result)
            
            print(f"[RESULT] P@1: {precision_at_1:.2f} | P@3: {precision_at_3:.2f} | P@5: {precision_at_5:.2f}")
            print(f"[RESULT] Avg Score: {avg_score:.3f}")
            print(f"[RESULT] Top 3 Faculty: {retrieved_names[:3]}")
            
            # Rate limit protection
            time.sleep(2)
        
        # Calculate aggregate metrics
        all_tests = self.results["retrieval_tests"]
        avg_p1 = np.mean([t['metrics']['precision_at_1'] for t in all_tests])
        avg_p3 = np.mean([t['metrics']['precision_at_3'] for t in all_tests])
        avg_p5 = np.mean([t['metrics']['precision_at_5'] for t in all_tests])
        avg_sim = np.mean([t['metrics']['avg_similarity_score'] for t in all_tests])
        
        print("\n" + "-" * 70)
        print("AGGREGATE RETRIEVAL METRICS:")
        print(f"  Average Precision@1: {avg_p1:.3f}")
        print(f"  Average Precision@3: {avg_p3:.3f}")
        print(f"  Average Precision@5: {avg_p5:.3f}")
        print(f"  Average Similarity:  {avg_sim:.3f}")
        print("-" * 70)
    
    def test_fuzzy_name_matching(self):
        print("\n" + "=" * 70)
        print("TEST 2: FUZZY NAME MATCHING (NO API)")
        print("=" * 70)

        test_cases = [
            ("Jun Zuang", "Jun Zhuang"),
            ("Fransesca Spezano", "Francesca Spezzano"),
            ("Edwardo Sera", "Edoardo Serra"),
            ("Gabi Dagger", "Gaby Dagher"),
            ("Jerry Fails", "Jerry Alan Fails"),
            ("Tim Anderson", "Tim Andersen"),
            ("Jyh Yeh", "Jyh-haw Yeh"),
            ("Casey Kenington", "Casey Kennington"),
            ("Xinyi Zou", "Xinyi Zhou"),
            ("Hoda Merpouyan", "Hoda Mehrpouyan"),
        ]

        correct = 0

        for misspelled, expected in test_cases:
            # Directly test retrieval (this triggers fuzzy logic)
            results = self.engine.retrieve_faculty(misspelled, top_k=1)

            matched = bool(results) and results[0]["name"] == expected

            self.results["fuzzy_match_tests"].append({
                "input": misspelled,
                "expected": expected,
                "matched": matched
            })

            print(f"[TEST] {misspelled} → {results[0]['name'] if results else 'NONE'}")

            if matched:
                correct += 1

        accuracy = correct / len(test_cases)
        self.results["fuzzy_match_accuracy"] = accuracy

        print("\nFUZZY MATCHING METRICS:")
        print(f"  Correct Matches: {correct}/{len(test_cases)}")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        
    def test_query_expansion_impact(self):
        """Test 3: Impact of Query Expansion with Synonyms"""
        print("\n" + "=" * 70)
        print("TEST 3: QUERY EXPANSION IMPACT ANALYSIS")
        print("=" * 70)
        
        # Test queries that benefit from expansion
        test_queries = [
            "AI research",
            "ML faculty",
            "security professors",
            "HCI work",
            "NLP experts",
            "CV research"
        ]
        
        for query in test_queries: 
            print(f"\n[TEST] Query: '{query}'")
            
            # Get results WITHOUT expansion (direct embedding)
            q_emb = self.engine. embed_model.encode([query])[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            sims_without = self.engine.embeddings @ q_emb
            avg_score_without = float(np.mean(np.sort(sims_without)[::-1][:5]))
            
            # Get results WITH expansion (using query processor)
            expanded = self.engine.query_processor. expand_query(query)
            q_emb_exp = self.engine.embed_model.encode([expanded])[0]
            q_emb_exp = q_emb_exp / (np.linalg.norm(q_emb_exp) + 1e-12)
            sims_with = self. engine.embeddings @ q_emb_exp
            avg_score_with = float(np.mean(np.sort(sims_with)[::-1][:5]))
            
            improvement = ((avg_score_with - avg_score_without) / avg_score_without) * 100 if avg_score_without > 0 else 0
            
            result = {
                "query": query,
                "expanded_query": expanded,
                "avg_score_without_expansion": avg_score_without,
                "avg_score_with_expansion": avg_score_with,
                "improvement_percent": improvement
            }
            
            self. results["query_expansion_tests"]. append(result)
            
            print(f"[RESULT] Expanded to: '{expanded}'")
            print(f"[RESULT] Score without expansion: {avg_score_without:.3f}")
            print(f"[RESULT] Score with expansion: {avg_score_with:.3f}")
            print(f"[RESULT] Improvement: {improvement:.1f}%")
            
            time.sleep(1)
        
        avg_improvement = np.mean([t['improvement_percent'] for t in self.results["query_expansion_tests"]])
        
        print("\n" + "-" * 70)
        print("QUERY EXPANSION METRICS:")
        print(f"  Average Improvement: {avg_improvement:.1f}%")
        print("-" * 70)
    
    def test_special_queries(self):
        """Test 4: Special Query Handling (List, Fallback, etc.)"""
        print("\n" + "=" * 70)
        print("TEST 4: SPECIAL QUERY HANDLING")
        print("=" * 70)
        
        test_cases = [
            {
                "type": "list_query",
                "query": "list all faculty",
                "expected_behavior": "Returns complete faculty list"
            },
            {
                "type": "list_with_research",
                "query": "list all faculty with their research areas",
                "expected_behavior":  "Returns faculty with research areas"
            },
            {
                "type": "negative_response",
                "query": "no",
                "expected_behavior": "Graceful acknowledgment"
            }
        ]
        
        for test_case in test_cases: 
            print(f"\n[TEST] Type: {test_case['type']}")
            print(f"[TEST] Query: '{test_case['query']}'")
            
            start_time = time.time()
            response = self.engine.generate_rag_answer(test_case['query'])
            latency = time.time() - start_time
            
            # Validation based on type
            if test_case['type'] == 'list_query':
                valid = "Tim Andersen" in response and "Xinyi Zhou" in response
            elif test_case['type'] == 'list_with_research': 
                valid = "Trustworthy Generative AI" in response or "Artificial Neural Network" in response
            elif test_case['type'] == 'negative_response':
                valid = "No problem" in response
            else: 
                valid = len(response) > 20
            
            result = {
                "type": test_case['type'],
                "query": test_case['query'],
                "expected_behavior": test_case['expected_behavior'],
                "response":  response[: 300] + "..." if len(response) > 300 else response,
                "latency_seconds":  latency,
                "validation_passed": valid
            }
            
            self.results["special_query_tests"].append(result)
            
            status = "✓" if valid else "✗"
            print(f"[RESULT] {status} Validation: {'PASS' if valid else 'FAIL'}")
            print(f"[RESULT] Response length: {len(response)} chars")
            print(f"[RESULT] Latency: {latency:.3f}s")
            
            time.sleep(2)
    
    def test_response_latency(self):
        """Test 5: Response Latency Measurements"""
        print("\n" + "=" * 70)
        print("TEST 5: RESPONSE LATENCY MEASUREMENT")
        print("=" * 70)
        print("Note:  Latency tests do NOT make API calls to avoid rate limits")
        print("Only measuring retrieval speed (no LLM generation)")
        print("=" * 70)
        
        test_queries = [
            "Who does AI research? ",
            "Tell me about Jun Zhuang",
            "Which professors work on security?",
            "list all faculty",
            "Who researches machine learning and deep learning?"
        ]
        
        latencies = []
        
        for query in test_queries:
            print(f"\n[TEST] Query: '{query}'")
            
            # Only test retrieval speed (no LLM call)
            query_latencies = []
            for i in range(3):
                start_time = time.time()
                _ = self.engine.retrieve_faculty(query, top_k=5)
                latency = time.time() - start_time
                query_latencies. append(latency)
            
            avg_latency = np.mean(query_latencies)
            latencies.append(avg_latency)
            
            result = {
                "query": query,
                "avg_latency_seconds": float(avg_latency),
                "min_latency_seconds": float(min(query_latencies)),
                "max_latency_seconds": float(max(query_latencies))
            }
            
            self.results["latency_tests"]. append(result)
            
            print(f"[RESULT] Avg Latency: {avg_latency:.3f}s")
            print(f"[RESULT] Range: {min(query_latencies):.3f}s - {max(query_latencies):.3f}s")
        
        overall_avg = np.mean(latencies)
        overall_std = np.std(latencies)
        
        print("\n" + "-" * 70)
        print("LATENCY METRICS (Retrieval Only):")
        print(f"  Average Latency:  {overall_avg:.3f}s")
        print(f"  Std Deviation: {overall_std:.3f}s")
        print(f"  Min Latency:  {min(latencies):.3f}s")
        print(f"  Max Latency:  {max(latencies):.3f}s")
        print("-" * 70)
    
    def run_baseline_comparison(self):
        """Test 6: Baseline Comparison (Ablation Study)"""
        print("\n" + "=" * 70)
        print("TEST 6: BASELINE COMPARISON (ABLATION STUDY)")
        print("=" * 70)
        
        test_query = "Who does machine learning and artificial intelligence research?"
        
        print(f"\n[TEST] Query: '{test_query}'")
        
        # Configuration 1: Full System
        print("\n[CONFIG 1] Full System (with query expansion)")
        results_full = self.engine.retrieve_faculty(test_query, top_k=5)
        scores_full = [r['score'] for r in results_full]
        avg_full = np.mean(scores_full) if scores_full else 0
        print(f"[RESULT] Top faculty: {[r['name'] for r in results_full[:3]]}")
        print(f"[RESULT] Avg Score: {avg_full:.3f}")
        
        # Configuration 2: Without Query Expansion
        print("\n[CONFIG 2] Without Query Expansion")
        q_emb = self.engine. embed_model.encode([test_query])[0]
        q_emb = q_emb / (np.linalg. norm(q_emb) + 1e-12)
        sims = self.engine.embeddings @ q_emb
        top_k_idxs = np.argsort(sims)[::-1][:5]
        scores_no_exp = [float(sims[i]) for i in top_k_idxs]
        avg_no_exp = np.mean(scores_no_exp)
        names_no_exp = [self.engine.faculty_ids[i] for i in top_k_idxs]
        print(f"[RESULT] Top faculty:  {names_no_exp[: 3]}")
        print(f"[RESULT] Avg Score: {avg_no_exp:.3f}")
        
        # Configuration 3: Smaller k (only top-3)
        print("\n[CONFIG 3] Top-3 Retrieval (k=3)")
        results_k3 = self.engine.retrieve_faculty(test_query, top_k=3)
        scores_k3 = [r['score'] for r in results_k3]
        avg_k3 = np.mean(scores_k3) if scores_k3 else 0
        print(f"[RESULT] Top faculty: {[r['name'] for r in results_k3]}")
        print(f"[RESULT] Avg Score: {avg_k3:.3f}")
        
        perf_drop = ((avg_full - avg_no_exp) / avg_full) * 100 if avg_full > 0 else 0
        
        self.results["baseline_comparison"] = {
            "test_query": test_query,
            "full_system": {
                "avg_score": float(avg_full),
                "top_faculty": [r['name'] for r in results_full[:3]],
                "all_scores": [float(s) for s in scores_full]
            },
            "without_query_expansion": {
                "avg_score": float(avg_no_exp),
                "top_faculty": names_no_exp[:3],
                "all_scores": scores_no_exp,
                "performance_drop_percent": float(perf_drop)
            },
            "top_k_3": {
                "avg_score": float(avg_k3),
                "top_faculty": [r['name'] for r in results_k3],
                "all_scores": [float(s) for s in scores_k3]
            }
        }
        
        print("\n" + "-" * 70)
        print("BASELINE COMPARISON SUMMARY:")
        print(f"  Full System: {avg_full:.3f}")
        print(f"  Without Expansion: {avg_no_exp:.3f} ({perf_drop:+.1f}%)")
        print(f"  Top-3 Only: {avg_k3:.3f}")
        print("-" * 70)
    
    def save_results(self):
        """Save all results to JSON file"""
        output_file = RESULTS_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Results saved to: {output_file}")
        
        # Also save latest results
        latest_file = RESULTS_DIR / "latest_results.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Latest results saved to: {latest_file}")
        
        return output_file
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n🚀 STARTING COMPREHENSIVE TEST SUITE\n")
        
        try:
            self.test_retrieval_quality()
            self.test_fuzzy_name_matching()
            self.test_query_expansion_impact()
            self.test_special_queries()
            self.test_response_latency()
            self.run_baseline_comparison()
            
            output_file = self.save_results()
            
            print("\n" + "=" * 70)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\nResults saved to: {output_file}")
            print("\nSummary:")
            print(f"  - Retrieval tests: {len(self.results['retrieval_tests'])} domains")
            print(f"  - Fuzzy match tests: {len(self.results['fuzzy_match_tests'])} cases")
            print(f"  - Query expansion tests: {len(self.results['query_expansion_tests'])} queries")
            print(f"  - Special query tests: {len(self.results['special_query_tests'])} cases")
            print(f"  - Latency tests: {len(self.results['latency_tests'])} queries")
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    suite = ComprehensiveTestSuite()
    suite.run_all_tests()


if __name__ == "__main__":
    main()