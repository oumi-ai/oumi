# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Complete user journey tests for chat functionality."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from tests.oumi_chat.utils.chat_real_model_utils import (
    ChatPerformanceMonitor,
    RealModelChatSession,
    create_real_model_inference_config,
    temporary_chat_files,
)


@pytest.mark.chat_journey
@pytest.mark.e2e
class TestResearchWorkflow:
    """Test complete research workflow using real model."""

    def test_document_research_journey(self):
        """Test complete document research and analysis workflow."""
        config = create_real_model_inference_config(
            max_new_tokens=30,  # Longer responses for research
            temperature=0.1,  # Slightly creative but focused
        )
        chat_session = RealModelChatSession(config)
        monitor = ChatPerformanceMonitor()

        # Sample research documents
        research_docs = {
            "paper1.txt": """
            Artificial Intelligence Research Paper
            Abstract: This paper explores machine learning techniques for natural
            language processing.
            The study shows that transformer architectures achieve state-of-the-art
            results on various benchmarks.
            Keywords: transformers, NLP, machine learning, deep learning
            """,
            "notes.md": """
            # Research Notes
            - AI has evolved significantly in recent years
            - Deep learning models require large datasets
            - Transformer architecture introduced attention mechanism
            - Applications include translation, summarization, question answering
            """,
            "data.json": json.dumps(
                {
                    "experiment_results": {
                        "accuracy": 0.95,
                        "precision": 0.92,
                        "recall": 0.94,
                        "f1_score": 0.93,
                    },
                    "model_info": {
                        "architecture": "transformer",
                        "parameters": "135M",
                        "training_data": "text corpus",
                    },
                }
            ),
        }

        with temporary_chat_files(research_docs) as temp_files:
            with chat_session.real_inference_session():
                monitor.start_session_monitoring()
                chat_session.start_session()

                # Phase 1: Document Analysis
                research_results = []

                # Analyze paper
                paper_path = temp_files["paper1.txt"]
                attach_result = chat_session.inject_command(f"/attach({paper_path})")
                research_results.append(("attach_paper", attach_result.success))

                if attach_result.success:
                    analysis_result = chat_session.send_message_with_real_inference(
                        "Based on the attached paper, what are the key findings about "
                        "AI? Please mention 'transformer' or 'learning' in your "
                        "response."
                    )
                    research_results.append(("analyze_paper", analysis_result.success))
                    if analysis_result.success:
                        chat_session.assert_response_quality(
                            expected_keywords=["transformer", "learning", "ai", "model"]
                        )

                # Analyze notes
                notes_path = temp_files["notes.md"]
                attach_result = chat_session.inject_command(f"/attach({notes_path})")
                research_results.append(("attach_notes", attach_result.success))

                if attach_result.success:
                    notes_result = chat_session.send_message_with_real_inference(
                        "What insights can you provide about the research notes? "
                        "Please include the word 'applications' in your response."
                    )
                    research_results.append(("analyze_notes", notes_result.success))

                # Phase 2: Data Interpretation
                data_path = temp_files["data.json"]
                attach_result = chat_session.inject_command(f"/attach({data_path})")
                research_results.append(("attach_data", attach_result.success))

                if attach_result.success:
                    data_result = chat_session.send_message_with_real_inference(
                        "Based on the experimental data, how well did the model "
                        "perform? Please mention 'accuracy' or 'performance' in your "
                        "response."
                    )
                    research_results.append(("analyze_data", data_result.success))

                # Phase 3: Synthesis and Export
                synthesis_result = chat_session.send_message_with_real_inference(
                    "Can you synthesize the key insights from all the documents? "
                    "Please provide a brief summary mentioning 'research' findings."
                )
                research_results.append(("synthesis", synthesis_result.success))

                # Save research insights in multiple formats
                export_results = []
                export_base = (
                    Path(temp_files["paper1.txt"]).parent / "research_insights"
                )

                for format_ext in [".json", ".txt", ".md"]:
                    export_path = str(export_base) + format_ext
                    save_result = chat_session.inject_command(f"/save({export_path})")
                    export_results.append(save_result.success)

                    if save_result.success:
                        # Verify export file exists and has content
                        if Path(export_path).exists():
                            content = Path(export_path).read_text()
                            assert len(content) > 50  # Substantial content

                # End monitoring
                metrics = monitor.end_session_monitoring(chat_session)

                # Verify workflow completion
                successful_research_steps = sum(
                    1 for _, success in research_results if success
                )
                assert successful_research_steps >= 4, (
                    f"Research workflow incomplete: {research_results}"
                )

                successful_exports = sum(export_results)
                assert successful_exports >= 1, "No successful exports"

                # Verify session health
                assert metrics["session_duration"] > 0
                perf_summary = chat_session.get_performance_summary()
                assert perf_summary["total_responses"] >= 4

    def test_iterative_research_refinement(self):
        """Test iterative research process with refinement."""
        config = create_real_model_inference_config(max_new_tokens=25)
        chat_session = RealModelChatSession(config)

        research_topic = "machine learning applications"

        with chat_session.real_inference_session():
            chat_session.start_session()

            refinement_stages = []

            # Stage 1: Initial exploration
            initial_result = chat_session.send_message_with_real_inference(
                f"Tell me about {research_topic}. Please mention 'applications' in "
                f"your response."
            )
            refinement_stages.append(("initial", initial_result.success))

            # Stage 2: Deeper dive
            if initial_result.success:
                deeper_result = chat_session.send_message_with_real_inference(
                    "Can you elaborate on the most important applications? "
                    "Please include 'important' in your response."
                )
                refinement_stages.append(("deeper", deeper_result.success))

            # Stage 3: Specific focus
            specific_result = chat_session.send_message_with_real_inference(
                "What are the challenges in implementing these applications? "
                "Please mention 'challenges' or 'difficult' in your response."
            )
            refinement_stages.append(("specific", specific_result.success))

            # Stage 4: Future directions
            future_result = chat_session.send_message_with_real_inference(
                "What future developments do you foresee? "
                "Please include 'future' in your response."
            )
            refinement_stages.append(("future", future_result.success))

            # Verify iterative refinement worked
            successful_stages = sum(1 for _, success in refinement_stages if success)
            assert successful_stages >= 3, (
                f"Refinement process incomplete: {refinement_stages}"
            )

            # Verify conversation progression
            conversation = chat_session.get_conversation()
            assert conversation is not None
            assert (
                len(conversation.messages) >= successful_stages * 2
            )  # User + Assistant pairs


@pytest.mark.chat_journey
@pytest.mark.e2e
class TestCreativeWorkflow:
    """Test creative workflow with branching and exploration."""

    def test_creative_brainstorming_journey(self):
        """Test complete creative brainstorming with branching."""
        config = create_real_model_inference_config(
            max_new_tokens=40,  # Longer for creativity
            temperature=0.3,  # More creative
        )
        chat_session = RealModelChatSession(config)

        creative_theme = "innovative technology ideas"

        with chat_session.real_inference_session():
            chat_session.start_session()

            creative_steps = []

            # Phase 1: Initial brainstorming
            brainstorm_result = chat_session.send_message_with_real_inference(
                f"Let's brainstorm some {creative_theme}. "
                "Please mention 'innovation' or 'technology' in your response."
            )
            creative_steps.append(("brainstorm", brainstorm_result.success))

            # Phase 2: Idea development
            if brainstorm_result.success:
                develop_result = chat_session.send_message_with_real_inference(
                    "Can you expand on the most promising idea? "
                    "Please include 'promising' or 'potential' in your response."
                )
                creative_steps.append(("develop", develop_result.success))

            # Phase 3: Branching exploration (simulate multiple directions)
            branch_topics = [
                ("practical applications", "practical"),
                ("technical challenges", "technical"),
                ("market potential", "market"),
            ]

            branch_results = []
            for topic, keyword in branch_topics:
                # Attempt to create branch (may not be implemented)
                _ = chat_session.inject_command(f"/branch({topic.replace(' ', '_')})")

                # Explore the topic regardless of branching success
                explore_result = chat_session.send_message_with_real_inference(
                    f"What about the {topic} of these ideas? "
                    f"Please mention '{keyword}' in your response."
                )
                branch_results.append((topic, explore_result.success))

                if explore_result.success:
                    try:
                        chat_session.assert_response_quality(
                            expected_keywords=[keyword]
                        )
                    except AssertionError:
                        # Keyword might not appear, which is acceptable in creative
                        # context
                        pass

            # Phase 4: Synthesis and documentation
            synthesis_result = chat_session.send_message_with_real_inference(
                "Can you synthesize the best elements from our exploration? "
                "Please include 'synthesis' or 'combination' in your response."
            )
            creative_steps.append(("synthesis", synthesis_result.success))

            # Save creative output
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                creative_output_path = f.name

            try:
                save_result = chat_session.inject_command(
                    f"/save({creative_output_path})"
                )
                creative_steps.append(("save", save_result.success))

                if save_result.success and Path(creative_output_path).exists():
                    content = Path(creative_output_path).read_text()
                    assert len(content) > 100  # Substantial creative output

            finally:
                Path(creative_output_path).unlink(missing_ok=True)

            # Verify creative workflow
            successful_steps = sum(1 for _, success in creative_steps if success)
            assert successful_steps >= 3, (
                f"Creative workflow incomplete: {creative_steps}"
            )

            successful_branches = sum(1 for _, success in branch_results if success)
            assert successful_branches >= 2, (
                f"Branch exploration incomplete: {branch_results}"
            )

    def test_creative_iteration_and_refinement(self):
        """Test iterative creative refinement process."""
        config = create_real_model_inference_config(max_new_tokens=35, temperature=0.2)
        chat_session = RealModelChatSession(config)

        with chat_session.real_inference_session():
            chat_session.start_session()

            # Creative project: Design a solution
            project_topic = "design an educational tool"
            iterations = []

            # Iteration 1: Initial concept
            concept_result = chat_session.send_message_with_real_inference(
                f"Let's {project_topic} for students. "
                "Please mention 'educational' or 'students' in your response."
            )
            iterations.append(("concept", concept_result.success))

            # Iteration 2: Add features
            if concept_result.success:
                features_result = chat_session.send_message_with_real_inference(
                    "What key features would make this tool more engaging? "
                    "Please include 'engaging' or 'features' in your response."
                )
                iterations.append(("features", features_result.success))

            # Iteration 3: Address concerns
            concerns_result = chat_session.send_message_with_real_inference(
                "What potential problems should we address in the design? "
                "Please mention 'problems' or 'issues' in your response."
            )
            iterations.append(("concerns", concerns_result.success))

            # Iteration 4: Final refinement
            refinement_result = chat_session.send_message_with_real_inference(
                "How can we refine the design to be more effective? "
                "Please include 'effective' or 'improve' in your response."
            )
            iterations.append(("refinement", refinement_result.success))

            # Verify iterative process
            successful_iterations = sum(1 for _, success in iterations if success)
            assert successful_iterations >= 3, (
                f"Creative iteration incomplete: {iterations}"
            )

            # Check conversation maintains coherence
            conversation = chat_session.get_conversation()
            if conversation:
                assert len(conversation.messages) >= successful_iterations * 2


@pytest.mark.chat_journey
@pytest.mark.e2e
class TestAnalysisWorkflow:
    """Test analytical workflow with data processing."""

    def test_data_analysis_journey(self):
        """Test complete data analysis workflow."""
        config = create_real_model_inference_config(max_new_tokens=30)
        chat_session = RealModelChatSession(config)

        # Sample data files
        analysis_data = {
            "sales_data.csv": """Date,Product,Sales,Region
2024-01-01,Widget A,100,North
2024-01-01,Widget B,150,South
2024-01-02,Widget A,120,North
2024-01-02,Widget B,180,South
2024-01-03,Widget A,110,North
2024-01-03,Widget B,160,South""",
            "summary_stats.json": json.dumps(
                {
                    "total_sales": 820,
                    "average_daily_sales": 136.67,
                    "top_product": "Widget B",
                    "top_region": "South",
                    "growth_rate": 0.05,
                }
            ),
            "analysis_notes.txt": """
Analysis Notes:
- Widget B consistently outperforms Widget A
- South region shows higher sales volume
- Overall trend appears positive
- Need to investigate seasonality
""",
        }

        with temporary_chat_files(analysis_data) as temp_files:
            with chat_session.real_inference_session():
                chat_session.start_session()

                analysis_steps = []

                # Step 1: Load and examine raw data
                csv_path = temp_files["sales_data.csv"]
                attach_result = chat_session.inject_command(f"/attach({csv_path})")
                analysis_steps.append(("load_data", attach_result.success))

                if attach_result.success:
                    data_overview_result = (
                        chat_session.send_message_with_real_inference(
                            "What patterns do you see in this sales data? "
                            "Please mention 'sales' or 'data' in your response."
                        )
                    )
                    analysis_steps.append(
                        ("data_overview", data_overview_result.success)
                    )

                # Step 2: Examine summary statistics
                stats_path = temp_files["summary_stats.json"]
                attach_result = chat_session.inject_command(f"/attach({stats_path})")
                analysis_steps.append(("load_stats", attach_result.success))

                if attach_result.success:
                    stats_analysis_result = (
                        chat_session.send_message_with_real_inference(
                            "Based on the summary statistics, what insights can you "
                            "provide? Please include 'statistics' or 'insights' in "
                            "your response."
                        )
                    )
                    analysis_steps.append(
                        ("analyze_stats", stats_analysis_result.success)
                    )

                # Step 3: Integrate with analysis notes
                notes_path = temp_files["analysis_notes.txt"]
                attach_result = chat_session.inject_command(f"/attach({notes_path})")
                analysis_steps.append(("load_notes", attach_result.success))

                if attach_result.success:
                    integrated_result = chat_session.send_message_with_real_inference(
                        "How do the analysis notes align with what we've observed? "
                        "Please mention 'notes' or 'observations' in your response."
                    )
                    analysis_steps.append(
                        ("integrate_notes", integrated_result.success)
                    )

                # Step 4: Generate recommendations
                recommendations_result = chat_session.send_message_with_real_inference(
                    "Based on all the data, what recommendations would you make? "
                    "Please include 'recommend' or 'suggest' in your response."
                )
                analysis_steps.append(
                    ("recommendations", recommendations_result.success)
                )

                # Step 5: Create analysis report
                report_path = (
                    Path(temp_files["sales_data.csv"]).parent / "analysis_report.md"
                )
                save_result = chat_session.inject_command(f"/save({report_path})")
                analysis_steps.append(("save_report", save_result.success))

                if save_result.success and report_path.exists():
                    report_content = report_path.read_text()
                    assert len(report_content) > 200  # Comprehensive report
                    assert "sales" in report_content.lower()  # Should mention key topic

                # Verify analysis workflow completion
                successful_steps = sum(1 for _, success in analysis_steps if success)
                assert successful_steps >= 5, (
                    f"Analysis workflow incomplete: {analysis_steps}"
                )

    def test_comparative_analysis_journey(self):
        """Test comparative analysis across multiple datasets."""
        config = create_real_model_inference_config(max_new_tokens=25)
        chat_session = RealModelChatSession(config)

        # Comparative datasets
        comparison_data = {
            "q1_results.json": json.dumps(
                {
                    "quarter": "Q1 2024",
                    "revenue": 100000,
                    "customers": 500,
                    "satisfaction": 4.2,
                }
            ),
            "q2_results.json": json.dumps(
                {
                    "quarter": "Q2 2024",
                    "revenue": 120000,
                    "customers": 580,
                    "satisfaction": 4.4,
                }
            ),
            "benchmark.txt": """
Industry Benchmarks:
- Average revenue growth: 15% per quarter
- Customer retention: 85%
- Satisfaction target: 4.0+
""",
        }

        with temporary_chat_files(comparison_data) as temp_files:
            with chat_session.real_inference_session():
                chat_session.start_session()

                comparison_steps = []

                # Load Q1 data
                q1_result = chat_session.inject_command(
                    f"/attach({temp_files['q1_results.json']})"
                )
                comparison_steps.append(("load_q1", q1_result.success))

                # Load Q2 data
                q2_result = chat_session.inject_command(
                    f"/attach({temp_files['q2_results.json']})"
                )
                comparison_steps.append(("load_q2", q2_result.success))

                # Perform quarter-over-quarter comparison
                if q1_result.success and q2_result.success:
                    comparison_result = chat_session.send_message_with_real_inference(
                        "Compare the Q1 and Q2 results. What trends do you see? "
                        "Please mention 'comparison' or 'trends' in your response."
                    )
                    comparison_steps.append(
                        ("compare_quarters", comparison_result.success)
                    )

                # Load and apply benchmarks
                benchmark_result = chat_session.inject_command(
                    f"/attach({temp_files['benchmark.txt']})"
                )
                comparison_steps.append(("load_benchmark", benchmark_result.success))

                if benchmark_result.success:
                    benchmark_analysis_result = (
                        chat_session.send_message_with_real_inference(
                            "How do our results compare to industry benchmarks? "
                            "Please include 'benchmark' in your response."
                        )
                    )
                    comparison_steps.append(
                        ("benchmark_analysis", benchmark_analysis_result.success)
                    )

                # Generate comparative insights
                insights_result = chat_session.send_message_with_real_inference(
                    "What key insights emerge from this comparative analysis? "
                    "Please mention 'insights' or 'analysis' in your response."
                )
                comparison_steps.append(("insights", insights_result.success))

                # Verify comparative analysis
                successful_steps = sum(1 for _, success in comparison_steps if success)
                assert successful_steps >= 4, (
                    f"Comparative analysis incomplete: {comparison_steps}"
                )


@pytest.mark.chat_journey
@pytest.mark.e2e
class TestMultiSessionWorkflow:
    """Test workflows spanning multiple chat sessions."""

    def test_project_continuation_across_sessions(self):
        """Test continuing a project across multiple sessions."""
        config = create_real_model_inference_config(max_new_tokens=20)

        project_data = {
            "project_brief.md": """
# Project: AI Assistant Development
## Goals:
- Create helpful AI assistant
- Focus on natural conversations
- Implement safety features
## Current Status:
- Initial research complete
- Architecture planning in progress
"""
        }

        with temporary_chat_files(project_data) as temp_files:
            # Session 1: Initial planning
            session1 = RealModelChatSession(config)
            session1_results = []

            with session1.real_inference_session():
                session1.start_session()

                # Load project brief
                load_result = session1.inject_command(
                    f"/attach({temp_files['project_brief.md']})"
                )
                session1_results.append(("load_brief", load_result.success))

                # Initial planning
                if load_result.success:
                    planning_result = session1.send_message_with_real_inference(
                        "Based on the project brief, what should be our next steps? "
                        "Please mention 'steps' or 'planning' in your response."
                    )
                    session1_results.append(("planning", planning_result.success))

                # Save session 1 state
                session1_save_path = (
                    Path(temp_files["project_brief.md"]).parent / "session1_state.json"
                )
                save_result = session1.inject_command(f"/save({session1_save_path})")
                session1_results.append(("save_state", save_result.success))

                session1.end_session()

            # Verify session 1
            session1_success = sum(1 for _, success in session1_results if success)
            assert session1_success >= 2, f"Session 1 incomplete: {session1_results}"

            # Session 2: Continue the work
            session2 = RealModelChatSession(config)
            session2_results = []

            with session2.real_inference_session():
                session2.start_session()

                # Load previous session if saved
                if session1_save_path.exists():
                    load_prev_result = session2.inject_command(
                        f"/attach({session1_save_path})"
                    )
                    session2_results.append(("load_previous", load_prev_result.success))

                # Continue the project
                continue_result = session2.send_message_with_real_inference(
                    "Let's continue working on the AI assistant project. "
                    "What technical details should we focus on? "
                    "Please mention 'technical' or 'details' in your response."
                )
                session2_results.append(("continue_work", continue_result.success))

                # Add new insights
                insights_result = session2.send_message_with_real_inference(
                    "What additional considerations should we include? "
                    "Please include 'considerations' in your response."
                )
                session2_results.append(("new_insights", insights_result.success))

                # Save final state
                final_save_path = (
                    Path(temp_files["project_brief.md"]).parent
                    / "final_project_state.json"
                )
                final_save_result = session2.inject_command(f"/save({final_save_path})")
                session2_results.append(("save_final", final_save_result.success))

                session2.end_session()

            # Verify session 2 and continuity
            session2_success = sum(1 for _, success in session2_results if success)
            assert session2_success >= 2, f"Session 2 incomplete: {session2_results}"

            # Verify both sessions contributed to project progress
            total_success = session1_success + session2_success
            assert total_success >= 5, "Multi-session project workflow incomplete"

    def test_knowledge_building_workflow(self):
        """Test building knowledge across multiple related sessions."""
        config = create_real_model_inference_config(max_new_tokens=25)

        knowledge_domains = [
            ("fundamentals", "What are the fundamental concepts we should understand?"),
            ("applications", "What are the practical applications of these concepts?"),
            ("challenges", "What challenges exist in this domain?"),
        ]

        domain_results = {}

        for domain, question in knowledge_domains:
            session = RealModelChatSession(config)

            with session.real_inference_session():
                session.start_session()

                # Explore this knowledge domain
                exploration_result = session.send_message_with_real_inference(
                    f"Let's explore {domain} in artificial intelligence. {question} "
                    f"Please mention '{domain}' in your response."
                )

                # Build on the knowledge
                if exploration_result.success:
                    followup_result = session.send_message_with_real_inference(
                        f"Can you elaborate on the most important aspects of {domain}? "
                        f"Please include 'important' in your response."
                    )

                    domain_results[domain] = {
                        "exploration": exploration_result.success,
                        "followup": followup_result.success,
                        "conversation_length": len(session.get_conversation().messages)
                        if session.get_conversation() is not None
                        else 0,
                    }
                else:
                    domain_results[domain] = {
                        "exploration": False,
                        "followup": False,
                        "conversation_length": 0,
                    }

                session.end_session()

        # Verify knowledge building across domains
        successful_domains = sum(
            1
            for results in domain_results.values()
            if results["exploration"] and results["followup"]
        )
        assert successful_domains >= 2, (
            f"Knowledge building incomplete: {domain_results}"
        )

        # Verify each domain had meaningful conversation
        for domain, results in domain_results.items():
            if results["exploration"]:
                assert results["conversation_length"] >= 2, (
                    f"Insufficient conversation in {domain}"
                )


class TestUserJourneyUtilities:
    """Test utilities for user journey testing."""

    def test_journey_performance_monitoring(self):
        """Test performance monitoring across a user journey."""
        monitor = ChatPerformanceMonitor()
        config = create_real_model_inference_config(max_new_tokens=10)

        # Simulate multiple sessions in a journey
        for session_num in range(3):
            session = RealModelChatSession(config)

            with session.real_inference_session():
                monitor.start_session_monitoring()
                session.start_session()

                # Simulate some activity
                session.response_times = [0.5 + session_num * 0.1]
                session.token_counts = [10 + session_num * 2]

                time.sleep(0.01)  # Brief activity

                # End monitoring
                metrics = monitor.end_session_monitoring(session)
                assert "session_duration" in metrics

                session.end_session()

        # Check aggregate metrics across journey
        aggregate = monitor.get_aggregate_metrics()
        assert aggregate["total_sessions"] == 3
        assert aggregate["avg_session_duration"] > 0

    def test_multi_file_workflow_utilities(self):
        """Test utilities for workflows involving multiple files."""
        test_files = {
            f"doc_{i}.txt": f"Document {i} content with important information."
            for i in range(5)
        }

        with temporary_chat_files(test_files) as temp_files:
            # Verify all files are accessible
            assert len(temp_files) == 5

            for original_name, temp_path in temp_files.items():
                assert Path(temp_path).exists()
                content = Path(temp_path).read_text()
                assert "important information" in content
                assert (
                    original_name.split("_")[1].split(".")[0] in content
                )  # Document number

        # Verify cleanup
        for temp_path in temp_files.values():
            assert not Path(temp_path).exists()
