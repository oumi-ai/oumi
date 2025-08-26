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

"""Integration tests for multimodal chat workflows with vision and text."""

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from tests.utils.chat_test_utils import (
    ChatTestSession,
    create_test_inference_config,
)


class TestVisionLanguageWorkflows:
    """Test suite for vision-language model workflows."""

    @pytest.fixture
    def vlm_session(self):
        """Create a test session configured for vision-language model."""
        config = create_test_inference_config()
        config.model.model_name = "SmolVLM-256M-Instruct"  # Vision-language model
        return ChatTestSession(config)

    def create_test_image(self, width=100, height=100, color=(255, 0, 0)):
        """Create a simple test image."""
        image = Image.new("RGB", (width, height), color)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

    def test_image_analysis_workflow(self, vlm_session):
        """Test basic image analysis workflow."""
        vlm_session.start_session()

        # Create test image
        test_image = self.create_test_image(color=(0, 255, 0))  # Green image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            temp_img.write(test_image)
            temp_img.flush()

            try:
                # Attach the image
                attach_result = vlm_session.execute_command(f"/attach({temp_img.name})")
                if not attach_result.success:
                    pytest.skip("Image attachment not supported")

                # Analyze the image
                analysis_result = vlm_session.send_message(
                    "What do you see in this image?"
                )
                assert analysis_result.success

                # Ask specific questions about the image
                color_result = vlm_session.send_message(
                    "What color is predominant in this image?"
                )
                assert color_result.success

                # Request detailed description
                detail_result = vlm_session.send_message(
                    "Can you provide a detailed description?"
                )
                assert detail_result.success
            finally:
                Path(temp_img.name).unlink(missing_ok=True)

    def test_multiple_image_comparison_workflow(self, vlm_session):
        """Test workflow comparing multiple images."""
        vlm_session.start_session()

        # Create different test images
        red_image = self.create_test_image(color=(255, 0, 0))
        blue_image = self.create_test_image(color=(0, 0, 255))

        temp_files = []
        try:
            # Save images
            for i, img_data in enumerate([red_image, blue_image]):
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f"_image_{i}.png", delete=False
                )
                temp_file.write(img_data)
                temp_file.close()
                temp_files.append(temp_file.name)

            # Attach first image
            attach1_result = vlm_session.execute_command(f"/attach({temp_files[0]})")
            if not attach1_result.success:
                pytest.skip("Image attachment not supported")

            vlm_session.send_message("This is the first image.")

            # Attach second image
            attach2_result = vlm_session.execute_command(f"/attach({temp_files[1]})")
            assert attach2_result.success

            # Compare images
            compare_result = vlm_session.send_message(
                "Now I've shown you two images. Can you compare them and tell me the main differences?"
            )
            assert compare_result.success

            # Ask for specific comparison
            color_compare_result = vlm_session.send_message(
                "Which image would be better for representing danger?"
            )
            assert color_compare_result.success

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    def test_image_and_document_analysis_workflow(self, vlm_session):
        """Test workflow combining image and text document analysis."""
        vlm_session.start_session()

        # Create test image
        chart_image = self.create_test_image(
            width=200, height=150, color=(100, 150, 200)
        )

        # Create related text document
        report_text = """
        # Sales Report Q1 2025

        ## Key Metrics
        - Total Revenue: $245,000
        - Growth Rate: 15% YoY
        - Customer Acquisition: 450 new customers
        - Customer Satisfaction: 4.2/5

        ## Analysis
        The chart shows positive trends across all metrics.
        Revenue growth exceeded expectations due to strong
        product adoption in the enterprise segment.
        """

        temp_files = []
        try:
            # Save image
            img_temp = tempfile.NamedTemporaryFile(suffix="_chart.png", delete=False)
            img_temp.write(chart_image)
            img_temp.close()
            temp_files.append(img_temp.name)

            # Save text document
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_report.md", delete=False
            ) as text_temp:
                text_temp.write(report_text)
                temp_files.append(text_temp.name)

            # Attach both files
            img_attach_result = vlm_session.execute_command(f"/attach({img_temp.name})")
            if not img_attach_result.success:
                pytest.skip("Image attachment not supported")

            text_attach_result = vlm_session.execute_command(
                f"/attach({text_temp.name})"
            )
            assert text_attach_result.success

            # Request combined analysis
            combined_result = vlm_session.send_message(
                "I've provided both a chart image and a text report. "
                "Can you analyze whether the visual data matches the written report?"
            )
            assert combined_result.success

            # Ask for insights
            insight_result = vlm_session.send_message(
                "Based on both the image and text, what insights can you provide about the business performance?"
            )
            assert insight_result.success

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    def test_image_editing_instructions_workflow(self, vlm_session):
        """Test workflow providing image editing instructions."""
        vlm_session.start_session()

        # Create test image
        original_image = self.create_test_image(
            width=300, height=200, color=(128, 128, 128)
        )

        with tempfile.NamedTemporaryFile(
            suffix="_original.png", delete=False
        ) as temp_img:
            temp_img.write(original_image)
            temp_img.flush()

            try:
                # Attach image
                attach_result = vlm_session.execute_command(f"/attach({temp_img.name})")
                if not attach_result.success:
                    pytest.skip("Image attachment not supported")

                # Request editing suggestions
                edit_result = vlm_session.send_message(
                    "Looking at this image, what editing improvements would you suggest?"
                )
                assert edit_result.success

                # Ask for specific editing steps
                steps_result = vlm_session.send_message(
                    "Can you provide step-by-step instructions to make this image more visually appealing?"
                )
                assert steps_result.success

                # Request technical editing advice
                tech_result = vlm_session.send_message(
                    "What technical parameters (brightness, contrast, saturation) would you adjust?"
                )
                assert tech_result.success

            finally:
                Path(temp_img.name).unlink(missing_ok=True)


class TestCodeVisualizationWorkflows:
    """Test suite for code visualization and diagram analysis workflows."""

    @pytest.fixture
    def code_vlm_session(self):
        """Create a session for code visualization tasks."""
        config = create_test_inference_config()
        config.model.model_name = "SmolVLM-256M-Instruct"
        return ChatTestSession(config)

    def test_flowchart_analysis_workflow(self, code_vlm_session):
        """Test analyzing flowchart images and generating code."""
        code_vlm_session.start_session()

        # Create a simple colored image representing a flowchart
        flowchart_image = self.create_test_image(
            width=400, height=300, color=(240, 240, 240)
        )

        with tempfile.NamedTemporaryFile(
            suffix="_flowchart.png", delete=False
        ) as temp_img:
            temp_img.write(flowchart_image)
            temp_img.flush()

            try:
                # Attach flowchart
                attach_result = code_vlm_session.execute_command(
                    f"/attach({temp_img.name})"
                )
                if not attach_result.success:
                    pytest.skip("Image attachment not supported")

                # Analyze the flowchart
                analysis_result = code_vlm_session.send_message(
                    "This is a flowchart diagram. Can you describe the logic flow you see?"
                )
                assert analysis_result.success

                # Request code generation
                code_result = code_vlm_session.send_message(
                    "Based on this flowchart, can you generate Python code that implements this logic?"
                )
                assert code_result.success

                # Ask for optimization suggestions
                optimize_result = code_vlm_session.send_message(
                    "How could this flowchart be optimized or simplified?"
                )
                assert optimize_result.success

            finally:
                Path(temp_img.name).unlink(missing_ok=True)

    def test_diagram_to_code_workflow(self, code_vlm_session):
        """Test converting architectural diagrams to code structure."""
        code_vlm_session.start_session()

        # Create test diagram image
        diagram_image = self.create_test_image(
            width=500, height=400, color=(200, 220, 255)
        )

        # Create accompanying specification
        spec_text = """
        # System Architecture Specification

        ## Components
        - API Gateway: Handles incoming requests
        - Authentication Service: User verification
        - Business Logic Layer: Core processing
        - Database Layer: Data persistence

        ## Flow
        1. Request -> API Gateway
        2. Authentication check
        3. Business logic processing
        4. Database operations
        5. Response return
        """

        temp_files = []
        try:
            # Save diagram image
            img_temp = tempfile.NamedTemporaryFile(
                suffix="_architecture.png", delete=False
            )
            img_temp.write(diagram_image)
            img_temp.close()
            temp_files.append(img_temp.name)

            # Save specification
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_spec.md", delete=False
            ) as spec_temp:
                spec_temp.write(spec_text)
                temp_files.append(spec_temp.name)

            # Attach both files
            img_result = code_vlm_session.execute_command(f"/attach({img_temp.name})")
            if not img_result.success:
                pytest.skip("Image attachment not supported")

            spec_result = code_vlm_session.execute_command(f"/attach({spec_temp.name})")
            assert spec_result.success

            # Request architecture analysis
            arch_result = code_vlm_session.send_message(
                "Based on the architecture diagram and specification, "
                "can you create a basic code structure for this system?"
            )
            assert arch_result.success

            # Ask for specific implementation
            impl_result = code_vlm_session.send_message(
                "Can you show me how to implement the API Gateway component in Python?"
            )
            assert impl_result.success

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    def create_test_image(self, width=100, height=100, color=(255, 0, 0)):
        """Create a simple test image."""
        image = Image.new("RGB", (width, height), color)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        return img_buffer.getvalue()


class TestScreenshotAnalysisWorkflows:
    """Test suite for screenshot and UI analysis workflows."""

    @pytest.fixture
    def ui_vlm_session(self):
        """Create a session for UI analysis tasks."""
        config = create_test_inference_config()
        config.model.model_name = "SmolVLM-256M-Instruct"
        return ChatTestSession(config)

    def create_test_image(self, width=100, height=100, color=(255, 0, 0)):
        """Create a simple test image."""
        image = Image.new("RGB", (width, height), color)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

    def test_ui_analysis_workflow(self, ui_vlm_session):
        """Test analyzing UI screenshots and providing feedback."""
        ui_vlm_session.start_session()

        # Create test UI screenshot
        ui_screenshot = self.create_test_image(
            width=800, height=600, color=(245, 245, 245)
        )

        with tempfile.NamedTemporaryFile(
            suffix="_ui_screenshot.png", delete=False
        ) as temp_img:
            temp_img.write(ui_screenshot)
            temp_img.flush()

            try:
                # Attach screenshot
                attach_result = ui_vlm_session.execute_command(
                    f"/attach({temp_img.name})"
                )
                if not attach_result.success:
                    pytest.skip("Image attachment not supported")

                # Analyze UI design
                ui_result = ui_vlm_session.send_message(
                    "This is a screenshot of a user interface. What elements do you see and how would you improve the design?"
                )
                assert ui_result.success

                # Ask for accessibility feedback
                a11y_result = ui_vlm_session.send_message(
                    "From an accessibility perspective, what improvements would you suggest?"
                )
                assert a11y_result.success

                # Request user experience analysis
                ux_result = ui_vlm_session.send_message(
                    "How could the user experience be improved based on this interface?"
                )
                assert ux_result.success

            finally:
                Path(temp_img.name).unlink(missing_ok=True)

    def test_error_screenshot_debugging_workflow(self, ui_vlm_session):
        """Test debugging workflow using error screenshots."""
        ui_vlm_session.start_session()

        # Create test error screenshot
        error_screenshot = self.create_test_image(
            width=600, height=400, color=(255, 200, 200)
        )

        # Create error log
        error_log = """
        ERROR: Application crashed at startup
        Stack trace:
        File "main.py", line 45, in initialize()
        File "config.py", line 23, in load_config()
        ConfigurationError: Missing required parameter 'api_key'

        Last user action: Clicked 'Start Application' button
        Browser: Chrome 91.0
        OS: macOS 12.1
        """

        temp_files = []
        try:
            # Save error screenshot
            img_temp = tempfile.NamedTemporaryFile(suffix="_error.png", delete=False)
            img_temp.write(error_screenshot)
            img_temp.close()
            temp_files.append(img_temp.name)

            # Save error log
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_error_log.txt", delete=False
            ) as log_temp:
                log_temp.write(error_log)
                temp_files.append(log_temp.name)

            # Attach both files
            img_result = ui_vlm_session.execute_command(f"/attach({img_temp.name})")
            if not img_result.success:
                pytest.skip("Image attachment not supported")

            log_result = ui_vlm_session.execute_command(f"/attach({log_temp.name})")
            assert log_result.success

            # Request debugging help
            debug_result = ui_vlm_session.send_message(
                "I have an error screenshot and error log. Can you help me understand what went wrong?"
            )
            assert debug_result.success

            # Ask for solution steps
            solution_result = ui_vlm_session.send_message(
                "What steps should I take to fix this issue?"
            )
            assert solution_result.success

            # Request prevention advice
            prevent_result = ui_vlm_session.send_message(
                "How can I prevent this type of error in the future?"
            )
            assert prevent_result.success

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    def test_before_after_ui_comparison_workflow(self, ui_vlm_session):
        """Test comparing before/after UI changes."""
        ui_vlm_session.start_session()

        # Create before and after images
        before_ui = self.create_test_image(width=600, height=400, color=(220, 220, 220))
        after_ui = self.create_test_image(width=600, height=400, color=(240, 240, 255))

        temp_files = []
        try:
            # Save images
            before_temp = tempfile.NamedTemporaryFile(
                suffix="_before.png", delete=False
            )
            before_temp.write(before_ui)
            before_temp.close()
            temp_files.append(before_temp.name)

            after_temp = tempfile.NamedTemporaryFile(suffix="_after.png", delete=False)
            after_temp.write(after_ui)
            after_temp.close()
            temp_files.append(after_temp.name)

            # Attach before image
            before_result = ui_vlm_session.execute_command(
                f"/attach({before_temp.name})"
            )
            if not before_result.success:
                pytest.skip("Image attachment not supported")

            ui_vlm_session.send_message("This is the before state of our UI.")

            # Attach after image
            after_result = ui_vlm_session.execute_command(f"/attach({after_temp.name})")
            assert after_result.success

            # Request comparison
            compare_result = ui_vlm_session.send_message(
                "Now I've shown you the after state. Can you compare the two versions and tell me what changed?"
            )
            assert compare_result.success

            # Ask for improvement assessment
            improvement_result = ui_vlm_session.send_message(
                "Do you think the changes are an improvement? Why or why not?"
            )
            assert improvement_result.success

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)
