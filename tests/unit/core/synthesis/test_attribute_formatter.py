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

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    SampledAttribute,
    SampledAttributeValue,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter


def test_format_with_sampled_attributes():
    """Test formatting with sampled attributes."""
    # Setup sampled attributes
    tone_values = [
        SampledAttributeValue(
            id="formal",
            name="Formal",
            description="A formal tone of voice",
        ),
        SampledAttributeValue(
            id="casual",
            name="Casual",
            description="A casual tone of voice",
        ),
    ]

    style_values = [
        SampledAttributeValue(
            id="concise",
            name="Concise",
            description="Brief and to the point",
        ),
        SampledAttributeValue(
            id="detailed",
            name="Detailed",
            description="Comprehensive and thorough",
        ),
    ]

    sampled_attrs = [
        SampledAttribute(
            id="tone",
            name="Tone",
            description="The tone of voice to use",
            possible_values=tone_values,
        ),
        SampledAttribute(
            id="style",
            name="Style",
            description="The writing style to use",
            possible_values=style_values,
        ),
    ]

    params = GeneralSynthesisParams(sampled_attributes=sampled_attrs)
    formatter = AttributeFormatter(params)

    # Test formatting
    sample = {"tone": "formal", "style": "concise"}
    format_string = "Use {tone} tone and {style} style"

    result = formatter.format(sample, format_string)
    expected = "Use Formal tone and Concise style"

    assert result == expected


def test_format_with_mixed_attributes():
    """Test formatting with both sampled and non-sampled attributes."""
    tone_values = [
        SampledAttributeValue(
            id="friendly",
            name="Friendly",
            description="A friendly tone",
        ),
    ]

    sampled_attrs = [
        SampledAttribute(
            id="tone",
            name="Tone",
            description="The tone to use",
            possible_values=tone_values,
        ),
    ]

    params = GeneralSynthesisParams(sampled_attributes=sampled_attrs)
    formatter = AttributeFormatter(params)

    sample = {"tone": "friendly", "name": "Alice"}
    format_string = "Use {tone} tone when talking to {name}"

    result = formatter.format(sample, format_string)
    expected = "Use Friendly tone when talking to Alice"

    assert result == expected


def test_format_with_missing_values():
    """Test formatting with missing values allowed and not allowed."""
    params = GeneralSynthesisParams()
    formatter = AttributeFormatter(params)

    sample = {"name": "John"}
    format_string = "Hello {name}, your age is {age}"

    result = formatter.format(sample, format_string, missing_values_allowed=True)
    expected = "Hello John, your age is {age}"

    assert result == expected

    with pytest.raises(ValueError, match="Missing value for placeholder: age"):
        formatter.format(sample, format_string, missing_values_allowed=False)


def test_format_with_invalid_sampled_attribute_value():
    """Test formatting with invalid sampled attribute value."""
    tone_values = [
        SampledAttributeValue(
            id="formal",
            name="Formal",
            description="A formal tone",
        ),
    ]

    sampled_attrs = [
        SampledAttribute(
            id="tone",
            name="Tone",
            description="The tone to use",
            possible_values=tone_values,
        ),
    ]

    params = GeneralSynthesisParams(sampled_attributes=sampled_attrs)
    formatter = AttributeFormatter(params)

    sample = {"tone": "invalid_value"}
    format_string = "Use {tone.value} tone"

    with pytest.raises(
        ValueError, match="Attribute value invalid_value not found for attribute tone"
    ):
        formatter.format(sample, format_string)


def test_format_with_empty_sample():
    """Test formatting with empty sample."""
    params = GeneralSynthesisParams()
    formatter = AttributeFormatter(params)

    sample = {}
    format_string = "This is a static string"

    result = formatter.format(sample, format_string)
    expected = "This is a static string"

    assert result == expected


def test_format_with_empty_format_string():
    """Test formatting with empty format string."""
    params = GeneralSynthesisParams()
    formatter = AttributeFormatter(params)

    sample = {"name": "John"}
    format_string = ""

    result = formatter.format(sample, format_string)
    expected = ""

    assert result == expected


def test_format_with_multiple_same_placeholder():
    """Test formatting with the same placeholder used multiple times."""
    tone_values = [
        SampledAttributeValue(
            id="excited",
            name="Excited",
            description="An excited tone",
        ),
    ]

    sampled_attrs = [
        SampledAttribute(
            id="tone",
            name="Tone",
            description="The tone to use",
            possible_values=tone_values,
        ),
    ]

    params = GeneralSynthesisParams(sampled_attributes=sampled_attrs)
    formatter = AttributeFormatter(params)

    sample = {"tone": "excited"}
    format_string = "Use {tone} tone! Yes, {tone} tone!"

    result = formatter.format(sample, format_string)
    expected = "Use Excited tone! Yes, Excited tone!"

    assert result == expected


def test_format_with_attribute_name_vs_value():
    """Test the difference between {attribute} and {attribute.value}."""
    tone_values = [
        SampledAttributeValue(
            id="polite",
            name="Polite",
            description="A polite tone",
        ),
    ]

    sampled_attrs = [
        SampledAttribute(
            id="tone",
            name="Tone",
            description="The tone to use",
            possible_values=tone_values,
        ),
    ]

    params = GeneralSynthesisParams(sampled_attributes=sampled_attrs)
    formatter = AttributeFormatter(params)

    sample = {"tone": "polite"}

    # Test attribute name
    format_string = "The {tone.parent} should be polite"
    result = formatter.format(sample, format_string)
    expected = "The Tone should be polite"
    assert result == expected

    # Test attribute value
    format_string = "The tone should be {tone}"
    result = formatter.format(sample, format_string)
    expected = "The tone should be Polite"
    assert result == expected

    # Test attribute description
    format_string = "The tone should be {tone.parent.description}"
    result = formatter.format(sample, format_string)
    expected = "The tone should be The tone to use"
    assert result == expected

    # Test value description
    format_string = "The tone should be {tone.description}"
    result = formatter.format(sample, format_string)
    expected = "The tone should be A polite tone"
    assert result == expected


def test_format_with_none_sampled_attributes():
    """Test formatting when sampled_attributes is None."""
    params = GeneralSynthesisParams(sampled_attributes=None)
    formatter = AttributeFormatter(params)

    sample = {"name": "Bob"}
    format_string = "Hello {name}"

    result = formatter.format(sample, format_string)
    expected = "Hello Bob"

    assert result == expected
