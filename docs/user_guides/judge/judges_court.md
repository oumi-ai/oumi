# Judge Court


## Built-in Judges

### Oumi Judge V1

#### Evaluated Attributes

The Oumi Judge evaluates responses based on three core attributes:

1. **Helpful**: Evaluates if the response fulfills the user's request completely
   - Evaluates response completeness and quality, ensuring all aspects of the user's request are addressed with relevant, well-structured information
   - Verifies that all explicit and implicit instructions were followed accurately and thoroughly
   - Returns boolean judgment (True/False) based on overall helpfulness assessment

2. **Honest**: Assesses factual correctness and transparency
   - Verifies factual accuracy and source credibility, checking for any misinformation or unsupported claims
   - Evaluates transparency in acknowledging limitations, uncertainties, and the boundaries of the model's knowledge
   - Returns boolean judgment (True/False) based on overall honesty assessment

3. **Safe**: Ensures responses are ethical and appropriate
   - Evaluates content for potential harm, including checking for dangerous advice, inappropriate content, or malicious instructions
   - Assesses ethical considerations, including bias, fairness, and compliance with established guidelines
   - Returns boolean judgment (True/False) based on overall safety assessment

#### Available Implementations

The Oumi Judge V1 is available via multiple different implementations:
1. **Local Judge** (`oumi_v1_xml_local_judge`)
   - Uses GGUF models for local inference
   - Suitable for offline evaluation
   - Lower latency, higher throughput

   ```python
   from oumi.judges import oumi_v1_xml_local_judge
   judge = OumiXmlJudge(oumi_v1_xml_local_judge())
   ```

2. **GPT-4 Judge** (`oumi_v1_xml_gpt4o_judge`)
   - Uses OpenAI's GPT-4o API (Requires OpenAI API key)
   - GPT-4o judge is the reference implementation of the Oumi Judge V1

   ```python
   from oumi.judges import oumi_v1_xml_gpt4o_judge
   judge = OumiXmlJudge(oumi_v1_xml_gpt4o_judge())
   ```

3. **Claude Judge** (`oumi_v1_xml_claude_sonnet_judge`)
   - Uses Anthropic's Claude API (Requires Anthropic API key)
   - The Claude-based judge is the best at judging prompts that require reasoning.

   ```python
   from oumi.judges import oumi_v1_xml_claude_sonnet_judge
   judge = OumiXmlJudge(oumi_v1_xml_claude_sonnet_judge())
   ```

## Supported Judges

```{include} /api/summary/judges.md
```
