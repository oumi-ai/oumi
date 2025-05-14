from transformers.models.phi3.configuration_phi3 import Phi3Config

# Consider using Cambrian clone of `modeling_phi3.py`,
# which has some XLA special cases.
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Model
