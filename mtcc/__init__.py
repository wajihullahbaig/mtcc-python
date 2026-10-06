"""MTCC: Minutia Texture Cylinder Codes for fingerprint matching (Baig et al., arXiv:1807.02251)."""
from .config import Params
from .cylinder import VARIANTS, make_template
from .match import match
from .pipeline import extract_features, load_template, read_image, save_template, templates
