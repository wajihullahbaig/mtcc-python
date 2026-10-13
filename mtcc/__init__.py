"""MTCC: Minutia Texture Cylinder Codes for fingerprint matching (Baig et al., arXiv:1807.02251)."""
from .config import Params
from .cylinder import VARIANTS
from .extractors import EXTRACTORS, get_extractor
from .match import MATCHERS, match, match_all, stages
from .pipeline import extract_features, make_template, load_template, read_image, save_template, templates
