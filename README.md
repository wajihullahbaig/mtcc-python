# mtcc-python
Using June-2025, ClaudeAI-sonnet,Grok3, DeepSeek-V3 (non-thinking), Gemini Flash 2.5 (thinking) to implement a Fingerprint paper, MTCC (my paper from 2018)
As of this date, all of them have failed. None of them could produce correct code except for Claude
which corrected to a great level the orientation processing. 

[MTCC: Minutiae Template Cylinder Codes for Fingerprint Matching (arXiv:1807.02251)](https://arxiv.org/abs/1807.02251)

### Fail1
Minutia seem to be detected but thinning, orientation images, freqyuency iamges was a struggle. 
30 odd prompt requests per model but no success. MTCC paper knowledge based provided

#### Prompt

```
You need to implement. small conscise modules in python for the paper MTCC
please ensure you have implemented
1- image normalization
2- gabor filter enhancement
3- image segmentation
4- MTCC features
5- Minutea extraction
6- matchers. 
7- Visualizers for each step. Turned on and off via boolean
8- FVC dataset loading, with one separate function to test two image match

```

### Fail2 ClaudeAI-sonnet,Grok3

Getting the sequence of events such as image enhancement, segmentation and binarization before
minutia detection fro claude was a challenge.
Will test again in 6 months time!
Around 8 prompts used. Knowledge based with 3 fingerprint papers with MTCC

#### Prompt 

        
```
Given the research papers we need to implement MTCC. Generally I will outline what steps or functions are needed to get to MTCC features

1- fingerprint image loading
2- Image normalization
3- Segmentation
4- Gabor filter enhancements. Very important!
5- SMQT (Successive Mean Quantization Transform)
6- STFT Analysis for creating features for MTCC
7- Image binarization and thinning
8- Minutae extraction
7- Cylinder creation
8- Matching
9- EERs calculations
10- Two finger matching functions. 
11- Visualation for steps between step 1 and 8 in a single figure. 

If we can get this together in such  a way that the code is nead, clean, modular, small functions 
This will really help. Get things simple enough. Function by function. In a single file

```

### Fail3 OpenAI, GPT4o, 4.1 mini, o3

Much better code than other models. On first run the code actuall ran through but 
gave no output. Tried to fixed gabor filters issue and it kept failing and artefacts
did not go away. Binarization was ok, so was thinning but no minutiae detected.

Much better and cleaner codes compared to other models I may safely say!

#### Prompt

```

1- fingerprint image loading
2- Image normalization
3- Segmentation
4- Gabor filter enhancements. Very important!
5- SMQT (Successive Mean Quantization Transform)
6- STFT Analysis for creating features for MTCC
7- Image binarization and thinning
8- Minutae extraction
7- Cylinder creation
8- Matching
9- EERs calculations
10- Two finger matching functions. 
11- Visualation for steps between step 1 and 8 in a single figure. 

If we can get this together in such  a way that the code is nead, clean, modular, small functions 
This will really help. Get things simple enough. Function by function. In a single file

```

### Fail4-6 OpenAI, GPT4o, 4.1 mini, o3, Gemini 2.5flash, ClaudeAi (sonnet and opus)

Created a better prompt and changes some research papers. As of now, the issue remains circular. 
Get one thing done, the next breaks down.

The major issue remains STFT analysis and Gabor filtering, they faulter. So bad that it is a shame
that the models cannot still replace me :)

#### Prompt

```
**Implement a complete MTCC (Minutia Texture Cylinder Codes) fingerprint recognition system for FVC datasets in Python using only OpenCV, NumPy, and SciPy. Your code should strictly adhere to current academic research and state-of-the-art, specifically: \[Baig et al., 2018]\[11], \[Gottschlich, 2014]\[8], \[Shimna & Neethu, 2015]\[9], \[Bazen & Gerez, 2001]\[10].**

### 1. **Pipeline: Modular Functions (single Python file)**

Your code *must* modularize the following functions, named exactly as shown:

python
def load_image(path): ...
def normalize(img): ...                     # Zero-mean, unit variance normalization
def segment(img, block_size=16): ...        # Block-wise variance or coherence-based segmentation (see [10])
def gabor_enhance(img, orientation_map, freq_map): ... # Use curved or context-based Gabor filtering ([8][9])
def smqt(img, levels=8): ...                # Successive Mean Quantization Transform ([11])
def stft_features(img, window=16): ...      # 3-channel texture (orientation, frequency, energy) ([11][9])
def binarize_thin(img): ...                 # Adaptive threshold + Zhang-Suen thinning
def extract_minutiae(skeleton): ...         # Crossing Number (CN) algorithm
def create_cylinders(minutiae, texture_maps, radius=70): ... # MTCC: replace angular info with STFT textures ([11])
def match(cylinders1, cylinders2): ...      # Local similarity sort (LSS) or relaxation ([11])
def calculate_eer(genuine_scores, impostor_scores): ...
def visualize_pipeline(original, *steps): ... # Subplot of all major pipeline stages


### 2. **Implementation Details:**

* **Segmentation:** Implement using *either* blockwise variance (\[11]) or coherence, mean, and variance features with linear classifier (\[10]).
* **Gabor Enhancement:** Gabor filtering *must* be context-adaptive, using local orientation and frequency maps, preferably with curved or block-adapted Gabor filters (\[8]\[9]).
* **SMQT:** Apply *before* STFT to enhance low-contrast ridges (\[11]).
* **STFT Features:** Extract orientation, frequency, and energy maps using overlapping windows. Output three images—one for each feature. Use these as cell-level features in the cylinder code (\[11]).
* **MTCC Descriptor:** For each detected minutia, construct a 3D “cylinder” where angular cells store STFT-based orientation/frequency/energy (not minutia angles as in classic MCC) (\[11], Sec. VI).
* **Minutiae Extraction:** Use classic Crossing Number (CN) algorithm after thinning.
* **Matching:** Implement as in MTCC—use cell-wise similarity (LSS or relaxation). Output a single similarity score (\[11]).
* **Visualization:** 3x3 grid: Original, Normalized, Segmented, Gabor, SMQT, STFT, Binarized, Thinned, Minutiae overlay (\[11]).
* **Performance:** Use numpy vectorization. Optionally use sliding_window_view for efficient local STFT computation. Consider Cython for performance if needed (\[11]).
* **No external ML libraries.** Only OpenCV, NumPy, SciPy.

### 3. **Testing Protocol**

* *For each image* in the FVC dataset, run the complete pipeline and store resulting MTCC descriptors.
* *Matching:* Compute similarity for all genuine pairs and impostor pairs (as per FVC protocol).
* *EER Calculation:* Compute Equal Error Rate from genuine/impostor scores (\[11]).

### 4. **Deliverables**

* A **single, clean Python file** with all modular functions.
* No dependencies except OpenCV, NumPy, SciPy.
* **Visual debug mode**: shows pipeline steps and minutiae overlay.
* **Scriptable test harness** for FVC (show EER at the end).

---

#### **References:**

* \[8] Gottschlich, "Curved Gabor Filters for Fingerprint Image Enhancement", 2014.
* \[9] Shimna & Neethu, "Fingerprint Image Enhancement Using STFT Analysis", 2015.
* \[10] Bazen & Gerez, "Segmentation of Fingerprint Images", 2001.
* \[11] Baig et al., "Minutia Texture Cylinder Codes for fingerprint matching", 2018.

---

**Summary:**
This prompt mandates a full MTCC implementation, not standard MCC, *with texture-based descriptors derived from STFT as in the latest research*. Each pipeline step is explicitly referenced to primary research, making it unambiguous what algorithm must be implemented for each step, and what the expected input/output for each function should be.

Let me know if you want an even more detailed breakdown or code template!

```