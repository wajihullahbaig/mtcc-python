# mtcc-python
Using June-2025, ClaudeAI-sonnet,Grok3, DeepSeek-V3 (non-thinking), Gemini Flash 2.5 (thinking) to implement a Fingerprint paper, MTCC (my paper from 2018)
As of this date, all of them have failed. None of them could produce correct code except for Claude
which corrected to a great level the orientation processing. 

### Fail1
Minutia seem to be detected but thinning, orientation images, freqyuency iamges was a struggle. 
30 odd prompt requests per model but no success. MTCC paper knowledge based provided

### Fail2 ClaudeAI-sonnet,Grok3

Getting the sequence of events such as image enhancement, segmentation and binarization before
minutia detection fro claude was a challenge.
Will test again in 6 months time!
Around 8 prompts used. Knowledge based with 3 fingerprint papers with MTCC


### Fail3 OpenAI, GPT4o, 4.1 mini, o3

Much better code than other models. On first run the code actuall ran through but 
gave no output. Tried to fixed gabor filters issue and it kept failing and artefacts
did not go away. Binarization was ok, so was thinning but no minutiae detected.

Much better and cleaner codes compared to other models I may safely say!


### Fail4-5 OpenAI, GPT4o, 4.1 mini, o3, Gemini 2.5flash, ClaudeAi (sonnet and opus)

Create a better prompt and changes some research papers. As of now, the issue remains circular. 
Get one thing done, the next breaks down.

The major issue remains STFT analysis and Gabor filtering, they faulter. So bad that it is a shame
that the models cannot still replace me :)
