# LLM_Vibration_Data

## Description

This repository contains data and visualizations generated from a study on tactile communication using Large Language Models (LLMs). The visualizations demonstrate distinct vibration patterns corresponding to 10 emotions and 6 touch gestures, which were generated using an LLM and mapped onto a 5x5 grid of vibration motors.

## Visualization Folder

The `visualization` folder contains two subfolders:

1. **LLM_Emotions**  
   This folder contains videos visualizing the vibration patterns for 10 distinct emotions.  
   - Each video represents one emotion.  
   - Each video lasts for 10 seconds.

2. **LLM_Gestures**  
   This folder contains videos visualizing the vibration patterns for 6 unique touch gestures.  
   - Each video represents one gesture.  
   - Each video lasts for 10 seconds.

---
## LLM_prompt Folder

The `LLM_prompt` folder contains a Python script for generating vibration data:

- Running the script `Prompt.py` will generate Python scripts for the 10 emotions and 6 touch gestures vibration data.
- To execute, use the following command:
  ```bash
  python3 Prompt.py

### Notes
- Make sure to explore both subfolders to view how the vibration patterns are represented for emotions and gestures.
- The videos aim to provide a clear visualization of the tactile signals generated by the LLM, showcasing their potential for conveying emotional and social touch.
