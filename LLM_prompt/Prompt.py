import openai
import pandas as pd
import random

# Set your OpenAI API key
openai.api_key = 'xxxxxxxxxxxxxxx'  # Replace with your actual OpenAI API key


# List of gestures and emotions (choose either one, not a combination)
gestures_and_emotions = [
    "Hold Gesture", "Rub Gesture", "Tickle Gesture", "Poke Gesture",
    "Pat Gesture", "Tap Gesture", "Anger", "Attention", "Calming",
    "Comfort", "Confusion", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
]


# Step 1: Define the first prompt to think about gesture or emotion features
def prompt_for_features(gesture_or_emotion):
    prompt = f"""

    Think about how the {gesture_or_emotion} would affect sensor readings in a 5x5 grid of pressure sensors.
    You are tasked with generating realistic sensor data from a 5x5 grid (25 sensors) for the gesture '{gesture_or_emotion}'.
    This data represents pressure readings recorded at a frequency of 45Hz over 10 seconds (450 rows of data).


    Please describe the key features of the '{gesture_or_emotion}' that would affect the sensor readings:
    - How the pressure would evolve over time.
    - How the pressure would vary across the grid.
    - What type of motion this gesture would create on the sensor readings.

     Example analysis for the 'Rub Gesture':
       For the "Rub" gesture:
    1. The sensor readings should **shift in a wave-like pattern** across the 5x5 grid, representing the motion of rubbing.
    2. The sensor values should vary smoothly across neighboring sensors in each time step (spatial variation) and change gradually over time (temporal variation).
    3. Ensure that the pressure increases and decreases as the "rub" moves back and forth across the grid, affecting sensors in a smooth motion.

    please provide your analysis like the example above.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # You can also use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a expert of tactie sensor data generator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response['choices'][0]['message']['content']


# Step 2: Use those features to generate Python code
def prompt_for_python_code(gesture_or_emotion, analysis):
    prompt = f"""
    Based on the following analysis of the {gesture_or_emotion}:\n{analysis}

    Generate Python code that creates a DataFrame with sensor data for a 5x5 grid of sensors.
    The data should represent pressure readings recorded at 45Hz over 10 seconds (450 rows). 
    Each row has 25 values, where the first 5 columns correspond to the first row of the grid, 
    the next 5 columns correspond to the second row, and so on, please note these reflect the spartial information of the sensors.

   **Important**:
    - Ensure that any significant changes in pressure (such as peaks) are not isolated to a single row. 
    - Pressure changes should be **gradual** and **sustained** over several time steps to allow vibration motors to respond effectively.
    - Avoid single, sharp spikes that last only one time step, as they are difficult for vibration motors to perceive and respond to.
    - Pressure values should evolve smoothly over time, especially for peak values, which should last at least a few consecutive time steps. 
    - The neighboring sensors should have similar pressure values to reflect the smooth motion of the gesture.   


    The code should:
    1. Use NumPy to generate the sensor data.
    2. Use Pandas to create a DataFrame from the sensor data.
    3. Save the DataFrame to a CSV file.
    4. Ensure that the pressure evolves over time based on the features described for the {gesture_or_emotion}:{analysis}.
    5. All the sensor data should be between 0 and 255,  in integer format, no decimal values,
    please note 255 is the maximum pressure value, which mean the pressure is the highest.
    6. The CSV file should have a header row with column names like 'sensor_0_0', 'sensor_0_1', etc.
    7 please generate the pattern more obvious in the data, so that it is easier to analyze the data.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that generates full sensor data in CSV format, please only generate sensor data"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Adjust as needed
        temperature=1.0
    )
    return response['choices'][0]['message']['content'].strip()


def main():
    for gesture_or_emotion in gestures_and_emotions:
        print(f"Analyzing features for: {gesture_or_emotion}")

        # Step 1: Ask the LLM to think about features for the gesture or emotion
        analysis = prompt_for_features(gesture_or_emotion)
        print(f"Feature Analysis:\n{analysis}")

        # Step 2: Pass the analysis into the second prompt to generate Python code
        print(f"\nGenerating Python code based on the analysis...")
        python_code = prompt_for_python_code(gesture_or_emotion, analysis)
        print(f"Generated Python Code:\n{python_code}")

        # Save the generated code to a file
        filename = f"{gesture_or_emotion.split(':')[0].replace(' ', '_')}_sensor_code_gpt4.py"
        with open(filename, "w") as code_file:
            code_file.write(python_code)

        print(f"Python code saved to {filename}\n\n")


# Example usage
if __name__ == "__main__":
    main()
