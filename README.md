# README

## Setup

1. create a virtual environment by running `python -m venv .venv`
2. activate virtual environment by running `source .venv/bin/activate`
(if on windows look up the command to activate the environment)
3. install the dependencies by running `pip install -r requirements.txt`

To view the marimo notebook use `marimo run eda.py`

## Notes TODO

Collection of ideas for the project.

### 1. The First Step: Data Preparation and Contributor Definition

- *Go In-Depth with Data Preparation:* The first step is to "prepare" the data and "go in depth" to understand its characteristics, including the distribution of data over time (e.g., daily, weekly).
- *Identify Characteristics and Variables:* This involves looking at the characteristics and variables inside the data.
- *Determine Contribution:* It's important to determine the number of subjects who provided "zero answer, one answer, two answers, and etc." to establish exactly who is contributing.
- *Use Notification Count and Days:* To define a contributor, you need to consider two steps: the *number of daysthe subject answered and the **number of notifications answered per day*.
    - Simply looking at the total number of notifications answered is insufficient because it doesn't distinguish between a subject who answered 100 notifications over two days and one who answered 10 per day for 10 days.
- *Set a Threshold:* A threshold for the minimum number of notifications per day must be set based on the "research hypothesis" to justify who counts as a contributor.
- *Separate Groups:* The first step is to establish who is a "contributor or not," resulting in distinct groups:
    - *Contributors:* People who are "very precise" in their answering behavior.
    - *Non-Contributors:* People who "don't contribute".
    - *Intermediate Group:* People who contribute but with a "very low or very loud" frequency.

### 2. The Second Step: Focusing on Data Richness and Behavior

- *Focus on the Richness of Data:* The second step is to "focus on the richness of your data".
- *Avoid Collapsing Prematurely:* Collapsing the data at the individual level (e.g., using only total time) can cause you to "lose absolutely the richness" and make the underlying differences and patterns "disappear".
- *Analyze Homogeneous Groups:* Once a contributor group is defined (the "homogeneous population"), you can compare their behaviors.
- *Analyze Behavior in Detail:* For the contributor group, you should go in depth on other characteristics, such as:
    - *Time Lapse:* The time lapse from when a notification was received to when they started to answer.
    - *Duration:* The duration of the answer.
    - *Behavioral Patterns:* Discovering specific patterns in how people manage their contribution (e.g., answering immediately vs. integrating it into a daily routine).
- *Examine Specific Data Examples (Music):*
    - For specific data like music, instead of simply collapsing the total time, you could use survival modeling to estimate the *time of music listening*.
    - Analyze correlations with other variables like *where they are* and *with whom they are*.
    - The *mood* variable should be the last step in the analysis, as it can be influenced by multiple factors, and causality is difficult to determine (e.g., are you sad and listening to music, or listening to music and feeling sad?).
- *Leverage All Information:* The overall approach is to "leverage the richness of the information". The final goal is to use the clean, homogeneous data to see how specific sociodemographic or psychological characteristics influence the probability of a certain behavior.
