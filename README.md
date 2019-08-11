# Components Evolution Map

### Introduction

This project is an implementation of the components evolution map method, as described in the paper "Evolution maps and applications" written by Ofer Biller, Irina Rabaev, Klara Kedem, Its'hak Dinstein, and Jihad J. El-Sana.

In order to implement it in Python, I used both the description given in the article, and a code in MATLAB which was used originally to implement this method.

### Method Explanation
The components evolution map is a method which helps to extract features from a corrupted or highly degraded images such as historic documents. Among the features that can be extracted are: letter width, letter height, letter stroke.

The goal is to make the process of extracting these features automatic, and use them for text-recognition algorithms.

### Project Structure
The project is composed out of 2 main classes: <br />
1. EvolutionMapBuilder - The class that is in charge of building the evolution map from an image that is given to it. It constructs the evolution map in the form of a 2d-nd-array using numpy's arrays.
2. EvolutionMapAnalyser - The class the is in charge of analysing the evolution map. Essentially, it recognizes the blobs in the evolution map, and gives them a score so that the blob with the maximal score is likely to indicate the feature that we want to extract.

### Running the Project
To run the program, you can use the main functions already implemented the classes' files, which reads the relevant files and give the relevant parameters to the classes.

Another option is to create a simple python file which constructs an evolution map using the build_em method of EvolutionMapBuilder, and then give the result to the EvolutionMapAnalyser.

### Notes
For debugging purposes, during the run the intermediate results can be written to the "res" and "threshed" folders. Use it for your own convenience.

"threshed" contains the image with each threshold value, and "res" contains the components found (marked in a black rectangle) for each threshold value.  