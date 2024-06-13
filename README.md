# Genetic-Algorithm-for-Biometric-Image-Encryption-Optimization
This project focuses on optimizing the parameters of a genetic algorithm (GA) to generate encrypted biometric images. The aim is to create cancelable biometric templates with high accuracy by fine-tuning the GA's crossover and mutation rates.



<!DOCTYPE html>
<html lang="en">
<body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
    <h1 style="color: #333;">Genetic Algorithm for Biometric Image Encryption Optimization</h1>
    <p>This project focuses on optimizing the parameters of a genetic algorithm (GA) to generate encrypted biometric images. The aim is to create cancelable biometric templates with high accuracy by fine-tuning the GA's crossover and mutation rates.</p>
    <h2 style="color: #333;">Table of Contents</h2>
    <ul style="padding-left: 20px;">
        <li><a href="#project-overview">Project Overview</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <li><a href="#functions-description">Functions Description</a></li>
        <li><a href="#acknowledgements">Acknowledgements</a></li>
    </ul>
    <h2 id="project-overview" style="color: #333;">Project Overview</h2>
    <p>The project involves the following steps:</p>
    <ol style="padding-left: 20px;">
        <li><strong>Generate Initial Population:</strong> Create an initial population of encrypted biometric images using multiple rounds of XOR encryption with random keys.</li>
        <li><strong>Optimize GA Parameters:</strong> Use the genetic algorithm to optimize the crossover and mutation rates for generating cancelable biometric templates.</li>
        <li><strong>Evaluate Accuracy:</strong> Evaluate the accuracy of the cancelable templates generated by the GA.</li>
    </ol>
    <h2 id="prerequisites" style="color: #333;">Prerequisites</h2>
    <ul style="padding-left: 20px;">
        <li>Python 3.x</li>
        <li>OpenCV</li>
        <li>NumPy</li>
    </ul>
    <h2 id="installation" style="color: #333;">Installation</h2>
    <ol style="padding-left: 20px;">
        <li>Clone the repository:
            <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto;"><code>git clone https://github.com/JayArcher9/TRWI.git
cd TRWI</code></pre>
        </li>
        <li>Install the required Python packages:
            <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto;"><code>pip install opencv-python numpy</code></pre>
        </li>
    </ol>
    <h2 id="usage" style="color: #333;">Usage</h2>
    <ol style="padding-left: 20px;">
        <li>Place the original biometric image in the project directory. The image should be named <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">download.jpg</code>.</li>
        <li>Run the main script:
            <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto;"><code>python main.py</code></pre>
        </li>
        <li>The script will generate initial encrypted images and optimize the GA parameters, printing the accuracy scores for each combination of crossover and mutation rates.</li>
    </ol>
    <h2 id="project-structure" style="color: #333;">Project Structure</h2>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">main.py</code>: Main script to execute the project.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">generate_initial_population</code>: Function to generate initial population images using multiple rounds of encryption.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">optimize_ga_parameters</code>: Function to optimize GA parameters (crossover and mutation rates).</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">evaluate_population</code>, <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">select_parents</code>, <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">crossover_and_mutate</code>, <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">crossover</code>, <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">mutate</code>, <code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">next_generation</code>: Helper functions for GA operations.</li>
    </ul>
    <h2 id="functions-description" style="color: #333;">Functions Description</h2>
    <h3><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">generate_initial_population(original_biometric_image, num_images, num_rounds)</code></h3>
    <p>Generates an initial population of encrypted biometric images using XOR encryption.</p>
    <p><strong>Parameters:</strong></p>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">original_biometric_image</code>: The original biometric image.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">num_images</code>: Number of initial population images to generate.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">num_rounds</code>: Number of encryption rounds.</li>
    </ul>
    <p><strong>Returns:</strong></p>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">initial_population</code>: List of encrypted images.</li>
    </ul>
    <h3><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">optimize_ga_parameters(initial_population)</code></h3>
    <p>Optimizes the genetic algorithm parameters (crossover and mutation rates) to find the best cancelable template.</p>
    <p><strong>Parameters:</strong></p>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">initial_population</code>: Initial population of encrypted images.</li>
    </ul>
    <p><strong>Returns:</strong></p>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">accuracy_scores</code>: Dictionary containing accuracy scores for each combination of crossover and mutation rates.</li>
    </ul>
    <h3 style="color: #333;">Helper Functions</h3>
    <ul style="padding-left: 20px;">
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">evaluate_population(population)</code>: Evaluates the fitness of each cancelable template in the population.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">select_parents(population, fitness_scores)</code>: Selects parent cancelable templates for reproduction based on fitness scores.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">crossover_and_mutate(parents, crossover_rate, mutation_rate)</code>: Performs crossover and mutation operations on the parent cancelable templates.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">crossover(parent1, parent2)</code>: Performs crossover operation on two parent cancelable templates.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">mutate(individual, mutation_rate)</code>: Performs mutation operation on a cancelable template.</li>
        <li><code style="background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;">next_generation(population, offspring, fitness_scores, offspring_fitness)</code>: Combines the offspring and parents, and selects the fittest individuals for the next generation.</li>
    </ul>
    <h2 id="acknowledgements" style="color: #333;">Acknowledgements</h2>
    <p>This project was developed using Python, OpenCV, and NumPy. Special thanks to the open-source community for providing valuable resources and libraries.</p>
</body>
</html>
