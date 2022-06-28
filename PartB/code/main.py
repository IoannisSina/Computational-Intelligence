"""
Genetic algorithm implementation.
"""

import os
import pathlib
from helpers import (
    generate_population,
    get_tf_idf_mean,
    fitness,
    roulette_wheel_pair_selection,
    tournament_selection,
    single_point_crossover,
    mutation,
    plot_fitness,
    VOCAB_LENGTH,
)

DATA_PATH = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.resolve()) , "data")

# Parameters
GENERATIONS_TO_WAIT = 10
ELITISM = True
MAX_GENERATIONS = 1000
TIMES_TO_RUN = 10

mean_tf_idf = get_tf_idf_mean("sklearn_mean_tf_idf.dat")
best_genome_of_all, best_fitness_of_all = [] , -1
parameters = [ 
    (20, .6, .00),
    (20, .6, .01),
    (20, .6, .10),
    (20, .9, .01),
    (20, .1, .01),
    (200, .6, .00),
    (200, .6, .01),
    (200, .6, .10),
    (200, .9, .01),
    (200, .1, .01),
    ]
custom_parameters = [(300, .7, (1.0 / VOCAB_LENGTH))]

for parameter in parameters:
    population_size, pc, pm = parameter
    all_best_fitness_history = []
    mean_of_best_fitness = 0
    mean_generations = 0
    max_generation_reached = 0
    print("----------------------------Population size: {} | PC: {} | PM: {}----------------------------".format(population_size, pc, pm))

    for l in range(TIMES_TO_RUN):
        population = generate_population(population_size, VOCAB_LENGTH)
        best_genome, best_fitness = population[0], fitness(population[0], mean_tf_idf)
        generations_without_improvement = 0
        temp_fitness_history = []
        temp_generation_reached = 0

        for i in range(MAX_GENERATIONS):
            population = sorted(population, key=lambda x: fitness(x, mean_tf_idf), reverse=True)
            temp_fitness_history.append(fitness(population[0], mean_tf_idf))

            if generations_without_improvement == GENERATIONS_TO_WAIT:
                break

            # Check the difference between the current best fitness and the previous best fitness
            improvement = (fitness(population[0], mean_tf_idf) / best_fitness) * 100 - 100
            generations_without_improvement += 1
            if fitness(population[0], mean_tf_idf) > best_fitness:
                best_genome, best_fitness = population[0], fitness(population[0], mean_tf_idf)
                generations_without_improvement = 0

            # print("Generation: {} | Best fitness: {} | Number of words selected: {} | Improvement: {}%".format(i, best_fitness, sum(population[0]), improvement))

            next_generation = []

            if ELITISM:
                next_generation = population[0:2]
                population = population[2:]

            parents = tournament_selection(population, mean_tf_idf)
            for parent1, parent2 in zip(parents[0::2], parents[1::2]):
                offspring_a, offspring_b = single_point_crossover(parent1, parent2, pc)
                offspring_a = mutation(offspring_a, pm)
                offspring_b = mutation(offspring_b, pm)
                next_generation += [offspring_a, offspring_b]

            population = next_generation
            assert len(population) == population_size, "Error: len(population) != POPULATION_SIZE"
            if i > max_generation_reached: max_generation_reached = i
            if i > temp_generation_reached: temp_generation_reached = i

        all_best_fitness_history.append(temp_fitness_history)
        mean_of_best_fitness += best_fitness
        mean_generations += temp_generation_reached
        print("Loop number {}  |  Best genome has {} words selected and a fitness of {}".format(l, sum(best_genome), best_fitness))

        # Keep the best genome of all runs
        if best_fitness > best_fitness_of_all:
            best_genome_of_all, best_fitness_of_all = best_genome, best_fitness

    generations = max_generation_reached + 2
    # Fill all histories with theit last non zero value, until the maximum generation
    for fit_history in all_best_fitness_history:
        temp_last_fitness = fit_history[-1]
        fit_history += [temp_last_fitness] * (generations - len(fit_history))
   
    # Check that all histories have the same length
    for h in range(len(all_best_fitness_history)):
        assert len(all_best_fitness_history[h]) == generations, "Error: len(all_best_fitness_history[h]) != generations"

    mean_of_best_fitness /= TIMES_TO_RUN
    mean_generations /= TIMES_TO_RUN
    mean_of_all_fitness_history = [sum(x) / len(x) for x in zip(*all_best_fitness_history)]
    plot_fitness(mean_of_all_fitness_history, generations, population_size, pc, pm)
    print("Mean of best fitness: {}".format(mean_of_best_fitness))
    print("Mean of generations: {}".format(int(mean_generations)))
    print("--------------------------------------------------------------------------------------------------------------------")

# Write best genome to file
assert len(best_genome_of_all) == VOCAB_LENGTH, "Error: len(best_genome_of_all) != VOCAB_LENGTH"
with open(os.path.join(DATA_PATH, "best_genome_from_genetic.dat"), "w") as f:
    f.write(str(best_fitness_of_all) + ", " + ' '.join(str(bit) for bit in best_genome_of_all))
