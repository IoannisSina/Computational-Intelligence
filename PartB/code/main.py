"""
Genetic algorithm implementation.
"""

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

# Parameters
GENERATIONS_TO_WAIT = 10
ELITISM = True
MAX_GENERATIONS = 600
TIMES_TO_RUN = 10

mean_tf_idf = get_tf_idf_mean("sklearn_mean_tf_idf.dat")
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

# custom_parameters = []
for parameter in parameters:
    population_size, pc, pm = parameter
    all_best_fitness_list = [0] * MAX_GENERATIONS
    mean_of_best_fitness = 0
    max_generation_reached = 0
    print("----------------------------Population size: {} | PC: {} | PM: {}----------------------------".format(population_size, pc, pm))

    for l in range(TIMES_TO_RUN):
        population = generate_population(population_size, VOCAB_LENGTH)
        best_genome, best_fitness = population[0], fitness(population[0], mean_tf_idf)
        generations_without_improvement = 0

        for i in range(MAX_GENERATIONS):
            population = sorted(population, key=lambda x: fitness(x, mean_tf_idf), reverse=True)

            if generations_without_improvement == GENERATIONS_TO_WAIT:
                break

            # Check the difference between the current best fitness and the previous best fitness
            improvement = (fitness(population[0], mean_tf_idf) / best_fitness) * 100 - 100
            generations_without_improvement += 1
            if fitness(population[0], mean_tf_idf) > best_fitness:
                best_genome, best_fitness = population[0], fitness(population[0], mean_tf_idf)
                generations_without_improvement = 0

            # print("Generation: {} | Best fitness: {} | Number of words selected: {} | Improvement: {}%".format(i, best_fitness, sum(population[0]), improvement))

            all_best_fitness_list[i] += best_fitness
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

        mean_of_best_fitness += best_fitness
        print("Loop number {}  |  Best genome has {} words selected and a fitness of {}".format(l, sum(best_genome), best_fitness))
    mean_of_best_fitness /= TIMES_TO_RUN
    all_best_fitness_list = [x / TIMES_TO_RUN for x in all_best_fitness_list]
    plot_fitness(all_best_fitness_list[:max_generation_reached], population_size, pc, pm)
    print("Mean of best fitness: {}".format(mean_of_best_fitness))
    print("--------------------------------------------------------------------------------------------------------------------")
