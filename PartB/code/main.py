"""
Genetic algorithm implementation.
"""

from helpers import (
    generate_population,
    get_tf_idf_mean,
    fitness,
    roulette_wheel_pair_selection,
    single_point_crossover,
    mutation,
    VOCAB_LENGTH,
)

# Parameters
ELITISM = False
MAX_GENERATIONS = 200
POPULATION_SIZE = 300
PC = .7
PM = .1

population = generate_population(POPULATION_SIZE, VOCAB_LENGTH)
mean_tf_idf = get_tf_idf_mean("sklearn_mean_tf_idf.dat")

for i in range(MAX_GENERATIONS):
    population = sorted(population, key=lambda x: fitness(x, mean_tf_idf), reverse=True)

    print("Generation:", i)
    print("Best fitness:", fitness(population[0], mean_tf_idf))

    next_generation = []

    if ELITISM:
        pass

    parents = roulette_wheel_pair_selection(population, mean_tf_idf, POPULATION_SIZE)
    for parent1, parent2 in zip(parents[0::2], parents[1::2]):
        offspring_a, offspring_b = single_point_crossover(parent1, parent2, PC)
        offspring_a = mutation(offspring_a, PM)
        offspring_b = mutation(offspring_b, PM)
        next_generation += [offspring_a, offspring_b]

    population = next_generation
    assert len(population) == POPULATION_SIZE, "Error: len(population) != POPULATION_SIZE"
